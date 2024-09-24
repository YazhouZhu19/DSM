"""
encoder resnet50, GNN geometrical feature extractor, feature transform operation
"""
import random

import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder_1 import Res50Encoder
from .detection_head import *



class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(-1)).view(b, c)
        y = self.fc(y).view(b, c)
        return x * y



class FewShotSeg(nn.Module):

    def __init__(self, args):
        super().__init__()

        # Encoder
        self.encoder = Res50Encoder(replace_stride_with_dilation=[True, True, False],
                                    pretrained_weights="COCO")  # or "ImageNet"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.scaler = 20.0
        self.args = args
        self.reference_layer1 = nn.Linear(512, 2, bias=True)
        self.epsilon_list = [0.01, 0.03, 0.05, 0.001, 0.003, 0.005]
        self.fg_sampler = np.random.RandomState(1289)


        self.criterion = nn.NLLLoss(ignore_index=255, weight=torch.FloatTensor([0.1, 1.0]).cuda())
        self.margin = 0.

        self.channel = nn.Conv2d(in_channels=507, out_channels=512, kernel_size=1, stride=1)
        self.mse_loss = nn.MSELoss()

        # self.function_layer = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)  # the f function
        self.function_layer = nn.Linear(1024, 512)  # the f function 
        self.att = ChannelAttention(in_channels=512)

        self.sampling_reshape_1 = nn.Linear(256, 512)
        self.sampling_reshape_2 = nn.Linear(128, 512)
        self.sampling_reshape_3 = nn.Linear(64, 512)



    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, opt, train=False):

        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors  (1, 3, 257, 257)
            qry_mask: label
                N x 2 x H x W, tensor
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W
        ## Feature Extracting With ResNet Backbone
        # # Extract features #
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)

        img_fts, tao = self.encoder(imgs_concat)

        supp_fts = img_fts[:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts.shape[-2:])
        qry_fts = img_fts[self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts.shape[-2:])

        # Get threshold #
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        self.t_ = tao[:self.n_ways * self.n_shots * supp_bs]  # t for support features
        self.thresh_pred_ = [self.t_ for _ in range(self.n_ways)]

        outputs_qry = []
        loss_wt_spt_1 = torch.zeros(1).to(self.device)
        loss_wt_qry_1 = torch.zeros(1).to(self.device)
        loss_wt_spt_2 = torch.zeros(1).to(self.device)
        loss_wt_qry_2 = torch.zeros(1).to(self.device)
        for epi in range(supp_bs):

            if supp_mask[[0], 0, 0].max() > 0.:

                spt_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                             for shot in range(self.n_shots)] for way in range(self.n_ways)]
                spt_fg_proto = self.getPrototype(spt_fts_)

                supp_fts_b = [[self.getFeatures(supp_fts[[epi], way, shot], 1. - supp_mask[[epi], way, shot])
                               for shot in range(self.n_shots)] for way in range(self.n_ways)]
                spt_bg_proto = self.getPrototype(supp_fts_b)

                qry_pred = torch.stack(
                        [self.getPred(qry_fts[way], spt_fg_proto[way], self.thresh_pred[way])
                         for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'
                qry_pred_coarse = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)
                qry_fts_ = [[self.getFeatures(qry_fts[way], qry_pred_coarse[epi]) for way in range(self.n_ways)]]
                qry_fg_proto = self.getPrototype(qry_fts_)

                if train:

                    # ******************** perturbation signal formulation **************** #
                    # signal a
                    spt_fg_proto_learnable_a = [torch.nn.Parameter(spt_fg_proto[way]) for way in range(self.n_ways)]
                    spt_fg_proto_learnable_a = [spt_fg_proto_learnable_a[way].requires_grad_() for way in
                                                range(self.n_ways)]
                    intra_class_loss = self.mse_loss(spt_fg_proto_learnable_a[0], qry_fg_proto[0])

                    # signal b
                    spt_fg_proto_learnable_b = [torch.nn.Parameter(spt_fg_proto[way]) for way in range(self.n_ways)]
                    spt_fg_proto_learnable_b = [spt_fg_proto_learnable_b[way].requires_grad_() for way in
                                                range(self.n_ways)]
                    bg_prototypes = [[self.compute_multiple_background_prototypes(
                        5, supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], self.fg_sampler)
                        for shot in range(self.n_shots)] for way in range(self.n_ways)]
                    inter_class_loss = self.infoNCE(qry_fg_proto[0], spt_fg_proto_learnable_b[0],
                                                    bg_prototypes[0][0][0])
                    opt.zero_grad()
                    intra_class_loss.backward(retain_graph=True)
                    inter_class_loss.backward(retain_graph=True)

                    grad_spt_fg_proto_learnable_a = [spt_fg_proto_learnable_a[way].grad.detach() for way in
                                                     range(self.n_ways)]
                    grad_spt_fg_proto_learnable_b = [spt_fg_proto_learnable_b[way].grad.detach() for way in
                                                     range(self.n_ways)]
                    # ******************************************************************* #

                    # ************************ Two-stage Attack ************************* #
                    # index = torch.randint(0, len(self.epsilon_list), (1,))[0]
                    # epsilon = self.epsilon_list[index]
                    selected_values = random.choices(self.epsilon_list, k=512)
                    epsilon = torch.tensor(selected_values, dtype=torch.float32).unsqueeze(0).to(self.device)

                    # *** the first stage attack *** #
                    adv_spt_fg_proto = [self.fgsm_attack(spt_fg_proto[way], epsilon,
                                                         grad_spt_fg_proto_learnable_a[way],
                                                         grad_spt_fg_proto_learnable_b[way])
                                        for way in range(self.n_ways)]
                    adv_qry_fg_proto = [self.fgsm_attack(qry_fg_proto[way], epsilon,
                                                         grad_spt_fg_proto_learnable_a[way],
                                                         grad_spt_fg_proto_learnable_b[way])
                                        for way in range(self.n_ways)]
                    # the first stage whitening
                    loss_wt_spt_1 = self.Whitening(adv_spt_fg_proto[0], spt_fg_proto[0])
                    loss_wt_qry_1 = self.Whitening(adv_qry_fg_proto[0], qry_fg_proto[0])
            
                    fg_proto_inter = [torch.cat([adv_spt_fg_proto[way], adv_qry_fg_proto[way]], dim=1)
                                      for way in range(self.n_ways)]
                    fg_proto_inter = [self.function_layer(fg_proto_inter[way]) for way in range(self.n_ways)]
                    proto_first = [self.att(fg_proto_inter[way]) for way in range(self.n_ways)]
                    
                    proto_sampling_1 = [proto_first[way][:, torch.randperm(512)[:256]] for way in range(self.n_ways)]
                    proto_sampling_1 = [self.sampling_reshape_1(proto_sampling_1[way]) for way in range(self.n_ways)]
                
                    proto_sampling_2 = [proto_first[way][:, torch.randperm(512)[:128]] for way in range(self.n_ways)]
                    proto_sampling_2 = [self.sampling_reshape_2(proto_sampling_2[way]) for way in range(self.n_ways)]

                    proto_sampling_3 = [proto_first[way][:, torch.randperm(512)[:64]] for way in range(self.n_ways)]
                    proto_sampling_3 = [self.sampling_reshape_3(proto_sampling_3[way]) for way in range(self.n_ways)]
       
                    
                    # *** the second stage attack *** #
                    adv_sampling_proto_1 = [self.fgsm_attack(proto_sampling_1[way], epsilon,
                                                         grad_spt_fg_proto_learnable_a[way],
                                                         grad_spt_fg_proto_learnable_b[way])
                                        for way in range(self.n_ways)]
                    adv_sampling_proto_2 = [self.fgsm_attack(proto_sampling_2[way], epsilon,
                                                         grad_spt_fg_proto_learnable_a[way],
                                                         grad_spt_fg_proto_learnable_b[way])
                                        for way in range(self.n_ways)]
                    adv_sampling_proto_3 = [self.fgsm_attack(proto_sampling_3[way], epsilon,
                                                         grad_spt_fg_proto_learnable_a[way],
                                                         grad_spt_fg_proto_learnable_b[way])
                                        for way in range(self.n_ways)]
                    
                    adv_spt_proto_two_1 = [torch.cat((adv_spt_fg_proto[way], adv_sampling_proto_1[way]), dim=1) for way in range(self.n_ways)] 
                    adv_spt_proto_two_1 = [self.function_layer(adv_spt_proto_two_1[way]) for way in range(self.n_ways)]
                    adv_spt_proto_two_2 = [torch.cat((adv_spt_fg_proto[way], adv_sampling_proto_2[way]), dim=1) for way in range(self.n_ways)] 
                    adv_spt_proto_two_2 = [self.function_layer(adv_spt_proto_two_2[way]) for way in range(self.n_ways)]
                    adv_spt_proto_two_3 = [torch.cat((adv_spt_fg_proto[way], adv_sampling_proto_3[way]), dim=1) for way in range(self.n_ways)]
                    adv_spt_proto_two_3 = [self.function_layer(adv_spt_proto_two_3[way]) for way in range(self.n_ways)]

                    proto_spt_inter_second = [(adv_spt_proto_two_1[way] + adv_spt_proto_two_2[way] + adv_spt_proto_two_3[way]) for way in range(self.n_ways)] 
                    adv_spt_proto_second = [self.att(proto_spt_inter_second[way]) for way in range(self.n_ways)] 

                    adv_qry_proto_two_1 = [torch.cat((adv_qry_fg_proto[way], adv_sampling_proto_1[way]), dim=1) for way in range(self.n_ways)]
                    adv_qry_proto_two_1 = [self.function_layer(adv_qry_proto_two_1[way]) for way in range(self.n_ways)]
                    adv_qry_proto_two_2 = [torch.cat((adv_qry_fg_proto[way], adv_sampling_proto_2[way]), dim=1) for way in range(self.n_ways)]
                    adv_qry_proto_two_2 = [self.function_layer(adv_qry_proto_two_2[way]) for way in range(self.n_ways)]
                    adv_qry_proto_two_3 = [torch.cat((adv_qry_fg_proto[way], adv_sampling_proto_3[way]), dim=1) for way in range(self.n_ways)]
                    adv_qry_proto_two_3 = [self.function_layer(adv_qry_proto_two_3[way]) for way in range(self.n_ways)]
                    
                    proto_qry_inter_second = [(adv_qry_proto_two_1[way] + adv_qry_proto_two_2[way] + adv_qry_proto_two_3[way]) for way in range(self.n_ways)] 
                    adv_qry_proto_second = [self.att(proto_qry_inter_second[way]) for way in range(self.n_ways)]

                    # the second stage whitening
                    loss_wt_spt_2 = self.Whitening(adv_spt_proto_second[0], spt_fg_proto[0])
                    loss_wt_qry_2 = self.Whitening(adv_qry_proto_second[0], qry_fg_proto[0])
                    
                    proto_second_ = [torch.cat((adv_spt_proto_second[way], adv_qry_proto_second[way]), dim=1) for way in range(self.n_ways)]
                    proto_second = [self.att(self.function_layer(proto_second_[way])) for way in range(self.n_ways)] 

                    qry_pred = torch.stack([self.getPred(qry_fts[epi], proto_second[way], self.thresh_pred[way]) for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'
                    qry_pred_up = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)
                    preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)
                    outputs_qry.append(preds)

                else:
                    fg_proto_inter = [torch.cat([spt_fg_proto[way], qry_fg_proto[way]], dim=1)
                                      for way in range(self.n_ways)]
                    fg_proto_inter = [self.function_layer(fg_proto_inter[way]) for way in range(self.n_ways)]
                    proto_first = [self.att(fg_proto_inter[way]) for way in range(self.n_ways)]

                    proto_sampling_1 = [proto_first[way][:, torch.randperm(512)[:256]] for way in range(self.n_ways)]
                    proto_sampling_1 = [self.sampling_reshape_1(proto_sampling_1[way]) for way in range(self.n_ways)]
                    proto_sampling_2 = [proto_first[way][:, torch.randperm(512)[:128]] for way in range(self.n_ways)]
                    proto_sampling_2 = [self.sampling_reshape_2(proto_sampling_2[way]) for way in range(self.n_ways)]
                    proto_sampling_3 = [proto_first[way][:, torch.randperm(512)[:64]] for way in range(self.n_ways)]
                    proto_sampling_3 = [self.sampling_reshape_3(proto_sampling_3[way]) for way in range(self.n_ways)]

                    spt_fg_proto_two_1 = [torch.cat((spt_fg_proto[way], proto_sampling_1[way]), dim=1) for way in range(self.n_ways)] 
                    spt_fg_proto_two_1 = [self.function_layer(spt_fg_proto_two_1[way]) for way in range(self.n_ways)]
                    spt_fg_proto_two_2 = [torch.cat((spt_fg_proto[way], proto_sampling_2[way]), dim=1) for way in range(self.n_ways)] 
                    spt_fg_proto_two_2 = [self.function_layer(spt_fg_proto_two_2[way]) for way in range(self.n_ways)]
                    spt_fg_proto_two_3 = [torch.cat((spt_fg_proto[way], proto_sampling_3[way]), dim=1) for way in range(self.n_ways)]
                    spt_fg_proto_two_3 = [self.function_layer(spt_fg_proto_two_3[way]) for way in range(self.n_ways)]

                    proto_spt_inter = [(spt_fg_proto_two_1[way] + spt_fg_proto_two_2[way] + spt_fg_proto_two_3[way]) for way in range(self.n_ways)] 
                    proto_spt = [self.att(proto_spt_inter[way]) for way in range(self.n_ways)]

                    qry_fg_proto_two_1 = [torch.cat((qry_fg_proto[way], proto_sampling_1[way]), dim=1) for way in range(self.n_ways)]
                    qry_fg_proto_two_1 = [self.function_layer(qry_fg_proto_two_1[way]) for way in range(self.n_ways)]
                    qry_fg_proto_two_2 = [torch.cat((qry_fg_proto[way], proto_sampling_2[way]), dim=1) for way in range(self.n_ways)]
                    qry_fg_proto_two_2 = [self.function_layer(qry_fg_proto_two_2[way]) for way in range(self.n_ways)]
                    qry_fg_proto_two_3 = [torch.cat((qry_fg_proto[way], proto_sampling_3[way]), dim=1) for way in range(self.n_ways)]
                    qry_fg_proto_two_3 = [self.function_layer(qry_fg_proto_two_3[way]) for way in range(self.n_ways)]

                    proto_qry_inter = [(qry_fg_proto_two_1[way] + qry_fg_proto_two_2[way] + qry_fg_proto_two_3[way]) for way in range(self.n_ways)] 
                    proto_qry = [self.att(proto_qry_inter[way]) for way in range(self.n_ways)]

                    proto = [torch.cat((proto_spt[way], proto_qry[way]), dim=1) for way in range(self.n_ways)]
                    proto = [self.att(self.function_layer(proto[way])) for way in range(self.n_ways)] 

                    qry_pred = torch.stack([self.getPred(qry_fts[epi], proto[way], self.thresh_pred[way]) for way in range(self.n_ways)], dim=1)
                    qry_pred_up = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)
                    preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)
                    outputs_qry.append(preds)  
                    
                ########################################################################

            else:
                ########################acquiesce prototypical network#################
                supp_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                              for shot in range(self.n_shots)] for way in range(self.n_ways)]
                fg_prototypes = self.getPrototype(supp_fts_)  # the coarse foreground

                qry_pred = torch.stack(
                    [self.getPred(qry_fts[epi], fg_prototypes[way], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'
                ########################################################################

                # Combine predictions of different feature maps #
                qry_pred_up = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)
                preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)

                outputs_qry.append(preds)

        output_qry = torch.stack(outputs_qry, dim=1)
        output_qry = output_qry.view(-1, *output_qry.shape[2:])

        return output_qry, loss_wt_spt_1 / supp_bs, loss_wt_qry_1 / supp_bs, loss_wt_spt_2 / supp_bs, loss_wt_qry_2 / supp_bs

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes

    def compute_multiple_background_prototypes(self, bg_num, sup_fts, sup_fg, sampler):
        """
        Parameters
        ----------
        bg_num: int
            Background partition numbers
        sup_fts: torch.Tensor
             [B, C, h, w], float32
        sup_fg: torch. Tensor
             [B, h, w], float32 (0,1)
        sampler: np.random.RandomState

        Returns
        -------
        bg_proto: torch.Tensor
            [B, k, C], where k is the number of background proxies
        """

        B, C, h, w = sup_fts.shape
        bg_mask = F.interpolate(1 - sup_fg.unsqueeze(0), size=sup_fts.shape[-2:], mode='bilinear')
        bg_mask = bg_mask.squeeze(0).bool()  # [B, h, w] --> bool
        batch_bg_protos = []

        for b in range(B):
            bg_protos = []

            bg_mask_i = bg_mask[b]  # [h, w]

            # Check if zero
            with torch.no_grad():
                if bg_mask_i.sum() < bg_num:
                    bg_mask_i = bg_mask[b].clone()  # don't change original mask
                    bg_mask_i.view(-1)[:bg_num] = True

            # Iteratively select farthest points as centers of background local regions
            all_centers = []
            first = True
            pts = torch.stack(torch.where(bg_mask_i), dim=1)
            for _ in range(bg_num):
                if first:
                    i = sampler.choice(pts.shape[0])
                    first = False
                else:
                    dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                    # choose the farthest point
                    i = torch.argmax((dist ** 2).sum(-1).min(1)[0])
                pt = pts[i]  # center y, x
                all_centers.append(pt)

            # Assign bg labels for bg pixels
            dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
            bg_labels = torch.argmin((dist ** 2).sum(-1), dim=1)

            # Compute bg prototypes
            bg_feats = sup_fts[b].permute(1, 2, 0)[bg_mask_i]  # [N, C]
            for i in range(bg_num):
                proto = bg_feats[bg_labels == i].mean(0)  # [C]
                bg_protos.append(proto)

            bg_protos = torch.stack(bg_protos, dim=1)  # [C, k]
            batch_bg_protos.append(bg_protos)
        bg_proto = torch.stack(batch_bg_protos, dim=0).transpose(1, 2)  # [B, k, C]

        return bg_proto

    def infoNCE(self, query_fg_proto, support_fg_proto, support_bg_protos, temperature=0.07):
        """
        计算自定义的InfoNCE loss

        参数:
        query_fg_proto: 形状为 [1, D] 的张量，表示query的前景原型
        support_fg_proto: 形状为 [1, D] 的张量，表示support的前景原型
        support_bg_protos: 形状为 [N, D] 的张量，表示N个support的背景原型
        temperature: 温度参数，控制分布的平滑程度

        返回:
        loss: 标量，InfoNCE loss
        """
        # 确保输入维度正确
        assert query_fg_proto.shape == (1, 512), "query_fg_proto should be [1, 512]"
        assert support_fg_proto.shape == (1, 512), "support_fg_proto should be [1, 512]"
        assert support_bg_protos.shape[1] == 512, "support_bg_protos should be [N, 512]"

        # 归一化特征
        query_fg_proto = F.normalize(query_fg_proto, dim=1)
        support_fg_proto = F.normalize(support_fg_proto, dim=1)
        support_bg_protos = F.normalize(support_bg_protos, dim=1)

        # 计算正样本的相似度
        pos_similarity = torch.sum(query_fg_proto * support_fg_proto) / temperature

        # 计算负样本的相似度
        neg_similarities = torch.matmul(query_fg_proto, support_bg_protos.T) / temperature

        # 构建 logits
        logits = torch.cat([pos_similarity.unsqueeze(0), neg_similarities.squeeze(0)])

        # 创建标签：第一个（索引0）是正样本，其余都是负样本
        labels = torch.zeros(1, dtype=torch.long, device=query_fg_proto.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits.unsqueeze(0), labels)

        return loss

    def fgsm_attack(self, init_input, epsilon, data_grad_a, data_grad_b):

        START_EPS = 16 / 255  # should be defined outside the function
        init_input = init_input + torch.empty_like(init_input).uniform_(START_EPS, START_EPS)
        sign_data_grad = data_grad_a.sign() + data_grad_b.sign()   # fusing perturbation

        adv_input = init_input + epsilon * sign_data_grad

        return adv_input

    def AdaIN(self, content_features, style_features):
        """
        Apply Adaptive Instance Normalization (AdaIN) for 1D features.

        Arguments:
        content_features -- Tensor of content features, shape [N, C, L]
        style_features -- Tensor of style features, shape [N, C, L]

        Returns:
        transformed_features -- the resulting features after AdaIN
        """
        content_features = content_features.unsqueeze(0)
        style_features = style_features.unsqueeze(0)

        # 计算内容特征的均值和方差
        content_mean, content_std = self.calc_mean_std(content_features)

        # 计算风格特征的均值和方差
        style_mean, style_std = self.calc_mean_std(style_features)

        # 将内容特征标准化并使用风格特征的均值和方差进行缩放
        normalized_features = (content_features - content_mean.expand_as(content_features)) / content_std.expand_as(
            content_features)
        transformed_features = normalized_features * style_std.expand_as(content_features) + style_mean.expand_as(
            content_features)

        transformed_features = transformed_features.squeeze(0)

        return transformed_features

    def Whitening(self, feature_perturbed, feature_original):

        loss_wt = torch.zeros(1).to(self.device)
        # gram matrix
        # gram_a = torch.mm(feature_perturbed.t(), feature_perturbed)
        # gram_b = torch.mm(feature_original.t(), feature_original)
        covar_a, B = self.get_covariance_matrix(feature_perturbed)
        covar_b, B = self.get_covariance_matrix(feature_original)

        absolute_difference_map = torch.abs(covar_a - covar_b + 1e-5)
        absolute_difference_map_1D = absolute_difference_map.view(-1, 1)
        labels, centroids = self.kmeans(absolute_difference_map_1D, n_clusters=5)
        _, indices = centroids.view(-1).sort(descending=True)
        reassigned_labels = torch.zeros_like(labels)
        for i, idx in enumerate(indices):
            reassigned_labels[labels == idx] = i + 1

        cluster_map = reassigned_labels.view(512, 512)
        mask = torch.where(cluster_map == 1, torch.tensor(1).cuda(), torch.tensor(0).cuda())

        map_masked = absolute_difference_map * mask

        map_masked = map_masked.unsqueeze(0)

        off_diag_sum = torch.sum(torch.abs(map_masked), dim=(1, 2), keepdim=True) - self.margin
        off_diag_sum = torch.torch.clamp(off_diag_sum, min=1e-5)

        reversal_i = torch.ones(512, 512).triu(diagonal=1).cuda()
        num_off_diagonal = torch.sum(reversal_i)
        num_off_diagonal = torch.clamp(num_off_diagonal, min=1e-5)

        loss_wt = torch.clamp(torch.div(off_diag_sum, num_off_diagonal), min=0)
        loss_wt = torch.sum(loss_wt) / B
        loss_wt = torch.where(torch.isnan(loss_wt), torch.zeros_like(loss_wt), loss_wt)

        return loss_wt

    def get_covariance_matrix(self, f_map, eye=None):
        eps = 1e-5
        B, C = f_map.shape  # feature size (B X C), here B is 1 and C is 512
        if eye is None:
            eye = torch.eye(C).cuda()

        # Reshape f_map to a 3D tensor with a new dimension of size 1
        # Now f_map has a shape of B X C X 1
        f_map = f_map.view(B, C, -1)  # B X C X 1

        # Since there is only one feature vector per batch, HW-1 becomes 1-1=0, which would cause a division by zero
        # Thus, we skip the division and directly compute the covariance as the outer product of the vector with itself
        f_cor = torch.bmm(f_map, f_map.transpose(1, 2)) + (eps * eye)

        return f_cor, B

    def kmeans(self, X, n_clusters, n_iters=100):
        # 步骤1: 随机初始化中心点
        centroids = X[torch.randperm(X.size(0))[:n_clusters]]

        for _ in range(n_iters):
            # 步骤2: 分配点到最近的中心点
            distances = torch.cdist(X, centroids)
            labels = torch.argmin(distances, dim=1)

            # 步骤3: 更新中心点
            new_centroids = []
            for i in range(n_clusters):
                cluster_points = X[labels == i]
                if cluster_points.size(0) > 0:
                    new_centroids.append(cluster_points.mean(0))
                else:
                    # 如果某个聚类为空，则保留原始中心
                    new_centroids.append(centroids[i])

            centroids = torch.stack(new_centroids)

        return labels, centroids

    



