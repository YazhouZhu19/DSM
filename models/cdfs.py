from re import A
from tkinter import W
import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res50Encoder
from .clip_encoders import *
from .decoders import *


from functools import reduce
from operator import add
from torchvision.models import resnet
from torchvision.models import vgg
import torchvision.models as models
from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import MixedLearner
import matplotlib.pyplot as plt


def visualize_segmentation_tensor(tensor):
    # Ensure the tensor is on CPU and detach from computation graph
    tensor = tensor.cpu().detach()

    # Squeeze the batch dimension
    tensor = tensor.squeeze(0)

    # Get the predicted class for each pixel (argmax along channel dimension)
    predicted_mask = torch.argmax(tensor, dim=0).numpy()

    # Create a color map: 0 -> black, 1 -> white
    cmap = plt.cm.gray

    # Create the plot
    plt.figure(figsize=(10, 10))
    plt.imshow(predicted_mask, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.title("Segmentation Mask")
    plt.axis('off')
    plt.show()


class AttentionFusion(nn.Module):
    def __init__(self, input_dim, attention_type='self', num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.attention_type = attention_type
        
        if attention_type == 'self':
            self.attention = SelfAttention(input_dim)
        elif attention_type == 'cross':
            self.attention = CrossAttention(input_dim)
        elif attention_type == 'additive':
            self.attention = AdditiveAttention(input_dim)
        elif attention_type == 'dot_product':
            self.attention = DotProductAttention(input_dim)
        elif attention_type == 'multi_head':
            self.attention = MultiHeadAttention(input_dim, num_heads)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")
        
        self.output_layer = nn.Linear(input_dim, input_dim)
        
    def forward(self, x1, x2):
        if self.attention_type in ['self', 'multi_head']:
            x = torch.cat([x1, x2], dim=1)  # (batch_size, 2, input_dim)
            attention_output = self.attention(x)
            # 对 attention_output 进行平均池化
            attention_output = attention_output.mean(dim=1, keepdim=True)  # (batch_size, 1, input_dim)
        else:
            attention_output = self.attention(x1, x2)  # 已经是 (batch_size, 1, input_dim)
        
        output = self.output_layer(attention_output)
        return output  # (batch_size, 1, input_dim)

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.query.out_features ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.query.out_features ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

class AdditiveAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1)
        
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        energy = self.v(torch.tanh(q + k))
        attention = F.softmax(energy, dim=1)
        return (attention * x2)

class DotProductAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        
    def forward(self, x1, x2):
        dot = torch.matmul(x1, x2.transpose(-2, -1)) * self.scale
        attention = F.softmax(dot, dim=-1)
        return torch.matmul(attention, x2)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.shape[0], -1, self.heads, t.shape[-1] // self.heads).transpose(1, 2), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], -1, out.shape[-1] * self.heads)
        return self.to_out(out)


class innerProtoFusion(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, prototypes):
        # prototypes shape: (n, 1, 512)
        n = prototypes.size(0)
        
        # Compute query, key, and value
        q = self.query(prototypes)  # (n, 1, 512)
        k = self.key(prototypes)    # (n, 1, 512)
        v = self.value(prototypes)  # (n, 1, 512)
        
        # Compute attention weights
        attn_weights = torch.bmm(q, k.transpose(1, 2))  # (n, 1, 1)
        attn_weights = torch.softmax(attn_weights, dim=0)  # (n, 1, 1)
        
        # Weighted sum
        fused_feature = torch.sum(attn_weights * v, dim=0)  # (1, 512)
        
        return fused_feature


class Weighting(nn.Module):
    def __init__(self, in_channels=512):
        super(Weighting, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        # x, y: (batch_size, in_channels, H, W)
        batch_size, C, H, W = x.size()

        # Compute Query, Key, and Values
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # (B, H*W, C')
        key = self.key_conv(y).view(batch_size, -1, H * W)  # (B, C', H*W)
        value1 = self.value_conv1(x).view(batch_size, -1, H * W)  # (B, C, H*W)
        value2 = self.value_conv2(y).view(batch_size, -1, H * W)  # (B, C, H*W)

        # Calculate attention
        attention = torch.bmm(query, key)  # (B, H*W, H*W)
        attention = F.softmax(attention, dim=-1)

        # Apply attention to both values
        out1 = torch.bmm(value1, attention.permute(0, 2, 1))  # (B, C, H*W)
        out2 = torch.bmm(value2, attention)  # (B, C, H*W)

        # Reshape outputs
        out1 = out1.view(batch_size, C, H, W)
        out2 = out2.view(batch_size, C, H, W)

        # Apply gamma and add residual connection
        out1 = self.gamma1 * out1 + x
        out2 = self.gamma2 * out2 + y

        return out1, out2


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
    
    def forward(self, x):
        return self.features(x)


class FewShotSeg(nn.Module):

    def __init__(self, args, backbone, use_original_imgsize):
        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize

        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = MixedLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # myself blocks
        self.scaler = 20.0
        self.weighting = Weighting()

        self.learnThreshold_a = nn.Sequential(
            nn.Conv2d(512, 2048, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.learnThreshold_b = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.Linear(in_features=1000, out_features=1, bias=True)
        )

        # attention_types = ['self', 'cross', 'additive', 'dot_product', 'multi_head']
        self.middle_fusion = AttentionFusion(input_dim=512, attention_type='cross')
        self.innerProtoFusion = innerProtoFusion()  

        self.criterion = nn.NLLLoss(ignore_index=255, weight=torch.FloatTensor([0.1, 1.0]).cuda())

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, prompt, train=False):

        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        query_img = qry_imgs[0]
        support_img = supp_imgs[0][0]
        support_mask = supp_mask[0][0]
        query_mask = qry_mask

        img_size = supp_imgs[0][0].shape[-2:]
        loss_spt_middle = torch.zeros(1).to(self.device)
        loss_qry_middle = torch.zeros(1).to(self.device)

        with torch.no_grad():

            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            # query_feats: (n-1) x (1, 512, 32 or 16, 32 or 16)   query_final_feats: (1, 512, 8, 8)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            # support_feats: (n-1) x (1, 512, 32 or 16, 32 or 16)  support_final_feats: (1, 512, 8, 8)
        
        num_inner_layer = len(self.feat_ids)
        # inner features  
        middle_prototypes = []
        for idx in range(num_inner_layer): 

            qry_fts_inner, spt_fts_inner = query_feats[idx], support_feats[idx]
            weighted_inner_ft_qry, weighted_inner_ft_spt  = self.weighting(qry_fts_inner, spt_fts_inner)
            # add semantic information and coarse prediction calculation
            coarse_pred = self.coarse_prediction(weighted_inner_ft_spt, weighted_inner_ft_qry, support_mask.clone())
            # middle prototype calculation
            middle_prototype = self.middle_information(support_mask.clone(), coarse_pred.clone(), weighted_inner_ft_spt, weighted_inner_ft_qry)
            
            if train:
                # loss for support image 
                t_spt = self.learnThreshold_a(support_feats[num_inner_layer - 1])
                t_spt = torch.flatten(t_spt, 1)
                t_spt = self.learnThreshold_b(t_spt)
                sim_spt = -F.cosine_similarity(support_feats[num_inner_layer - 1], middle_prototype[..., None, None], dim=1) * self.scaler
                pred_spt = 1.0 - torch.sigmoid(0.5 * (sim_spt - t_spt))
                pred_spt = pred_spt.unsqueeze(0)
                pred_spt = F.interpolate(pred_spt, size=img_size, mode='bilinear', align_corners=True)
                pred_spt = torch.cat((1.0 - pred_spt, pred_spt), dim=1)

                loss_spt_middle = self.CE_loss(pred_spt, support_mask.clone())

                # loss for query image
                t_qry = self.learnThreshold_a(query_feats[num_inner_layer - 1])
                t_qry = torch.flatten(t_qry, 1)
                t_qry = self.learnThreshold_b(t_qry)
                sim_qry = -F.cosine_similarity(query_feats[num_inner_layer - 1], middle_prototype[..., None, None], dim=1) * self.scaler
                pred_qry = 1.0 - torch.sigmoid(0.5 * (sim_qry - t_qry))
                pred_qry = pred_qry.unsqueeze(0)
                pred_qry = F.interpolate(pred_qry, size=img_size, mode='bilinear', align_corners=True)
                pred_qry = torch.cat((1.0 - pred_qry, pred_qry), dim=1)

                loss_qry_middle = self.CE_loss(pred_qry, query_mask.clone())

            middle_prototypes.append(middle_prototype)
        
        support_feats = self.mask_feature(support_feats, support_mask.clone())
        corr = Correlation.multilayer_correlation_new(query_feats, support_feats, self.stack_ids, middle_prototypes)

        logit_mask = self.hpn_learner(corr)

        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, size=img_size, mode='bilinear', align_corners=True)

        return logit_mask, loss_spt_middle, loss_qry_middle
    
    
    # designed functions
    def masked(self, feature, mask):
        
        """
        mask: (1, H, W)
        feature: (1, C, h, w)
        """
        mask = F.interpolate(mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
        feature = feature * mask

        return feature
    
    def coarse_prediction(self, spt_fts, qry_fts, mask): 
        """
        mask: (1, H, W)
        spt_fts: (1, C, h, w)
        qry_fts: (1, C, h, w)
        """

        fts = F.interpolate(spt_fts, size=mask.shape[-2:], mode='bilinear')
        
        # prototype
        prototype = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C
        
        # threshold learning
        t = self.learnThreshold_a(fts)
        t = torch.flatten(t, 1)
        t = self.learnThreshold_b(t)

        # prediction
        sim = -F.cosine_similarity(qry_fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - t))

        return pred

    def middle_information(self, spt_pred, qry_pred, spt_fts, qry_fts):

        """
        使用加权均值和加权中位数计算 原型 
        """

        def weighted_prototype(fts, pred, method): 
            if method == 'mean': 
                proto = torch.sum(fts * pred[None, ...], dim=(-2, -1)) \
                     / (pred[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C
            elif method == 'median': 
                B, C, H, W = fts.shape
                fts_flat = fts.view(B, C, -1)  # (B, C, H*W)
                pred_flat = pred.view(B, 1, -1)  # (B, 1, H*W)


                proto_list = []
                for c in range(C):
                    channel_fts = fts_flat[:, c, :]  # (B, H*W)

                    sorted_fts, indices = torch.sort(channel_fts, dim=-1)
                    indices = indices.unsqueeze(1)  # (B, 1, H*W)


                    sorted_weights = torch.gather(pred_flat, -1, indices)
                
                    cumsum_weights = torch.cumsum(sorted_weights, dim=-1)
                    total_weight = cumsum_weights[:, :, -1:]
                    median_weight = total_weight / 2
                
                    median_idx = torch.searchsorted(cumsum_weights, median_weight)
                    median_value = torch.gather(sorted_fts, -1, median_idx.squeeze(1))
                    proto_list.append(median_value)

                proto = torch.cat(proto_list, dim=1)  # (B, C)

            else: 

                raise ValueError("Method must be either 'mean' or 'median'")

            return proto 
 
        spt_fts = F.interpolate(spt_fts, size=spt_pred.shape[-2:], mode='bilinear')
        qry_fts = F.interpolate(qry_fts, size=qry_pred.shape[-2:], mode='bilinear')
        
        spt_proto_mean = weighted_prototype(spt_fts, spt_pred, method = 'mean')
        qry_proto_mean = weighted_prototype(qry_fts, qry_pred, method = 'mean')
        
        middle_proto_mean = self.middle_fusion(spt_proto_mean, qry_proto_mean)
        
        spt_proto_median = weighted_prototype(spt_fts, spt_pred, method = 'median')
        qry_proto_median = weighted_prototype(qry_fts, qry_pred, method = 'median')

        middle_proto_median = self.middle_fusion(spt_proto_median, qry_proto_median)

        proto = self.middle_fusion(middle_proto_mean, middle_proto_median)

        return proto 

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features
    
    def predict_mask(self, supp_imgs, supp_mask, qry_imgs, qry_mask, prompt, train=False):

        # query_img = qry_imgs[0]
        # support_img = supp_imgs[0][0]
        # support_mask = supp_mask[0][0]

        # Perform prediction for a single support set
        logit_mask = self(supp_imgs, supp_mask, qry_imgs, qry_mask, prompt, train=False)
        
        if self.use_original_imgsize:
            query_img = qry_imgs[0]
            org_qry_imsize = query_img.shape[2:]  # Assuming query_img is a tensor with shape [B, C, H, W]
            logit_mask = F.interpolate(logit_mask, size=org_qry_imsize, mode='bilinear', align_corners=True)
            

        visualize_segmentation_tensor(logit_mask)
        # Get the predicted mask
        pred_mask = logit_mask.argmax(dim=1)

        
        
        # Average & quantize predictions given threshold (=0.5)
        bsz = pred_mask.size(0)
        max_vote = pred_mask.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = pred_mask.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask 

    def compute_objective(self, logit_mask, gt_mask):

        bsz = logit_mask.size(0)

        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
    
    def CE_loss(self, pred, label):

        """
        :param pred: (1, 2, H, W)
        :param label: (1, 2, H, W)
        :return:
        """

        pred = pred.long()
        label = label.long()
        loss = self.criterion(torch.log(torch.clamp(pred, torch.finfo(torch.float32).eps,
                                        1 - torch.finfo(torch.float32).eps)), label)
        return loss







