import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res101Encoder
from .attention import MultiHeadAttention
from .attention import MultiLayerPerceptron


class FewShotSeg(nn.Module):
    

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # ============ Original Components (Primary) ============
        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.criterion_MSE = nn.MSELoss()
        self.fg_sampler = np.random.RandomState(1289)
        self.fg_num = 100  # number of foreground partitions
        
        # Original attention modules
        self.MHA = MultiHeadAttention(n_head=3, d_model=512, d_k=512, d_v=512)
        self.MLP = MultiLayerPerceptron(dim=512, mlp_dim=1024)
        self.layer_norm = nn.LayerNorm(512)
        
        # ============ DSM Enhancement Components (Secondary) ============
        self.feat_dim = 512
        self.hidden_levels = 3
        
        # Weight to control DSM influence (0 = no DSM, 1 = full DSM)
        self.dsm_weight = nn.Parameter(torch.tensor(0.3))  # learnable weight
        
        # SFR Module Components
        self.v_s = nn.Parameter(torch.zeros(self.hidden_levels))
        self.v_q = nn.Parameter(torch.zeros(self.hidden_levels))
        nn.init.kaiming_uniform_(self.v_s.unsqueeze(0))
        nn.init.kaiming_uniform_(self.v_q.unsqueeze(0))
        
        self.G_s = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.Sigmoid()
        )
        self.G_q = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.Sigmoid()
        )
        
        self.rect_s = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.feat_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.feat_dim, self.feat_dim, 3, padding=1)
        )
        self.rect_q = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.feat_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.feat_dim, self.feat_dim, 3, padding=1)
        )
        
        # DSIS Module Components
        self.F_1d = nn.Conv1d(self.feat_dim, self.feat_dim, 1)
        self.dsis_layer_norm1 = nn.LayerNorm(self.feat_dim)
        self.dsis_MHA = MultiHeadAttention(n_head=3, d_model=self.feat_dim, d_k=self.feat_dim, d_v=self.feat_dim)
        self.dsis_layer_norm2 = nn.LayerNorm(self.feat_dim)
        self.spatial_pool = nn.AdaptiveAvgPool1d(1)
        self.channel_mapping = nn.Linear(self.feat_dim, 1)
        self.F_cc = nn.Linear(self.feat_dim, self.feat_dim // 2)
        self.F_rc = nn.Linear(self.feat_dim // 2, self.feat_dim)
        self.cross_attention = nn.MultiheadAttention(self.feat_dim, num_heads=4, batch_first=True)
        self.threshold_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, train=False, t_loss_scaler=1, n_iters=20):
        """
        Args:
            supp_imgs: support images - way x shot x [B x 3 x H x W]
            supp_mask: foreground masks - way x shot x [B x H x W]
            qry_imgs: query images - N x [B x 3 x H x W]
            qry_mask: query masks (for training)
            train: training mode flag
        """
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        self.iter = 3  # BATE iterations
        
        assert self.n_ways == 1
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)

        # ============ Dilate the mask (Original) ============
        kernel = np.ones((3, 3), np.uint8)
        supp_mask_ = supp_mask.cpu().numpy()[0][0][0]
        supp_dilated_mask = cv2.dilate(supp_mask_, kernel, iterations=1)
        supp_periphery_mask = supp_dilated_mask - supp_mask_
        supp_periphery_mask = np.reshape(supp_periphery_mask, (supp_bs, self.n_ways, self.n_shots,
                                                               np.shape(supp_periphery_mask)[0],
                                                               np.shape(supp_periphery_mask)[1]))
        supp_dilated_mask = np.reshape(supp_dilated_mask, (supp_bs, self.n_ways, self.n_shots,
                                                           np.shape(supp_dilated_mask)[0],
                                                           np.shape(supp_dilated_mask)[1]))
        supp_periphery_mask = torch.tensor(supp_periphery_mask).cuda()
        supp_dilated_mask = torch.tensor(supp_dilated_mask).cuda()

        # ============ Extract features ============
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)

        supp_fts = img_fts[:self.n_ways * self.n_shots * supp_bs].view(
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts.shape[-2:])
        qry_fts = img_fts[self.n_ways * self.n_shots * supp_bs:].view(
            qry_bs, self.n_queries, -1, *img_fts.shape[-2:])

        # Get threshold
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]
        self.thresh_pred = [self.t for _ in range(self.n_ways)]
        self.t_ = tao[:self.n_ways * self.n_shots * supp_bs]
        self.thresh_pred_ = [self.t_ for _ in range(self.n_ways)]

        # ============ Compute losses and outputs ============
        periphery_loss = torch.zeros(1).to(self.device)
        align_loss = torch.zeros(1).to(self.device)
        mse_loss = torch.zeros(1).to(self.device)
        qry_loss = torch.zeros(1).to(self.device)
        outputs = []

        for epi in range(supp_bs):
            # ============ Original: Partition prototypes (FPS) ============
            fg_partition_prototypes = [[self.compute_multiple_prototypes(
                self.fg_num, supp_fts[[epi], way, shot], supp_mask[[epi], way, shot], self.fg_sampler)
                for shot in range(self.n_shots)] for way in range(self.n_ways)]

            # ============ Original: Calculate coarse support prototype ============
            supp_fts_ = [[self.getFeatures(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]
            fg_prototypes = self.getPrototype(supp_fts_)

            # ============ Original: Dilated region prototypes ============
            supp_fts_dilated = [[self.getFeatures(supp_fts[[epi], way, shot], supp_dilated_mask[[epi], way, shot])
                                  for shot in range(self.n_shots)] for way in range(self.n_ways)]
            fg_prototypes_dilated = self.getPrototype(supp_fts_dilated)

            # ============ Original: Periphery prediction and loss ============
            supp_pred_object = torch.stack([self.getPred(supp_fts[epi][way], fg_prototypes[way], self.thresh_pred_[way])
                             for way in range(self.n_ways)], dim=1)
            supp_pred_object = F.interpolate(supp_pred_object, size=img_size, mode='bilinear', align_corners=True)

            supp_pred_dilated = torch.stack([self.getPred(supp_fts[epi][way], fg_prototypes_dilated[way], self.thresh_pred_[way])
                             for way in range(self.n_ways)], dim=1)
            supp_pred_dilated = F.interpolate(supp_pred_dilated, size=img_size, mode='bilinear', align_corners=True)

            pred_periphery = supp_pred_dilated - supp_pred_object
            pred_periphery = torch.cat((1.0 - pred_periphery, pred_periphery), dim=1)
            label_periphery = torch.full_like(supp_periphery_mask[epi][0][0], 255, device=supp_periphery_mask.device)
            label_periphery[supp_periphery_mask[epi][0][0] == 1] = 1
            label_periphery[supp_periphery_mask[epi][0][0] == 0] = 0

            eps_ = torch.finfo(torch.float32).eps
            log_prob_ = torch.log(torch.clamp(pred_periphery, eps_, 1 - eps_))
            periphery_loss += self.criterion(log_prob_, label_periphery[None, ...].long()) / self.n_shots / self.n_ways

            # ============ Original: Initial query prediction ============
            qry_pred = torch.stack(
                [self.getPred(qry_fts[epi], fg_prototypes[way], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1)
            qry_prototype_coarse = self.getFeatures(qry_fts[epi], qry_pred[epi])

            # ============ Original: BATE iterations (Primary) ============
            for i in range(self.iter):
                fg_partition_prototypes = [[self.BATE(fg_partition_prototypes[way][shot][epi], qry_prototype_coarse)
                                            for shot in range(self.n_shots)] for way in range(self.n_ways)]

                supp_proto = [[torch.mean(fg_partition_prototypes[way][shot], dim=1) + fg_prototypes[way] 
                               for shot in range(self.n_shots)] for way in range(self.n_ways)]

                # CQPC module
                qry_pred_coarse = torch.stack(
                    [self.getPred(qry_fts[epi], supp_proto[way][epi], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)
                qry_prototype_coarse = self.getFeatures(qry_fts[epi], qry_pred_coarse[epi])

            # ============ DSM Enhancement (Secondary) ============
            # Apply DSM to get enhanced prototype
            dsm_proto = self.dsm_enhance(supp_fts[epi, 0, 0], qry_fts[epi, 0], supp_mask[epi, 0, 0])
            
            # Fuse original prototype with DSM prototype
            dsm_w = torch.sigmoid(self.dsm_weight)  # constrain to [0, 1]
            final_proto = [[((1 - dsm_w) * supp_proto[way][shot] + dsm_w * dsm_proto)
                           for shot in range(self.n_shots)] for way in range(self.n_ways)]

            # ============ Final Query Prediction ============
            qry_pred = torch.stack(
                [self.getPred(qry_fts[epi], final_proto[way][epi], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1)

            qry_pred_up = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)
            preds = torch.cat((1.0 - qry_pred_up, qry_pred_up), dim=1)
            outputs.append(preds)

            # ============ Compute Training Losses ============
            if train:
                align_loss_epi = self.alignLoss(supp_fts[epi], qry_fts[epi], preds, supp_mask[epi])
                align_loss += align_loss_epi
            if train:
                proto_mse_loss_epi = self.proto_mse(qry_fts[epi], preds, supp_mask[epi], fg_prototypes)
                mse_loss += proto_mse_loss_epi
            if train:
                qry_fts_ = [[self.getFeatures(qry_fts[epi], qry_mask)]]
                qry_prototypes = self.getPrototype(qry_fts_)
                qry_pred_train = self.getPred(qry_fts[epi], qry_prototypes[epi], self.thresh_pred[epi])

                qry_pred_train = F.interpolate(qry_pred_train[None, ...], size=img_size, mode='bilinear', align_corners=True)
                preds_train = torch.cat((1.0 - qry_pred_train, qry_pred_train), dim=1)

                qry_label = torch.full_like(qry_mask[epi], 255, device=qry_mask.device)
                qry_label[qry_mask[epi] == 1] = 1
                qry_label[qry_mask[epi] == 0] = 0

                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(preds_train, eps, 1 - eps))
                qry_loss += self.criterion(log_prob, qry_label[None, ...].long()) / self.n_shots / self.n_ways

        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])

        return output, periphery_loss / supp_bs, align_loss / supp_bs, mse_loss / supp_bs, qry_loss / supp_bs

    # ============ DSM Enhancement Module ============
    def dsm_enhance(self, F_s, F_q, M_s):
        """
        DSM Enhancement: SFR + DSIS
        Returns enhanced prototype to be fused with original prototype
        
        Args:
            F_s: support features [C x H' x W']
            F_q: query features [C x H' x W']
            M_s: support mask [H x W]
        Returns:
            P_star: enhanced prototype [1 x C]
        """
        # Generate hidden features (simulated multi-level)
        h, w = F_s.shape[-2:]
        hidden_s = [
            F.interpolate(F_s.unsqueeze(0), size=(max(1, h//2), max(1, w//2)), mode='bilinear', align_corners=True).squeeze(0),
            F_s,
            F_s
        ]
        hidden_q = [
            F.interpolate(F_q.unsqueeze(0), size=(max(1, h//2), max(1, w//2)), mode='bilinear', align_corners=True).squeeze(0),
            F_q,
            F_q
        ]
        
        # SFR: Support-query Feature Re-weighting
        F_dot_s, F_dot_q = self.support_query_reweighting(F_s, F_q, hidden_s, hidden_q)
        
        # DSIS: Dynamic Semantic Information Selection
        P_star = self.dynamic_semantic_selection(F_dot_s, F_dot_q, M_s)
        
        return P_star

    def support_query_reweighting(self, F_s, F_q, hidden_s, hidden_q):
        """SFR Module"""
        C, H, W = F_s.shape
        N = H * W
        
        F_s_flat = F_s.view(C, N).permute(1, 0)
        F_q_flat = F_q.view(C, N).permute(1, 0)
        
        d_k = C ** 0.5
        S = torch.softmax(torch.mm(F_s_flat, F_q_flat.t()) / d_k, dim=-1)
        
        F_hat_s = torch.mm(S, F_s_flat)
        F_hat_q = torch.mm(S, F_q_flat)
        
        # WFG
        f_bar_s = [h.mean(dim=(-2, -1)) for h in hidden_s]
        f_bar_q = [h.mean(dim=(-2, -1)) for h in hidden_q]
        
        w_s = torch.sigmoid(self.v_s)
        w_q = torch.sigmoid(self.v_q)
        
        weighted_s = sum(w_s[i] * f_bar_s[i] for i in range(len(f_bar_s)))
        weighted_q = sum(w_q[i] * f_bar_q[i] for i in range(len(f_bar_q)))
        
        gamma_s = self.G_s(weighted_s)
        gamma_q = self.G_q(weighted_q)
        
        F_hat_s_scaled = F_hat_s * gamma_s.unsqueeze(0)
        F_hat_q_scaled = F_hat_q * gamma_q.unsqueeze(0)
        
        F_hat_s_2d = F_hat_s_scaled.permute(1, 0).view(1, C, H, W)
        F_hat_q_2d = F_hat_q_scaled.permute(1, 0).view(1, C, H, W)
        
        F_dot_s = self.rect_s(F_hat_s_2d) + F_s.unsqueeze(0)
        F_dot_q = self.rect_q(F_hat_q_2d) + F_q.unsqueeze(0)
        
        return F_dot_s.squeeze(0), F_dot_q.squeeze(0)

    def dynamic_semantic_selection(self, F_dot_s, F_dot_q, M_s):
        """DSIS Module"""
        C = F_dot_s.shape[0]
        
        # Mean-value centers
        P_bar_s = self.masked_average_pooling(F_dot_s, M_s)
        M_tilde_q = self.get_coarse_query_mask(P_bar_s, F_dot_q)
        P_bar_q = self.masked_average_pooling(F_dot_q, M_tilde_q)
        
        # Median-value centers
        P_hat_s = self.median_value_center(F_dot_s, M_s)
        P_hat_q = self.median_value_center(F_dot_q, M_tilde_q)
        
        # Cross attention fusion
        P_bar, _ = self.cross_attention(P_bar_s.unsqueeze(0), P_bar_q.unsqueeze(0), P_bar_q.unsqueeze(0))
        P_bar = P_bar.squeeze(0)
        
        P_hat, _ = self.cross_attention(P_hat_s.unsqueeze(0), P_hat_q.unsqueeze(0), P_hat_q.unsqueeze(0))
        P_hat = P_hat.squeeze(0)
        
        # Selection factor
        xi = self.learn_selection_factor(F_dot_s, F_dot_q)
        
        # Dynamic selection
        half_C = C // 2
        indices = torch.randperm(C, device=F_dot_s.device)
        P = torch.cat([P_hat[:, indices[:half_C]], P_bar[:, indices[half_C:]]], dim=1)
        
        C_r = torch.sigmoid(self.F_rc(F.relu(self.F_cc(P))))
        k = max(1, int(C * xi.item()))
        _, top_indices = torch.topk(C_r, k, dim=1)
        
        P_star_sparse = torch.gather(P, 1, top_indices)
        P_star = F.adaptive_avg_pool1d(P_star_sparse.unsqueeze(0), C).squeeze(0)
        
        return P_star

    def masked_average_pooling(self, features, mask):
        C, H_feat, W_feat = features.shape
        mask_resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), 
                                     size=(H_feat, W_feat), mode='bilinear', align_corners=True).squeeze()
        masked_features = features * mask_resized.unsqueeze(0)
        prototype = masked_features.sum(dim=(-2, -1)) / (mask_resized.sum() + 1e-5)
        return prototype.unsqueeze(0)

    def get_coarse_query_mask(self, P_s, F_q):
        C, H, W = F_q.shape
        F_q_flat = F_q.view(C, -1).permute(1, 0)
        similarity = F.cosine_similarity(P_s, F_q_flat, dim=1).view(H, W)
        tau = self.threshold_fc(F_q.unsqueeze(0))
        mask = (torch.softmax(similarity.view(-1), dim=0).view(H, W) > tau).float()
        return mask

    def median_value_center(self, features, mask):
        C, H_feat, W_feat = features.shape
        mask_resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                                     size=(H_feat, W_feat), mode='bilinear', align_corners=True).squeeze()
        mask_bool = mask_resized > 0.5
        if mask_bool.sum() == 0:
            return features.mean(dim=(-2, -1)).unsqueeze(0)
        
        fg_features = features[:, mask_bool].permute(1, 0)
        pixel_sums = fg_features.sum(dim=1)
        sorted_indices = torch.argsort(pixel_sums)
        median_idx = sorted_indices[len(sorted_indices) // 2]
        return fg_features[median_idx].unsqueeze(0)

    def learn_selection_factor(self, F_dot_s, F_dot_q):
        C, H, W = F_dot_s.shape
        N = H * W
        
        F_s_flat = F_dot_s.view(C, N).permute(1, 0)
        F_q_flat = F_dot_q.view(C, N).permute(1, 0)
        F_concat = torch.cat([F_s_flat, F_q_flat], dim=0)
        
        F_concat_conv = F_concat.t().unsqueeze(0)
        F_a = self.F_1d(F_concat_conv).squeeze(0).t()
        F_a = F.relu(self.dsis_layer_norm1(F_a))
        
        F_a_exp = F_a.unsqueeze(0)
        F_b = self.dsis_layer_norm2(self.dsis_MHA(F_a_exp, F_a_exp, F_a_exp) + F_a_exp).squeeze(0)
        
        F_b_t = F_b.t().unsqueeze(0)
        F_b_spatial = self.spatial_pool(F_b_t).squeeze(-1).squeeze(0)
        xi = torch.sigmoid(self.channel_mapping(F_b_spatial.unsqueeze(0)))
        
        return xi.squeeze()

    # ============ Original Methods (Primary) ============
    def getPred(self, fts, prototype, thresh):
        """Calculate the distance between features and prototypes"""
        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))
        return pred

    def getFeatures(self, fts, mask):
        """Extract foreground features via masked average pooling"""
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        return masked_fts

    def getFeatures_fg(self, fts, mask):
        fts_ = fts.squeeze(0).permute(1, 2, 0)
        fts_ = fts_.view(fts_.size()[0] * fts_.size()[1], fts_.size()[2])
        mask_ = F.interpolate(mask.unsqueeze(0), size=fts.shape[-2:], mode='bilinear')
        mask_ = mask_.view(-1)
        l = math.ceil(mask_.sum())
        c = torch.argsort(mask_, descending=True, dim=0)
        fg = c[:l]
        fts_fg = fts_[fg]
        return fts_fg

    def getPrototype(self, fg_fts):
        """Average the features to obtain the prototype"""
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots 
                         for way in fg_fts]
        return fg_prototypes

    def compute_multiple_prototypes(self, fg_num, sup_fts, sup_fg, sampler):
        """Compute multiple prototypes using farthest point sampling"""
        B, C, h, w = sup_fts.shape
        fg_mask = F.interpolate(sup_fg.unsqueeze(0), size=sup_fts.shape[-2:], mode='bilinear')
        fg_mask = fg_mask.squeeze(0).bool()
        batch_fg_protos = []

        for b in range(B):
            fg_protos = []
            fg_mask_i = fg_mask[b]

            with torch.no_grad():
                if fg_mask_i.sum() < fg_num:
                    fg_mask_i = fg_mask[b].clone()
                    fg_mask_i.view(-1)[:fg_num] = True

            all_centers = []
            first = True
            pts = torch.stack(torch.where(fg_mask_i), dim=1)
            for _ in range(fg_num):
                if first:
                    i = sampler.choice(pts.shape[0])
                    first = False
                else:
                    dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                    i = torch.argmax((dist ** 2).sum(-1).min(1)[0])
                pt = pts[i]
                all_centers.append(pt)

            dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
            fg_labels = torch.argmin((dist ** 2).sum(-1), dim=1)

            fg_feats = sup_fts[b].permute(1, 2, 0)[fg_mask_i]
            for i in range(fg_num):
                proto = fg_feats[fg_labels == i].mean(0)
                fg_protos.append(proto)

            fg_protos = torch.stack(fg_protos, dim=1)
            batch_fg_protos.append(fg_protos)
        fg_proto = torch.stack(batch_fg_protos, dim=0).transpose(1, 2)

        return fg_proto

    def BATE(self, fg_prototypes, qry_prototype_coarse):
        """Bidirectional Attention-based Prototype Enhancement"""
        A = torch.mm(fg_prototypes, qry_prototype_coarse.t())
        kc = ((A.min() + A.mean()) / 2).floor()

        if A is not None:
            S = torch.zeros(A.size(), dtype=torch.float).cuda()
            S[A < kc] = -10000.0

        A = torch.softmax((A + S), dim=0)
        A = torch.mm(A, qry_prototype_coarse)
        A = self.layer_norm(A + fg_prototypes)

        T = self.MHA(A.unsqueeze(0), A.unsqueeze(0), A.unsqueeze(0))
        T = self.MLP(T)

        return T

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()

        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            for shot in range(n_shots):
                qry_fts_ = [self.getFeatures(qry_fts, pred_mask[way + 1])]
                fg_prototypes = self.getPrototype([qry_fts_])

                supp_pred = self.getPred(supp_fts[way, [shot]], fg_prototypes[way], self.thresh_pred[way])
                supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)

                preds = supp_pred
                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def proto_mse(self, qry_fts, pred, fore_mask, supp_prototypes):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()

        loss_sim = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            for shot in range(n_shots):
                qry_fts_ = [[self.getFeatures(qry_fts, pred_mask[way + 1])]]
                fg_prototypes = self.getPrototype(qry_fts_)

                fg_prototypes_ = torch.sum(torch.stack(fg_prototypes, dim=0), dim=0)
                supp_prototypes_ = torch.sum(torch.stack(supp_prototypes, dim=0), dim=0)

                loss_sim += self.criterion_MSE(fg_prototypes_, supp_prototypes_)

        return loss_sim
    


