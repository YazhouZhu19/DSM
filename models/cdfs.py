"""
encoder resnet50, GNN geometrical feature extractor, feature transform operation
"""
from re import A
from tkinter import W
from turtle import forward
from xml.etree.ElementInclude import include
import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res50Encoder
# from .clip_encoders import *
from .decoders import *


from functools import reduce
from operator import add
from torchvision.models import resnet
from torchvision.models import vgg
import torchvision.models as models
from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import MixedLearner
# from .learner import Matching
import matplotlib.pyplot as plt

from mamba_ssm import Mamba 


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
    def __init__(self, in_channels=512, max_hidden_layers=10):
        super(Weighting, self).__init__()
        self.in_channels = in_channels
        self.max_hidden_layers = max_hidden_layers
        
        # Original convolutional layers
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable weights for combining hidden features
        self.hidden_weights1 = nn.Parameter(torch.ones(max_hidden_layers))
        self.hidden_weights2 = nn.Parameter(torch.ones(max_hidden_layers))
        
        # Layers for gamma generation
        self.gamma_gen1 = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.gamma_gen2 = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # residual learning block
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

        self.relu = nn.ReLU()


    def forward(self, x, y, x_hidden, y_hidden):
        batch_size, C, H, W = x.size()

        # Compute Query, Key, and Values
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key_conv(y).view(batch_size, -1, H * W)
        value1 = self.value_conv1(x).view(batch_size, -1, H * W)
        value2 = self.value_conv2(y).view(batch_size, -1, H * W)

        # Calculate attention
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        # Apply attention to both values
        out1 = torch.bmm(value1, attention.permute(0, 2, 1))
        out2 = torch.bmm(value2, attention)

        # Reshape outputs
        out1 = out1.view(batch_size, C, H, W)
        out2 = out2.view(batch_size, C, H, W)

        # Combine hidden features with learnable weights
        num_layers = min(len(x_hidden), self.max_hidden_layers)
        x_weights = F.softmax(self.hidden_weights1[:num_layers], dim=0)
        y_weights = F.softmax(self.hidden_weights2[:num_layers], dim=0)
        
        x_hidden_weighted = sum([w * h.mean(dim=[2, 3]) for w, h in zip(x_weights, x_hidden[:num_layers])])
        y_hidden_weighted = sum([w * h.mean(dim=[2, 3]) for w, h in zip(y_weights, y_hidden[:num_layers])])

        # Generate gammas using weighted hidden features
        gamma1 = self.gamma_gen1(x_hidden_weighted).view(batch_size, 1, 1, 1)
        gamma2 = self.gamma_gen2(y_hidden_weighted).view(batch_size, 1, 1, 1)

        # Apply gamma and add residual connection
        out1 = self.residual(gamma1 * out1) + x
        out1 = self.relu(out1)

        out2 = self.residual(gamma2 * out2) + y 
        out2 = self.relu(out2)

        return out1, out2

class Weighting_old(nn.Module):
    def __init__(self, in_channels=512):
        super(Weighting_old, self).__init__()
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
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # (B, H*W, C')  (1, 1024, 64)
        key = self.key_conv(y).view(batch_size, -1, H * W)  # (B, C', H*W)                       (1, 64, 1024)
        value1 = self.value_conv1(x).view(batch_size, -1, H * W)  # (B, C, H*W)                  (1, 512, 32*32)
        value2 = self.value_conv2(y).view(batch_size, -1, H * W)  # (B, C, H*W)                  (1, 512, 32*32)


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

class factor_learning(nn.Module):

    def __init__(self, dim):
        super(factor_learning, self).__init__()

        self.in_channel = dim
        self.batch = 1

        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        ).to("cuda")
        self.LN = nn.LayerNorm(self.in_channel)

        self.linear = nn.Conv1d(
            in_channels=self.in_channel,
            out_channels=self.in_channel,
            kernel_size=3, 
            padding='same'
        )
        
        self.relu = nn.ReLU()
        
        self.mlp_a = nn.Linear(dim, 1)
        self.mlp_b = nn.Linear(128, 1)

    def forward(self, spt_fts, qry_fts):

        """
        spt_fts: (1, 512, 8, 8)
        qry_fts: (1, 512, 8, 8)
        """

        B, dim, h, w = spt_fts.size()
        spt_fts = spt_fts.view(B, h*w, dim)             # (1, 64, 512)
        qry_fts = qry_fts.view(B, h*w, dim)             # (1, 64, 512)
        
        fts = torch.cat([spt_fts, qry_fts], dim=1)      # (1, 128, 512)
        fts = fts.permute(0, 2, 1)  # (1, 512, 128)
        fts = self.linear(fts)      # (1, 512, 128)
        fts = self.LN(fts.transpose(1, 2)).transpose(2, 1)      # (1, 512, 128)
        fts = self.relu(fts)        # (1, 512, 128)

        fts_ = self.mamba(fts.transpose(1, 2)).transpose(2, 1)  # (1, 512, 128)   
        fts = fts_ + fts            # (1, 512, 128)
        fts = self.LN(fts.transpose(1, 2)).transpose(2, 1)      # (1, 512, 128)          
        
        coff = self.mlp_a(fts.transpose(1, 2)).transpose(2, 1)  # (1, 1, 128)
        coff = self.mlp_b(coff).squeeze(-1).squeeze(-1)         # (1)
        coff = torch.sigmoid(coff)

        return coff

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
        self.criterion_MSE = nn.MSELoss()

        # myself blocks
        self.scaler = 20.0
        self.weighting = Weighting()  
        self.weighting_old = Weighting_old()

        self.learnThreshold_a = nn.Sequential(
            nn.Conv2d(512, 2048, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.learnThreshold_b = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.Linear(in_features=1000, out_features=1, bias=True)
        )
        self.decoder = Decoder(input_channels=512, num_classes=1)

        # attention_types = ['self', 'cross', 'additive', 'dot_product', 'multi_head']
        self.middle_fusion = AttentionFusion(input_dim=512, attention_type='cross')
        self.innerProtoFusion = innerProtoFusion()  

        self.criterion = nn.NLLLoss(ignore_index=255, weight=torch.FloatTensor([0.1, 1.0]).cuda())

        # self.matching = Matching(list(reversed(nbottlenecks[-3:])))

        self.factor_learning = factor_learning(dim=512) 

        self.channel_importance = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(512)




    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask, train=False):

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
        support_mask = supp_mask[0][0]  # (1, 256, 256)
        query_mask = qry_mask


        img_size = supp_imgs[0][0].shape[-2:]
        loss_spt_middle = torch.zeros(1).to(self.device)
        loss_qry_middle = torch.zeros(1).to(self.device)

        # with torch.no_grad():

        #     query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        #     # query_feats: (n-1) x (1, 512, 32 or 16, 32 or 16)   query_final_feats: (1, 512, 8, 8)
        #     support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        #     # support_feats: (n-1) x (1, 512, 32 or 16, 32 or 16)  support_final_feats: (1, 512, 8, 8)

        #     # support_feats = self.mask_feature(support_feats, support_mask.clone())
        #     # corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)

        query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        
        num_inner_layer = len(self.feat_ids)
        # inner features  
        middle_prototypes = []

        ################################### Support-query features re-weighting ##################################
        spt_fts, qry_fts = support_feats[num_inner_layer - 1], query_feats[num_inner_layer - 1]
        spt_hidden_fts, qry_hidden_fts = support_feats[:-1], query_feats[:-1]
        spt_weighted_fts, qry_weighted_fts = self.weighting(spt_fts, qry_fts, spt_hidden_fts, qry_hidden_fts)     # (1, C, h, w) (1, C, h, w)

        ################################ Dynamic Selected Semantic Information Learning ###############################
        # step 1: Selection Factor Learning
        coff = self.factor_learning(spt_weighted_fts, qry_weighted_fts)
        coarse_pred = self.coarse_prediction(spt_weighted_fts, qry_weighted_fts, support_mask.clone())
        coarse_pred = coarse_pred.unsqueeze(0)  # (1, 1, h, w)
        coarse_pred = F.interpolate(coarse_pred, size=img_size, mode='bilinear', align_corners=True)

        preds_coarse = torch.cat((1.0 - coarse_pred, coarse_pred), dim=1)  # (1, 2, 256, 256)
       
        align_loss = torch.zeros(1).to(self.device)
        if train:
            # spt_fts: (1, C, h, w)  qry_fts: (1, C, h, w)
            spt_fts_align = spt_fts.unsqueeze(0)    # (1, 1, C, h, w)
            qry_fts_align = qry_fts                 # (1, C, h, w)
            preds = preds_coarse                    # (1, 2, 256, 256)
            supp_mask = support_mask.unsqueeze(0)   # (1, 1, 256, 256)
            align_loss_epi = self.alignLoss(spt_fts_align, qry_fts_align, preds, supp_mask)
            align_loss += align_loss_epi
        
        
        semantic_information = self.dynamic_selection(support_mask.clone(), coarse_pred.clone(), spt_weighted_fts, qry_weighted_fts, coff) 

    
        mse_loss = torch.zeros(1).to(self.device)
        if train:
            spt_fts_mse = spt_fts.unsqueeze(0)   # (1, 1, C, h, w)
            qry_fts_mse = qry_fts                # (1, C, h, w)
            proto_mse = []
            proto_mse.append(semantic_information)
            proto_mse_loss_epi = self.proto_mse(qry_fts_mse, preds_coarse, spt_fts_mse, proto_mse)
            mse_loss += proto_mse_loss_epi


        # mask prediction
        qry_fts_resized = F.interpolate(qry_weighted_fts, size=support_mask.shape[-2:], mode='bilinear')
        qry_similarity_fg = F.cosine_similarity(qry_fts_resized, semantic_information[..., None, None], dim=1) * self.scaler
        qry_similarity_bg = 1.0 - qry_similarity_fg
        qry_fts_fg = qry_fts_resized * qry_similarity_fg
        qry_fts_bg = qry_fts_resized * qry_similarity_bg
        pred_fg = self.decoder(qry_fts_fg)
        pred_bg = self.decoder(qry_fts_bg)
        pred = torch.cat([pred_bg, pred_fg], dim=1)

        ################################ Semantic Information Guided Correlation Estimation ###########################  
        # spt_hidden_fts.append(spt_weighted_fts)
        # qry_hidden_fts.append(qry_weighted_fts)
       
        # spt_fts = self.mask_feature(spt_hidden_fts, support_mask.clone())
        # corr = Correlation.multilayer_correlation_new(qry_hidden_fts, spt_fts, self.stack_ids, semantic_information)
        
       
        # ##################################### Similarity Calculation ####################################### 
        # support_feats = self.mask_feature(support_feats, support_mask.clone())
        # corr = Correlation.multilayer_correlation_new(query_feats, support_feats, self.stack_ids, middle_prototypes)
        # #####################################################################################################

 
        ########################################## Decoder #########################################
        # logit_mask = self.hpn_learner(corr)
        ################################################################################################

        # if not self.use_original_imgsize:
        #     logit_mask = F.interpolate(logit_mask, size=img_size, mode='bilinear', align_corners=True)

        return pred, loss_spt_middle, loss_qry_middle, preds_coarse, align_loss, mse_loss
    
    
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
            

        # visualize_segmentation_tensor(logit_mask)
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

    def weighted_protos(self, fts, pred, type): 
        if type == 'mean':
            proto = torch.sum(fts * pred[None, ...], dim=(-2, -1)) \
                     / (pred[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C
        elif type == 'median': 

            B, C, H, W = fts.shape
            fts_flat = fts.view(B, C, -1)    # (B, C, H*W)
            pred_flat = pred.view(B, 1, -1)  # (B, 1, H*W)
        
            fts_masked = fts_flat * pred_flat  # (1, C, H*W)

            # remove zero elements 
            sum_along_c = fts_masked.sum(dim=1) # (1, H*W)
            nonzero_mask = sum_along_c != 0     # (1, H*W)
            expanded_mask = nonzero_mask.unsqueeze(1)
            expanded_mask = expanded_mask.expand(-1, fts_masked.shape[1], -1)

            fts_masked_nonzero = fts_masked[expanded_mask].reshape(1, fts_masked.shape[1], -1)  # (1, C, Nonzero_num)
            
            _, _, dimension = fts_masked_nonzero.shape

            if dimension != 0:
                fts_masked_nonzero_sum = torch.sum(fts_masked_nonzero, dim=1)  # (1, Nonzero_num)
                
                # get median id
                values = fts_masked_nonzero_sum[0]
                _, indices = torch.sort(values)
                n = len(values)
                if n % 2 == 0:
                    median_pos = n // 2 - 1 
                else: 
                    median_pos = n // 2 
                    
                median_index = indices[median_pos].item()
                proto = fts_masked_nonzero[:, :, median_index]
            else:
                # employ mean-value based prototype for replacing
                proto = torch.sum(fts * pred[None, ...], dim=(-2, -1)) \
                     / (pred[None, ...].sum(dim=(-2, -1)) + 1e-5)
            
        return proto
    
    def dynamic_selection(self, spt_mask, qry_mask, spt_fts, qry_fts, coff):

        """
        spt_mask: (1, H, W)
        qry_mask: (1, 1, H, W)
        spt_fts: (1, C, h, w)
        qry_fts: (1, C, h, w)
        coff: (1)
        """

        # Step 2: Semantic Information Gaining # 
        B, C, h, w = spt_fts.size()
        qry_mask = qry_mask.squeeze(0)  # (1, H, W)
        B, H, W = qry_mask.size()

        spt_fts = F.interpolate(spt_fts, size=spt_mask.shape[-2:], mode='bilinear')
        qry_fts = F.interpolate(qry_fts, size=qry_mask.shape[-2:], mode='bilinear')

        spt_proto_mean = self.weighted_protos(spt_fts, spt_mask, type='mean')
        spt_proto_median = self.weighted_protos(spt_fts, spt_mask, type = 'median')

        qry_proto_mean = self.weighted_protos(qry_fts, qry_mask, type = 'mean')
        qry_proto_median = self.weighted_protos(qry_fts, qry_mask, type = 'median')

        proto_mean = self.middle_fusion(spt_proto_mean, qry_proto_mean)         # (1, 512)
        proto_median = self.middle_fusion(spt_proto_median, qry_proto_median)   # (1, 512)

        # Step 3: Dynamic Channels Selection # 

        ### channel randomization 
        _, C_mean = proto_mean.size()
        chosen_slices_num_1 = C_mean // 2  
        indices_mean = torch.randperm(proto_mean.size(1))[:chosen_slices_num_1]
        sampled_proto_mean = proto_mean[:, indices_mean]                       # (1, 256)
        
        _, C_median = proto_median.size()
        chosen_slices_num_2 = C_median // 2
        indices_median = torch.randperm(proto_median.size(1))[:chosen_slices_num_2] 
        sampled_proto_median = proto_median[:, indices_median]                 # (1, 256)

        # concatenate
        proto = torch.cat([sampled_proto_mean, sampled_proto_median], dim=1)  # (1, 512)
     
        # proto_mean_chunks = proto_mean.chunk(2, dim=1)  
        # proto_median_chunks = proto_median.chunk(2, dim=1)
        # proto = torch.cat([
        #     torch.cat([proto_mean_chunks[0], proto_median_chunks[0]], dim=1),
        #     torch.cat([proto_mean_chunks[1], proto_median_chunks[1]], dim=1)
        # ], dim=1)   # (1, 1024)

        _, feat_dim = proto.shape
        
        channel_weights = self.channel_importance(proto)
        
        num_channels_to_keep = int(feat_dim * coff)
        _, indices = torch.topk(channel_weights, num_channels_to_keep, dim=1)
        indices = indices.sort(dim=1)[0]
        proto = torch.gather(proto, 1, indices)  # (1, N*)
        
        # adaptive pooling for channels
        proto = proto.unsqueeze(1)
        proto = self.adaptive_pool(proto)   
        proto = proto.squeeze(1)


        return proto

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        """
        supp_fts: (1, 1, C, h', w')
        fore_mask: (1, 1, H, W)
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        
        # Learn the threshold
        mask_ = fore_mask.squeeze(0)  # (1, H, W)
        fts_ = supp_fts.squeeze(0)    # (1, C, h', w')
        fts_ = F.interpolate(fts_, size=mask_.shape[-2:], mode='bilinear')
        t = self.learnThreshold_a(fts_)
        t = torch.flatten(t, 1)
        t = self.learnThreshold_b(t)

        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [self.getFeatures(qry_fts, pred_mask[way + 1])]
                fg_prototypes = self.getPrototype([qry_fts_])

                # Get predictions
                supp_pred = self.getPred(supp_fts[way, [shot]], fg_prototypes[way], t)  # N x Wa x H' x W'
                supp_pred = F.interpolate(supp_pred[None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)


                # Combine predictions of different feature maps
                preds = supp_pred
                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss
    
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
    
    def proto_mse(self, qry_fts, pred, fore_mask, supp_prototypes):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss_sim = torch.zeros(1).to(self.device)
        for way in range(n_ways): 
            if way in skip_ways:
                continue
            for shot in range(n_shots):
                # Get prototypes
                qry_fts_ = [[self.getFeatures(qry_fts, pred_mask[way + 1])]]

                fg_prototypes = self.getPrototype(qry_fts_)

                fg_prototypes_ = torch.sum(torch.stack(fg_prototypes, dim=0), dim=0)
                supp_prototypes_ = torch.sum(torch.stack(supp_prototypes, dim=0), dim=0)
                
                # Combine prototypes from different scales
                # fg_prototypes = self.alpha * fg_prototypes[way]
                # fg_prototypes = torch.sum(torch.stack(fg_prototypes, dim=0), dim=0) / torch.sum(self.alpha)
                # supp_prototypes_ = [self.alpha[n] * supp_prototypes[n][way] for n in range(len(supp_fts))]
                # supp_prototypes_ = torch.sum(torch.stack(supp_prototypes_, dim=0), dim=0) / torch.sum(self.alpha)

                # Compute the MSE loss

                loss_sim += self.criterion_MSE(fg_prototypes_, supp_prototypes_)
            
        return loss_sim
            

        

        

        



