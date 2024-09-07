r""" Provides functions that builds/manipulates correlation tensors """
import torch
import torch.nn.functional as F


class Correlation:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb) # 相似度计算 
            corr = corr.clamp(min=0)
            corrs.append(corr)

        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()


        return [corr_l4, corr_l3, corr_l2]

    @classmethod
    def multilayer_correlation_new(cls, query_feats, support_feats, stack_ids, prototypes):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat, prototype) in enumerate(zip(query_feats, support_feats, prototypes)):
            
            """
            query_feat: (1, C, h, w)
            support_feat: (1, C, h, w)
            prototype: (1, C)
            """

            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)  # (1, C, h*w)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)  # (1, C, h*w)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb) # 相似度计算 
            corr = corr.clamp(min=0)

            corr = corr.view(bsz, ha, wa, hb*wb).permute(0, 3, 1, 2)   # (1, hb*wb, ha, wa)  
            
            prototype_similarity = F.cosine_similarity(query_feat.view(bsz, ch, ha, wa), prototype.unsqueeze(-1).unsqueeze(-1), dim=1)  # (1, ha, wa)
            
            similarity_enhanced_corr = corr * prototype_similarity.unsqueeze(1)
            similarity_enhanced_corr = similarity_enhanced_corr.permute(0, 2, 3, 1).view(bsz, ha, wa, hb, wb)   # (1, ha, wa, hb, wb)
            similarity_enhanced_corr = similarity_enhanced_corr.clamp(min=0)

            corrs.append(similarity_enhanced_corr)

        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]

           
            

            

            
            


            
            


        return True 




