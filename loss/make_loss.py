# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
import torch
from torch import nn

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0
    def forward(self, text_features, image_features, t_label, i_targets): 
        batch_size = text_features.shape[0] 
        batch_size_N = image_features.shape[0] 
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device) 

        logits = torch.div(torch.matmul(text_features, image_features.T),self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach() 
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 
        loss = - mean_log_prob_pos.mean()

        return loss

def make_loss(cfg, num_classes, age_range=[16,78]):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=age_range[1]-age_range[0], feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=age_range[1]-age_range[0])  # label smooth
        print("label smooth on, numclasses:", age_range[1]-age_range[0])

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        mae_loss = nn.L1Loss()
        # def loss_func(score, feat, target, target_cam, i2tscore = None):
        #     if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        #         if cfg.MODEL.IF_LABELSMOOTH == 'on':
        #             if isinstance(score, list):
        #                 ID_LOSS = [xent(scor, target) for scor in score[0:]]
        #                 ID_LOSS = sum(ID_LOSS)
        #             else:
        #                 ID_LOSS = xent(score, target)

        #             if isinstance(feat, list):
        #                 TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
        #                 TRI_LOSS = sum(TRI_LOSS) 
        #             else:   
        #                 TRI_LOSS = triplet(feat, target)[0]
                    
        #             loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

        #             if i2tscore != None:
        #                 I2TLOSS = xent(i2tscore, target)
        #                 loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                        
        #             return loss
        #         else:
        #             if isinstance(score, list):
        #                 ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
        #                 ID_LOSS = sum(ID_LOSS)
        #             else:
        #                 ID_LOSS = F.cross_entropy(score, target)

        #             if isinstance(feat, list):
        #                     TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
        #                     TRI_LOSS = sum(TRI_LOSS)
        #             else:
        #                     TRI_LOSS = triplet(feat, target)[0]

        #             loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    
        #             if i2tscore != None:
        #                 I2TLOSS = F.cross_entropy(i2tscore, target)
        #                 loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss


        #             return loss

        def loss_func(target, logits, cls_score):
            # I2TLOSS_FUNC = SupConLoss(target.device)
            # ID_LOSS = F.cross_entropy(cls_score, target-16) 
            # labels = torch.cat([target-16, target-16], dim=0)
            I2TLOSS = F.cross_entropy(logits, target-16) * cfg.MODEL.I2T_LOSS_WEIGHT
            CLSLOSS = F.cross_entropy(cls_score, target-16) * cfg.MODEL.I2T_LOSS_WEIGHT
            # probs = logits.softmax(dim=-1)
            # pred = torch.sum(probs * torch.arange(16, 78).cuda(), dim=1) 
            # loss_mae = 50 * mae_loss(pred/100., target.float()/100.)
            loss = I2TLOSS #+ ID_LOSS #+ loss_mae
            return loss, {"loss_i2t": I2TLOSS} #, "loss_mae": loss_mae}
           

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


