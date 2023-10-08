"""
Ordinal Entropy regularizer
"""
import torch
import torch.nn.functional as F
import random
from torch import nn


def ordinal_entropy(logits, img_features, txt_features, gt, age_range=[16,78]):
    age_list = torch.arange(age_range[0], age_range[1]).float().to(logits.device)
    _weight = euclidean_dist((gt.float()).view(-1,1), (age_list).view(-1,1))
    _max = torch.max(_weight)
    _min = torch.min(_weight)
    _weight = ((_weight - _min) / _max)
    it_dis = euclidean_dist(img_features, txt_features)
    _distance = it_dis * _weight
    L_d = -torch.mean(_distance)

    # img_features = logits[:,gt]
    gt_text_features = txt_features[gt-age_range[0]]
    _features_center = gt_text_features
    _features = img_features - _features_center
    _features = _features.pow(2)
    _tightness = torch.sum(_features, dim=1)
    _tightness = torch.mean(_tightness)
    return  _tightness + L_d * 1


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.sum(torch.pow(x, 2), dim=1)
    xx = xx.unsqueeze(1).expand(m, n)
    yy = torch.sum(torch.pow(y, 2), dim=1).unsqueeze(1).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def up_triu(x):
    # return a flattened view of up triangular elements of a square matrix
    n, m = x.shape
    assert n == m
    _tmp = torch.triu(torch.ones(n, n), diagonal=1).to(torch.bool)
    return x[_tmp]