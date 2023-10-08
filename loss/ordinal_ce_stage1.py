"""
Ordinal Entropy regularizer
"""
import torch
import torch.nn.functional as F
import random


def ordinal_entropy_stage1(imgfeatures,txtfeatures,gl_text_features, gt):
    _distance = torch.cdist(imgfeatures, txtfeatures, p=2)
    _distance = up_triu(_distance)

    t_distance = torch.cdist(txtfeatures, txtfeatures, p=2)
    t_distance = up_triu(t_distance)


    _weight = torch.cdist((gt.float()).view(-1,1), (gt.float()).view(-1,1), p=2)
    _weight = up_triu(_weight)
    _max = torch.max(_weight)
    _min = torch.min(_weight)
    _weight = ((_weight - _min) / _max)

    """
    L_d = - mean(w_ij ||z_{c_i} - z_{c_j}||_2)
    """
    _distance = _distance * _weight 
    t_distance = t_distance * _weight
    L_d = -torch.mean(_distance) 
    
    L_center = -torch.mean(t_distance)
    _features_center = gl_text_features
    _features = imgfeatures - _features_center
    _features = _features.pow(2)
    _tightness = torch.sum(_features, dim=1)
    L_t = torch.mean(_tightness)
    
    return  L_d + 0.5 * L_t + L_center * 0.5
    # return 0.5 * L_t - L_d


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