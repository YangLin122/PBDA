from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

#def get_pseudo_labels(y1_t, y2_t):
#    r = random.random()
#    if (r>0.5):
#        confidence, pseudo_labels = torch.max(y1_t, 1)
#    else:
#        confidence, pseudo_labels = torch.max(y2_t, 1)
#    return confidence, pseudo_labels


#def get_weights(confidence, pseudo_labels,  threshold):
#    temp = torch.zeros_like(confidence)
#    weights = torch.where(confidence>threshold, confidence, temp)
#    return weights

def jointdisagreement(y1_s, y1_t, y2_s, y2_t, labels_s, labels_t, weights_t, loss_type='ce'):
    if loss_type == 'mae':
        labels_s = F.one_hot(labels_s, y1_s.size(1)).float()
        labels_t = F.one_hot(labels_t, y1_s.size(1)).float()
        y1_s, y2_s = F.softmax(y1_s, dim=1), F.softmax(y2_s, dim=1)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        t1 = (torch.abs(y1_s-labels_s).sum(1)*torch.abs(y2_s-labels_s).sum(1)).mean()
        t2 = (torch.abs(y1_t-labels_t).sum(1)*torch.abs(y2_t-labels_t).sum(1)*weights_t).sum()
        return torch.abs(t1-t2)
    elif loss_type == 'mse':
        labels_s = F.one_hot(labels_s, y1_s.size(1)).float()
        labels_t = F.one_hot(labels_t, y1_s.size(1)).float()
        y1_s, y2_s = F.softmax(y1_s, dim=1), F.softmax(y2_s, dim=1)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        t1 = (torch.pow(y1_s-labels_s, 2).sum(1)*torch.pow(y2_s-labels_s, 2).sum(1)).mean()
        t2 = (torch.pow(y1_t-labels_t, 2).sum(1)*torch.pow(y2_t-labels_t, 2).sum(1)*weights_t).sum()
        return torch.abs(t1-t2)
    elif loss_type == 'ce':
        t1 = (F.cross_entropy(y1_s, labels_s, reduction='none') * F.cross_entropy(y2_s, labels_s, reduction='none')).mean()
        t2 = (F.cross_entropy(y1_t, labels_t, reduction='none') * F.cross_entropy(y2_t, labels_t, reduction='none') * weights_t).sum()
        return torch.abs(t1-t2)
    elif loss_type == 'kl':
        # labels_t is soft label 
        labels_s = F.one_hot(labels_s, y1_s.size(1)).float()
        labels_t = F.one_hot(labels_t, y1_s.size(1)).float()
        y1_s, y2_s = F.log_softmax(y1_s, dim=1), F.log_softmax(y2_s, dim=1)
        y1_t, y2_t = F.log_softmax(y1_t, dim=1), F.log_softmax(y2_t, dim=1)
        
        t1 = (F.kl_div(y1_s, labels_s, reduction='none').sum(1) * F.kl_div(y2_s, labels_s, reduction='none').sum(1)).mean()
        t2 = (F.kl_div(y1_t, labels_t, reduction='none').sum(1) * F.kl_div(y2_t, labels_t, reduction='none').sum(1) * weights_t).sum()
        return torch.abs(t1-t2)
    elif loss_type == 'bce':
        # labels_t is soft label 
        labels_s = F.one_hot(labels_s, y1_s.size(1)).float()
        labels_t = F.one_hot(labels_t, y1_s.size(1)).float()
        y1_s, y2_s = F.softmax(y1_s, dim=1), F.softmax(y2_s, dim=1)
        y1_t, y2_t = F.softmax(y1_t, dim=1), F.softmax(y2_t, dim=1)
        
        t1 = (F.binary_cross_entropy(y1_s, labels_s, reduction='none').sum(1) * F.binary_cross_entropy(y2_s, labels_s, reduction='none').sum(1)).mean()
        t2 = (F.binary_cross_entropy(y1_t, labels_t, reduction='none').sum(1) * F.binary_cross_entropy(y2_t, labels_t, reduction='none').sum(1) * weights_t).sum()
        return torch.abs(t1-t2)
    else:
        raise ValueError(f'loss type not found')

def CE_disagreement(y1_s, y1_t, y2_s, y2_t, labels_s, labels_t, weights_t):
    t1 = F.cross_entropy(y1_s * y2_s, labels_s, reduction='mean')
    t2 = (F.cross_entropy(y1_t * y2_t, labels_t, reduction='none') * weights_t).sum()
    return torch.abs(t2-t1)

def SQ_disagreement(y1_s, y1_t, y2_s, y2_t, onehot_l_s, soft_l_t, weights_t):
    t1 = 0.5 * torch.abs((torch.sum(F.mse_loss(y1_t, y2_t, reduction='none'), dim=1) * weights_t).sum()
                         -(torch.sum(F.mse_loss(y1_s, y2_s, reduction='none'), dim=1)).mean())
    xt = torch.sqrt(F.mse_loss(y1_t, soft_l_t, reduction='none')) * torch.sqrt(F.mse_loss(y2_t, soft_l_t, reduction='none'))
    xs = torch.sqrt(F.mse_loss(y1_s, onehot_l_s, reduction='none')) * torch.sqrt(F.mse_loss(y2_s, onehot_l_s, reduction='none'))
    t2 = torch.abs((torch.sum(xt, dim=1) * weights_t).sum() - torch.sum(xs, dim=1).mean())
    return t1+t2
