import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import copy

class PseudoLabeling(object):
    def __init__(self, size, num_classes, alpha, gamma_c, threshold, device):
        self.size = size
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma_c = gamma_c
        self.threshold = threshold
        self.t = 0
        self.device = device
        self.p = torch.zeros((size, num_classes), device=device)
        self.p_history = torch.zeros((size,num_classes), device=device)
        self.ct = torch.zeros(size, device=device)
        self.weight = torch.zeros(size, device=device)

    def copy_history(self):
        self.p_history = copy.deepcopy(self.p)

    def threshold_weight(self):
        confidence, _ = torch.max(self.p, dim=1)
        weight = confidence.masked_fill(confidence < self.threshold, -np.inf)
        self.weight = weight

    def difference_to_one_weight(self):
        confidence, index = torch.max(self.p, dim=1)
        difference_to_one = 1.0-confidence
        self.weight = -1.0*difference_to_one

    def entropy_weight(self):
        entropy = -torch.sum(self.p*torch.log(self.p+1e-6), 1)
        self.weight = -1.0*entropy

    def time_consistency_weight(self):
        KL_div = torch.sum(self.p_history * (torch.log(self.p_history) - torch.log(self.p)), 1)
        confidence1, _ = torch.max(self.p_history, dim=1)
        confidence2, _ = torch.max(self.p, dim=1)
        log_odds_ratio = torch.abs(torch.log(confidence1) - torch.log(confidence2))
        at = (KL_div + log_odds_ratio)
        self.t += 1
        self.ct = self.gamma_c*self.ct+ (1-self.gamma_c)*(-at)
        self.weight = self.ct/(1.0-pow(self.gamma_c, self.t))

    def update_p(self, prediction, index):
        self.p[index] = prediction

    def EMA_update_p(self, prediction, index, epoch):
        if epoch>=10:
            self.p[index] = self.alpha*self.p[index] + (1.0-self.alpha)*prediction
        else:
            self.p[index] = prediction

    def get_weight(self, index):
        return self.weight[index]

    def get_hard_pseudo_label(self, index):
        _, hard_label = torch.max(self.p[index], dim=1)
        return hard_label

    def get_soft_pseudo_label(self, index):
        return self.p[index]

    def count_t(self):
        count_t = [0]*self.num_classes
        confidence, index = torch.max(self.p, dim=1)
        for i in range(index.shape[0]):
            count_t[int(index[i].item())] += 1
        return count_t

