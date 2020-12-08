import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def VariationRatio(sampled_y, device):
    # sampled_y: (sample_num, batch_size/dataset_size, num_class)
    sample_num, data_size, num_class = sampled_y.size()
    _, indices = torch.max(sampled_y, -1)
    distribution_count = torch.zeros(data_size, num_class).to(deivce)
    distribution = distribution_count.scatter_add(1, indices.t(), torch.ones(num_class, 1).to(device).expand(num_class, sample_num))
    variation_ratio = 1.0 - distribution.max(1)[0] / sampled_num
    return variation_ratio

def Entropy(p):
    entropy = -torch.sum(p*torch.log(p+1e-6), -1)
    return entropy

def PredictiveEntropy(sampled_y):
    y = sampled_y.mean(0)
    predictive_entropy = Entropy(y)
    return predictive_entropy

def MutualInfo(sampled_y):
    predictive_entropy = PredictiveEntropy(sampled_y)
    mutual_info = predictive_entropy - Entropy(sampled_y).mean(0)
    return mutual_info
