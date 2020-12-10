import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, List, Dict

class GaussianDropout(nn.Module):
    def __init__(self, drop_type, drop_rate):
        self.drop_rate = drop_rate
        self.keep_rate = 1.0-self.drop_rate
        self.drop_type = drop_type
    
    def forward(self, x, noise_type='normal'):
        if not self.training:
            return x

        if dropout_type == 'Bernoulli':
            if noise_type == 'normal':
                noise = torch.bernoulli(torch.ones_like(x)*self.keep_rate).to(x.device) / self.keep_rate
            elif noise_type == 'same':
                size = (1,x.size(-1))
                noise = torch.bernoulli(torch.ones(1,x.size(-1))*self.keep_rate).to(x.device) / self.keep_rate
                noise = noise.expand_as(x)
        elif dropout_type == 'Gaussian':
            mean = 1.0
            std = math.sqrt(drop_rate/(1.0-drop_rate))
            if noise_type == 'normal':
                noise = torch.randn_like(x, requires_grad=False).to(x.device) * std + mean
            elif noise_type == 'same':
                size = (1,x.size(-1))
                noise = torch.randn(size, requires_grad=False).to(x.device) * std + mean
                noise = noise.expand_as(x)
        
        return x * noise


class GaussianDropout(nn.Module):
    def __init__(self, drop_rate):
        self.drop_rate = drop_rate
        self.mean = 1.0
        self.std = math.sqrt(drop_rate/(1.0-drop_rate))
    
    def forward(self, x):
        if self.training:
            # gaussian_noise = torch.normal(self.mean, self.std, x.size()).to(x.device)
            gaussian_noise = torch.randn_like(x, requires_grad=False).to(x.device) * self.std + self.mean
            return x * gaussian_noise
        else:
            return x

class MCdropClassifier(nn.Module):
    def __init__(self, 
                backbone: nn.Module,
                num_classes: int,
                bottleneck_dim: Optional[int] = 256,
                classifier_width: Optional[int] = 256,
                dropout_rate: Optional[float] = 0.5,
                dropout_type: Optional[str]='Bernoulli'):
        super(MCdropClassifier, self).__init__()

        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim
        self.classifier_width = classifier_width
        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type

        self.backbone = backbone
        self.backbone_drop = self._make_dropout(dropout_rate, dropout_type)
        
        self.bottleneck_fc = nn.Linear(backbone.out_features, bottleneck_dim)
        self.bottleneck_act = nn.Sequential(
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.bottleneck_drop = self._make_dropout(dropout_rate, dropout_type)
        self.bottleneck_layer =  nn.Sequential(
            self.bottleneck_fc,
            self.bottleneck_act,
            self.bottleneck_drop
        )

        self.classifier_fc = nn.Linear(bottleneck_dim, classifier_width)
        self.classifier_act = nn.ReLU()
        self.classifier_drop = self._make_dropout(dropout_rate, dropout_type)
        self.classifier_layer =  nn.Sequential(
            self.classifier_fc,
            self.classifier_act,
            self.classifier_drop
        )

        self.predition_layer = nn.Linear(classifier_width, num_classes)
    
    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self.classifier_width
    
    def _make_dropout(self, dropout_rate, dropout_type) -> nn.Module:
        if dropout_type == 'Bernoulli':
            return nn.Dropout(dropout_rate)
        elif dropout_type == 'Gaussian':
            return GaussianDropout(dropout_rate)
        else:
            raise ValueError(f'Dropout type not found')
    
    def activate_dropout(self):
        self.bottleneck_drop.train()
        self.classifier_drop.train()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(x)
        x = x.view(-1, self.backbone.out_features)
        hidden = self.bottleneck_layer(x)
        # hidden = self.classifier_layer(hidden)
        pred = self.predition_layer(hidden)
        return pred, hidden
    
    def backbone_forward(self, x) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(-1, self.backbone.out_features)
        return x
    
    def head_forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.bottleneck_layer(x)
        # hidden = self.classifier_layer(hidden)
        pred = self.predition_layer(hidden)
        return pred, hidden
    
    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck_layer.parameters(), "lr_mult": 1.},
            {"params": self.classifier_layer.parameters(), "lr_mult": 1.},
            {"params": self.predition_layer.parameters(), "lr_mult": 1.},
        ]
        return params
