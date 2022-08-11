# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
...
"""

__author__ = "..."
__email__ = "..."
__license__ = "..."
__version__ = "1.0"

# External modules
from torchvision.models.resnet import resnet50
import torch.nn.functional as F
from torch import nn
import torch

# Internal modules
from model.simclr import SimCLR


class CorrNet(torch.nn.Module):
    """
        Implementation of CorrNet
    """
    def __init__(self, feature_dim, pre_trained_encoder=None):
        super(CorrNet, self).__init__()

        # Encoder
        if pre_trained_encoder is None:
            self.f = []
            for name, module in resnet50().named_children():
                if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d) and not isinstance(module, nn.AdaptiveAvgPool2d):
                    self.f.append(module)
            self.f = nn.Sequential(*self.f)
        else:
            pre_trained_encoder_cornet = SimCLR()
            pre_trained_encoder_cornet.load_state_dict(torch.load(pre_trained_encoder))
            self.f = pre_trained_encoder_cornet.f
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection head
        self.g = nn.Sequential(nn.Linear(2048, 2048),
                               nn.BatchNorm1d(2048),
                               nn.ReLU(inplace=True),
                               nn.Linear(2048, 512),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim))

    def forward(self, x):
        x = self.f(x)
        x = self.avg_pool(x)
        h = torch.flatten(x, start_dim=1)
        z = self.g(h)

        return F.normalize(h, dim=-1), F.normalize(z, dim=-1)
