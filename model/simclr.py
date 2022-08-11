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
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.resnet import resnet50


class SimCLR(torch.nn.Module):
    def __init__(self):
        super(SimCLR, self).__init__()

        # Encoder
        self.f = []
        for name, module in resnet50().named_children():
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d) and not isinstance(module, nn.AdaptiveAvgPool2d):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, 128))

    def forward(self, x):
        x = self.f(x)
        x = self.avg_pool(x)
        h = torch.flatten(x, start_dim=1)
        z = self.g(h)

        return F.normalize(h, dim=-1), F.normalize(z, dim=-1)
