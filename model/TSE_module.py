#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 11:14:17 2022

@author: 22905553
"""

from torch import nn


class TSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(TSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        N, C, T, v = x.size()
        x = x.permute(0,2,1,3)
        y = self.avg_pool(x).view(N, T)
        y = self.fc(y).view(N, T, 1, 1)
        y = x * y.expand_as(x)
        y = y.permute(0,2,1,3)
        return y