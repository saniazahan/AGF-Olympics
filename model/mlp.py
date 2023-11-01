#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 19:21:16 2022

@author: 22905553
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.activation import activation_factory


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', dropout=0):
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            if dropout > 0.001:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=1))
            self.layers.append(nn.BatchNorm2d(channels[i]))
            self.layers.append(nn.ReLU(inplace=True))

    def forward(self, x):
        # Input shape: (N,C,T,V)
        for layer in self.layers:
            x = layer(x)
        return x
    
if __name__ == "__main__":
    # For debugging purposes
#     cd /home/uniwa/students3/students/22905553/linux/phd_codes/Olympics/Olympic_code/
    import sys
    sys.path.append('..')
    import thop
    from thop import clever_format
    
    mlp = MLP(6400, 512, 1)
    
    macs, params = thop.profile(seq_model, inputs=(torch.randn(1,224,6400),[125],), verbose=False)
    macs, params = clever_format([macs, params], "%.2f")
 
    x = torch.randn(1, 125, 6400)#.cuda()
    lens = [len(xi) for xi in x]
    out = seq_model.forward(x,lens)