#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:24:54 2022

@author: 22905553
"""
import numpy as np
import torch
import torch.nn as nn
from model.activation import activation_factory
from model.Random_Drops import *
#from graph.olympic import AdjMatrixGraph
#from graph.tools import k_adjacency, normalize_adjacency_matrix
from model.adjGraph import adjGraph


graph_hop = adjGraph(layout='olympic', strategy='spatial', max_hop=3)
A = torch.tensor(graph_hop.A, dtype=torch.float32, requires_grad=False)

class norm_data(nn.Module):
    def __init__(self, dim= 64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim* 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x

class embed(nn.Module):
    def __init__(self, dim = 3, dim1 = 128, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x




class gcn_spa(nn.Module):
    def __init__(self, in_feature, out_feature, bias = False):
        super(gcn_spa, self).__init__()
        self.bn = nn.BatchNorm2d(out_feature)
        self.relu = nn.ReLU()
        self.w = cnn1x1(in_feature, out_feature, bias=False)
        self.w1 = cnn1x1(in_feature, out_feature, bias=bias)


    def forward(self, x1, g):
        x = x1.permute(0, 3, 2, 1).contiguous()
        x = g.matmul(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.w(x) + self.w1(x1)
        x = self.relu(self.bn(x))
        return x


    
class GTCN(nn.Module):
    def __init__(self, inchannel, outchannel, bias=True):
        super(GTCN, self).__init__()
        num_joint = 25
        seg = 500
        bs=4
        self.spa = self.one_hot(bs, num_joint, seg)
        self.spa = self.spa.permute(0, 3, 2, 1).cuda()
        self.tem = self.one_hot(bs, seg, num_joint)
        self.tem = self.tem.permute(0, 3, 1, 2).cuda()
        self.spa_embed = embed(num_joint, inchannel, norm=False, bias=bias)
        self.tem_embed = embed(seg, outchannel, norm=False, bias=bias)
        self.compute_g1 = compute_g_spa(outchannel, outchannel, bias=bias)
        self.gcn1 = gcn_spa(outchannel, outchannel, bias=bias)
        self.cnn = local(outchannel, outchannel, bias=bias)
        
    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot
        
    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0,1,3,2)
        #print(self.tem.shape)
        #self.tem = self.tem.expand(N,-1,-1,-1)
        #print(self.tem.shape)
        tem1 = self.tem_embed(self.tem)
        #print(tem1.shape)
        self.spa = self.spa.expand(N,-1,-1,-1)
        spa1 = self.spa_embed(self.spa)
        x = torch.cat([x,spa1], 1)
        #print(x.shape)
        g = self.compute_g1(x)
        x = self.gcn1(x, g)
        #print(x.shape)
        x = x + tem1
        x = self.cnn(x)
        x = x.permute(0,1,3,2)
        return x

class local(nn.Module):
    def __init__(self, dim1 = 3, dim2 = 3, bias = False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((25, 250))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class compute_g_spa(nn.Module):
    def __init__(self, dim1 = 64 *3, dim2 = 64*3, bias = False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):

        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous()
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous()
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g
        
class Xspace_time_Pose_Module(nn.Module):
    ''' 
    This class models the pose, learns a learnable adjacency matrix
    Focus: High speed higher, low speed lower
           Jump height, dictance covered in y axis
           Straightness of body parts
    '''
    def __init__(self, in_channel, out_channel, num_scales = 6, window_size=25, window_stride=2, frame = 250, bias=True, act_type='relu', residual=True, **kwargs):
        super(Xspace_time_Pose_Module, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((frame, 25))
        #self.compute_g1 = compute_g_spa(in_channel, out_channel, bias=bias)
        self.w = cnn1x1(in_channel, out_channel, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        #self.gtcn = GTCN(in_channel, out_channel)
        
    def forward(self, inputs):
        x = self.maxpool(inputs)
        #g = self.compute_g1(x.permute(0,1,3,2))
        #x = g.matmul(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.w(x)
        
        return x
    

if __name__ == "__main__":
    # For debugging purposes
#     cd /home/uniwa/students3/students/22905553/linux/phd_codes/Olympics/Olympic_code/
    import sys
    sys.path.append('..')
    import thop
    from thop import clever_format
    import numpy as np
        
    x1 = torch.randn(1, 48, 500, 25)#.cuda()
    
    g = torch.randn(1, 250, 25, 25)#.cuda()
    
    x3 = torch.randn(1, 192, 32, 25)#.cuda()
    
    model = Xspace_time_Pose_Module(in_channel=48, out_channel=192, num_scales=2, window_size=25, window_stride=16, frame = [500,250])
    pred = model(x1)    

    macs, params = thop.profile(model, inputs=(x1,), verbose=False)
    macs, params = clever_format([macs, params], "%.2f")
    