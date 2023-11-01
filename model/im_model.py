#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 13:04:29 2022

@author: 22905553
"""
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
import json
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
import math
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import copy 

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k) # torch.Size([1, 16, 8, 320])
        #print(k.shape)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)# calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,2560), stride=(1,2560))
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)   
        #q = self.maxpool(x2)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Im_Transformer(nn.Module):
            def __init__(self, d_model = 2560, heads = 8, N = 2):
                super(Im_Transformer, self).__init__()
                self.N = N
                self.layers = get_clones(EncoderLayer(d_model, heads), N)
                self.norm = Norm(d_model)
            def forward(self, x, mask=None):
                for i in range(self.N):
                    x = self.layers[i](x, mask)
                x = self.norm(x)
                return x
        
                
class SeqImModel(nn.Module):
            def __init__(self, seq_len, hidden_size, bidirectional, self_attention = False, dropout = 0.1):
                super(SeqImModel, self).__init__()
                
                self.self_attention = self_attention
                im_model = torchvision.models.efficientnet_b7(pretrained=True)
                self.features = nn.Sequential(
                    # stop at conv4
                    *list(im_model.children())[:-2] 
                    #*list(im_model.blocks.children())[:-1] #x3d
                )
                
                self.pool =  nn.AdaptiveAvgPool2d((1,1))
                self.hidden_size = hidden_size
                self.seq_len = seq_len
                if not self.self_attention:
                    self.gru = nn.GRU(2560, self.hidden_size, num_layers=1, bidirectional=bidirectional, batch_first=True)
                    self.emb = hidden_size * 2 * self.seq_len
                    self.fc = nn.Linear(self.emb, 1)
                else:
                    self.dropout = nn.Dropout(p=dropout)
                    max_len = 6115
                    d_model = 2560
                    pe = torch.zeros(max_len, d_model)
                    position = torch.arange(0, max_len).unsqueeze(1)
                    div_term = torch.exp(
                        torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
                        )
                    pe[:, 0::2] = torch.sin(position * div_term)
                    pe[:, 1::2] = torch.cos(position * div_term)
                    pe = pe.unsqueeze(0)
                    #print(pe.shape)
                    self.register_buffer("pe", pe)
                    self.transformer = Im_Transformer()
                    self.fc = nn.Sequential(            
                                nn.Linear(2560, hidden_size*4),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_size*4, hidden_size),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_size, 1)           
                            )   
                
            def forward(self, inputs):
                N, T, C, W, H = inputs.size()
                encoder_hidden = torch.Tensor().to(device)
                
                for it in range(T):
                    #print(inputs[:, it].shape)    # torch.Size([1, 3, 224, 224])
                    x = self.features(inputs[:, it])  # torch.Size([1, 2560, 7, 7])
                    #x = x.view(N, 2560, -1)
                    #print(x.shape)
                    x = torch.unsqueeze(torch.squeeze(torch.squeeze(self.pool(x), dim=2), dim=2), dim=1) # torch.Size([1, 1, 2560])
                    
                    #print(x.shape)
                    #print(it)
                    if not self.self_attention:
                        if it == 0:
                            enout_tmp, hidden_tmp = self.gru(x)
                        else:
                            enout_tmp, hidden_tmp = self.gru(x, hidden_tmp)
                        encoder_hidden = torch.cat((encoder_hidden, enout_tmp),1)
                    else:
                        
                        encoder_hidden = torch.cat((encoder_hidden, x),1)
                print(encoder_hidden.shape)     
                if not self.self_attention:
                    encoder_hidden = encoder_hidden.view(N, -1)
                    #print(self.emb)
                    out = self.fc(encoder_hidden)
                    out = out[:,0]
                    return out, encoder_hidden
                else:
                    x = self.dropout(encoder_hidden + self.pe[:, : encoder_hidden.size(1)].requires_grad_(False))
                    #print(x.shape)
                    x = self.transformer(x)
                    x = x.mean(dim=1).squeeze(dim=1)
                    #print(x.shape)
                    x = self.fc(x)
                    return x[:,0], encoder_hidden


if __name__ == "__main__":
    # For debugging purposes
    import thop
    from thop import clever_format
#     cd /home/uniwa/students3/students/22905553/linux/phd_codes/Olympics/Olympic_code/
    model = SeqImModel(seq_len=16, hidden_size=128, bidirectional=True, self_attention = True)
    
    x = torch.rand(1,16,3,224,224)
    preds = model(x)
    
    #macs, params = thop.profile(model, inputs=(torch.randn(1,112,3,224,224),), verbose=False)
    #macs, params = clever_format([macs, params], "%.2f")
