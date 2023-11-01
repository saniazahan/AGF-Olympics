#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 20:03:35 2021

@author: 22905553
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils import import_class, count_params
from model.activations import Swish


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class Attention_Layer(nn.Module):
    def __init__(self, out_channel, max_frame, att_type, act='swish', **kwargs):
        super(Attention_Layer, self).__init__()

        __attention = {
            'stja': ST_Joint_Att,
            'tja': Temporal_Att
        }

        self.att = __attention[att_type](out_channel, max_frame, **kwargs)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = Swish()
        self.apply(weights_init)
        
    def forward(self, x):
        res = x
        x = x * self.att(x)
        return self.act(self.bn(x) + res)


class ST_Joint_Att(nn.Module):
    def __init__(self, channel, max_frame, reduct_ratio=3, bias=True, **kwargs):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)
        #self.max_pool_v = nn.AdaptiveMaxPool2d((1, 25))
        #self.max_pool_t = nn.AdaptiveMaxPool2d((max_frame, 1))

    def forward(self, x):
        N, C, T, V = x.size()
        #print(x.shape)
        #pool_t = self.max_pool_t(x)
        #print(pool_t.shape)
        x_t = x.mean(3, keepdims=True)  # N, C, T, 1
        #print(x_t.shape)
        #x_t = x_t + pool_t
        #pool_v = self.max_pool_v(x).transpose(2, 3)
        #print(pool_v.shape)        
        x_v = x.mean(2, keepdims=True).transpose(2, 3)  # N, C, V, 1
        #print(x_v.shape)
        #x_v = x_v + pool_v
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        return x_att


class Temporal_Att(nn.Module):
    def __init__(self, channel, **kwargs):
        super(Temporal_Att, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((25, 1))
        #self.max_pool = nn.AdaptiveMaxPool2d((1,25))
        self.conv = nn.Conv2d(
            channel, channel, kernel_size=(9, 1), padding=(4, 0))

    def forward(self, x):
        #x = x.transpose(1, 2)
        # print(self.avg_pool(x).shape)
        x = x * self.avg_pool(x)
        # print(x.shape)
        return self.conv(x)

#dy = torch.randn(12,64,300,25)
#tem = Temporal_Att(64)
#a = tem(dy)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=310):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i)/d_model)))
                if i+1 < d_model:
                    pe[pos, i + 1] = \
                        math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # print(x.shape)
        # add constant to embedding
        seq_len = x.size(1)
        batch_size = x.size(0)
        num_feature = x.size(2)
        # print(self.pe.shape)
        z = Variable(self.pe[:, :seq_len], requires_grad=False)
        # print(z.shape)
        #z = z.unsqueeze(-1).unsqueeze(-1)
        # print(z.shape)
        z = z.expand(batch_size, seq_len, num_feature)
        # print(z.shape)
        x = x + z
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # print(x.shape)
        #print(self.fn(x, **kwargs).shape)
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim=75, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # print(*x.shape)
        N, _, h = *x.shape, self.heads
        # print(x.shape)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # print(qkv.shape)
        q, k, v = map(lambda t: rearrange(t, 'N d -> N d'), qkv)
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        dots = einsum('n d, n d -> n d', q, k) * self.scale
        # print(dots.shape)
        attn = dots.softmax(dim=-1)
        # print(attn.shape)
        out = einsum('n d, n d -> n d', attn, v)
        # print(out.shape)
        #out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # print(out.shape)
        return out
# att=Attention()
#a = att(m)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads,
                         dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(
                    dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x):
        x_mod = []
        # print(x.shape)
        for i in range(x.shape[0]):
            for attn, ff in self.layers:
                # print(x[i,:,:].shape)
                # print(attn)
                a = attn(x[i, :, :])
                # print(a.shape)
                a = ff(a)
                # print(a.shape)
            x_mod.append(a)
        # print(len(x_mod))
        # print(x_mod.shape)
        x_mod = torch.stack(x_mod, dim=0)
        # print(x_mod.shape)
        return x_mod


class Action_Transformer(nn.Module):
    def __init__(self, num_features, dim, depth, heads, mlp_dim, pool='mean', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super(Action_Transformer, self).__init__()
        self.num_features = num_features
        #self.num_frames = num_frames
        self.pos_embedding = PositionalEncoder(self.num_features)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        #x = self.to_embedding(inp)
        N, C, T, V = x.shape
        x = x.view(N, T, C*V)
        # print(x.shape)

        x = self.pos_embedding(x)
        # print(x.shape)
        x = self.dropout(x)
        # print(x.shape)
        x = self.transformer(x)
        # print(x.shape)
        return x


if __name__ == '__main__':
    transformer = Action_Transformer(
        num_features=75,
        dim=75,
        depth=2,
        heads=8,
        mlp_dim=10,
        dropout=0.2,
        emb_dropout=0.1
    )
    N, C, T, V = 12, 3, 300, 25
    x = torch.randn(N, C, T, V)

    preds = transformer(x)  # torch.Size([12, 300, 96])

    print('Model total # params:', count_params(transformer))
