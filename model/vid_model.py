#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:11:14 2022

@author: 22905553
"""
# https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html
#https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
import torch
import torch.nn as nn
from torchvision import models
import thop
from thop import clever_format
import torch.nn.functional as F
#from torchsummary import summary    
from model.activations import Swish
from model.transformer import Transformer
import math
from model.TSE_module import TSELayer

from model.non_local import NLBlockND

def init_param_fc(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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

class ST_Vid_Att(nn.Module):
    def __init__(self, channel, max_frame, reduct_ratio=3, bias=True, **kwargs):
        super(ST_Vid_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.dropT = nn.Dropout(p=0.2)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.dropV = nn.Dropout(p=0.2)
        #self.max_pool_v = nn.AdaptiveMaxPool2d((1, 25))
        #self.max_pool_t = nn.AdaptiveMaxPool2d((max_frame, 1))

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)  # N, C, T, 1    
        x_v = x.mean(2, keepdims=True).transpose(2, 3)  # N, C, V, 1
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.dropT(self.conv_t(x_t).sigmoid())
        x_v_att = self.dropV(self.conv_v(x_v.transpose(2, 3)).sigmoid())
        x_att = x_t_att * x_v_att
        return x_att
            
class Attention_Layer(nn.Module):
    def __init__(self, out_channel, max_frame, att_type, act='swish', **kwargs):
        super(Attention_Layer, self).__init__()
        self.att_type = att_type
        
        __attention = {
            'stva': ST_Vid_Att
        }
        if att_type =='trf':
            max_len = 6115
            d_model = 192
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
            
            self.att = Transformer(d_model, heads = 4, N = 1)
            
            self.dropout = nn.Dropout(p=0.2)
        else:
            self.att = __attention[att_type](out_channel, max_frame, **kwargs)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = Swish()
        self.apply(weights_init)
        
    def forward(self, x):
        if self.att_type == 'trf':
            res = x
            x = self.dropout(x + self.pe[:, : x.size(1)].requires_grad_(False))
            x = self.att(x)
        else:
            res = x
            x = x * self.att(x)
            x = self.act(self.bn(x) + res)            
        return x

class sparse_temporal_distillation(nn.Module):
    def __init__(self, bias=True):
        super(sparse_temporal_distillation, self).__init__()
        self.avgpool =  nn.AdaptiveAvgPool2d(1)
        self.k = 100 # 30% frames
        self.bn = nn.BatchNorm1d(192)#.double()
        self.cnn = nn.Conv1d(192, 192, kernel_size=1, bias=bias)#.double()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        N, C, T, H, W = x.size()
        x = self.avgpool(x)#.squeeze(-1) # torch.Size([1, 192, 6115, 1, 1])
        x = x.view(N,C,T)#squeeze(-1)   # torch.Size([1, 192, 6115])
       # x = x.permute(0,2,1)
        U, s, VT = torch.linalg.svd(x)
        Sigma = torch.zeros((x.shape[0], x.shape[1], x.shape[2])).to(x.device)
        Sigma[:, :min(x.shape[1], x.shape[2]), :min(x.shape[1], x.shape[2])] = torch.stack([torch.diag(s[i,:]) for i in range(N)])
        x_svd = U[:, :, :self.k].float() @ Sigma[:, :self.k , :self.k ].float()
        x_svd = self.relu(self.cnn(self.bn(x_svd)))
        #print(x_svd.shape)
        return x_svd

class sparse_dyadic_distillation(nn.Module):
    def __init__(self, in_channels):
        super(sparse_dyadic_distillation, self).__init__()
        self.in_channels = in_channels
        #self.avgpool =  nn.AdaptiveAvgPool2d(1)
        
    def normalizeRAM(self, RAM):
        RAM = torch.where(RAM < 0.5, torch.zeros_like(RAM).to(RAM.device), RAM)
        da = torch.sum(RAM, dim=1)  # sum over column
        db = torch.sum(RAM, dim=2)  # sum over line
        norm = torch.einsum('nc,nt->nct', db, da)
        norm = torch.pow((norm+1e-5), -0.5)
        RAM = RAM*norm
        return RAM

    def forward(self, x):
        N,C,T = x.size()
        
        #x = self.avgpool(x).squeeze(dim=3).squeeze(dim=3)
        
        even_idx = torch.Tensor([i for i in range(T) if i%2==0]).to(x.device).int()
        x1 = torch.index_select(x, 2, even_idx)
        del even_idx
        if T % 2 == 1:
            x1 = x1[:,:,1:]
        odd_idx = torch.Tensor([i for i in range(T) if i%2==1]).to(x.device).int()        
        x2 = torch.index_select(x, 2, odd_idx)
        del odd_idx
        
        # geometric_component
        RAM_g_p1 = torch.exp(x1)
        RAM_g_p2 = torch.exp(-x2)
        RAM_g = torch.einsum(
                'nct, ncp->nctp', RAM_g_p1, RAM_g_p2)
        RAM_g = (torch.log(RAM_g+1e-5)).pow(2)
        RAM_g = (1/self.in_channels)*(
                torch.einsum('nctp->ntp', RAM_g))
        RAM_g = torch.exp(RAM_g)
        RAM = self.normalizeRAM(RAM_g)
        #print(RAM.shape)
        RAM=torch.repeat_interleave(RAM, 2, dim=1)
        #print(RAM.shape)
        RAM=torch.repeat_interleave(RAM, 2, dim=2)
        #print(RAM.shape)
        #RAM_i = torch.zeros(N,T,T).double().to(x.device)
        #RAM_i[:,1:T,1:T] = RAM
        #RAM_i[:,0,:1], x[:1,0] = RAM[:,0,:1], RAM[:,:1,0]
        x_hat = torch.einsum('nct, ntp->ncp',x, RAM)
        #print(x_hat.shape)
        return x_hat
    
class VideoModel(nn.Module):
            def __init__(self, channel_kernel=16, seq_kernel=64, hidden1=512, hidden2=128, ddm=False, sdd=False, stream='1s', attention=False, class_loss = False):
                super(VideoModel, self).__init__()
                hin = 192
                win = 6115
                self.attention=attention
                self.ddm = ddm
                self.sdd = sdd
                self.class_loss =  class_loss
                self.stream = stream
                self.attention = attention
                self.avgpool =  nn.AdaptiveAvgPool2d(1)
                self.maxpool = nn.MaxPool2d(kernel_size=(channel_kernel,seq_kernel), stride=(channel_kernel,seq_kernel-1))
                h0 = ((hin + (2*0) - 1 * (channel_kernel-1) - 1) // channel_kernel) + 1
                w0 = ((win + (2*0) - 1 * (seq_kernel-1-1) - 1) // (seq_kernel-1)) + 1
                self.linear = nn.Sequential(
                                #nn.Linear(h0*w0, hidden1),
                                #nn.Linear(1219, hidden1),
                                #nn.Linear(19200, hidden1),   # DDM
                                #nn.Linear(1092, hidden1),   # MTL
                                #nn.Linear(1800, hidden1),   # UNLV-Dive
                                nn.Linear(1128, hidden1),   # SDD
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden1, hidden2),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden2, 1)
                                )
                #self.TSELayer = TSELayer(win)     
                self.relu = nn.ReLU(inplace=True)
                if attention=='trf':
                    self.maxpool = nn.MaxPool2d(kernel_size=(25,97), stride=(5,96))               
                    self.attn = Attention_Layer(192,  6115, att_type=attention)
                elif attention=='stva':
                    self.attn = Attention_Layer(192,  6115, att_type=attention)
                    self.maxpool = nn.MaxPool2d(kernel_size=(channel_kernel,seq_kernel), stride=(channel_kernel,seq_kernel-1))
                elif attention=='DNL':
                    #self.NLnet = NLBlockND(in_channels=192, mode='embedded', dimension=1, bn_layer=True)
                    self.NLnet = NLBlockND(in_channels=192, mode='embedded', dimension=2, bn_layer=True)
                else:
                    if self.ddm:
                        self.ddm_module = sparse_temporal_distillation(bias=True)
                        self.cnnddm = nn.Conv2d(192, 192, kernel_size=20, bias=True)#.double()
                        self.maxpool = nn.MaxPool2d(kernel_size=(channel_kernel,seq_kernel), stride=(channel_kernel,seq_kernel+50))
                    else:
                        if self.sdd:
                            self.sdd_module = sparse_dyadic_distillation(192)
                        self.maxpool = nn.MaxPool2d(kernel_size=(channel_kernel,seq_kernel), stride=(channel_kernel,seq_kernel-1))
                
                if class_loss:
                    self.fc = nn.Sequential(
                                    #nn.Linear(h0*w0, hidden1),
                                    #nn.Linear(1219, hidden1),
                                    #nn.Linear(19200, hidden1),   # DDM
                                    #nn.Linear(1092, hidden1),   # MTL
                                    #nn.Linear(1800, hidden1),   # UNLV-Dive
                                    nn.Linear(1128, hidden1),   # SDD
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden1, hidden2),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden2, 2)
                                    )
                init_param_fc(self.modules())
                
            def forward(self, x):
                #x = x.permute(0,2,1,3,4)
                N, C, T, H, W = x.size()
                #print(x.shape)
                if self.attention=='trf':
                    x = x.permute(0,2,1,3,4).contiguous()  
                    x=x.mean(3).mean(3).view(N,T,C)                    
                    x = self.attn(x)
                    x = self.maxpool(x)   
                elif self.attention=='stva':
                    x = x.view(N,C,T,-1)
                    x = self.attn(x)
                    x = x.mean(dim=3)
                    x = self.maxpool(x)
                else:
                    if self.ddm:
                        #res = self.cnnddm(x.view(N,C,T,-1))
                        res = self.maxpool(x.view(N,C,T,-1))
                        #print(res.shape)
                        x = self.ddm_module(x)
                        
                    else:
                        #tse_out = self.TSELayer(x.view(N,C,T,-1)).view(N, C, T, H, W ) # torch.Size([1, 192, 6115, 64])
                        #print(tse_out.shape)
                        #x = self.relu(x+tse_out)
                        x = self.avgpool(x)#.squeeze(-1) # torch.Size([1, 192, 6115, 1, 1])
                        x = x.view(N,C,T)#squeeze(-1)   # torch.Size([1, 192, 6115])
                        #print(x.shape)
                        if self.attention == 'DNL':
                            x = self.NLnet(x)
                            #print(x.shape)
                        if self.sdd:
                            x_hat = self.sdd_module(x)                        
                        if self.sdd:
                            x = self.relu(x+x_hat)    
                        
                        x = self.maxpool(x)    # torch.Size([1, 12, 97])
                        
                        
                out = x.view(N,-1)
                #print(out.shape)
                if self.stream=='1s':
                    x = self.linear(out)
                    print(x.shape)
                    x = x[:,0]
                if self.class_loss:                                            
                    out = self.fc(out)
                
                return x, out

if __name__ == "__main__":
    # For debugging purposes
#     cd /home/uniwa/students3/students/22905553/linux/phd_codes/Olympics/Olympic_code/
    vid_model = VideoModel(channel_kernel=16, seq_kernel=32, hidden1=512, hidden2=128, ddm=False, 
                           sdd=False, stream='1s',  attention='DNL', class_loss=True)

    #vid_model.eval()
    T = 16#2938 #151#274 #6115
    x = torch.rand(1, 192, T, 8, 8)
    preds, out = vid_model(x)
    
    #dummy_input = [torch.rand(1,3, 8,256, 256),torch.rand(1,3, 32,256, 256)]
    #inputs = [i for i in dummy_input]
    #preds = model(inputs)
   
    '''
    
    macs, params = thop.profile(vid_model, inputs=(x,), verbose=False)
    macs, params = clever_format([macs, params], "%.2f")
    print( macs, params)
    #model = model.cuda()
    #summary(model,input_size=(3, 16,300, 300))  


    '''
    


    ## Prediction
'''
    post_act = torch.nn.Softmax(dim=1)

    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices[0]

    import urllib
    import json
    json_url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
    json_filename = "kinetics_classnames.json"
    try: urllib.URLopener().retrieve(json_url, json_filename)
    except: urllib.request.urlretrieve(json_url, json_filename)

    with open(json_filename, "r") as f:
        kinetics_classnames = json.load(f)
    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")


    # Map the predicted classes to the label names
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
    print("Top 5 predicted labels: %s" % ", ".join(pred_class_names))
    
    
 '''   


'''

model = model.cuda()
summary(model,input_size=(3, 16,300, 300))   


import torch
import torchvision
import torchextractor as tx

model1 = torchvision.models.resnet18(pretrained=True)
dummy_input = torch.rand(7, 3, 224, 224)

model1 = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
dummy_input = torch.rand(1,3, 8,256, 256)

model = tx.Extractor(model1.blocks, ["res_blocks"])


model_output, features = model(dummy_input)
feature_shapes = {name: f.shape for name, f in features.items()}
print(feature_shapes)

'''









