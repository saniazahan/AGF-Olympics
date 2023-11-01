#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 19:03:05 2022

@author: 22905553
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from model.non_local_delta import NLBlockND
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

def init_param(modules):
    for m in modules:
        if isinstance(m, nn.GRU):
            #print(m)
            for child in list(m.children()):
                print(child)
                for param in list(child.parameters()):
                    if param.dim() == 2:
                        nn.init.xavier_uniform_(param)
            #nn.init.xavier_uniform_(m)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
# network module only set encoder to be bidirection
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, drop_out=0.2):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.num_layers = num_layers
        self.gap =  nn.AdaptiveAvgPool2d((224,1))
        if drop_out:
           self.drop_out = nn.Dropout(drop_out)
        else:
           self.drop_out = lambda x: x
        self.fc = nn.Linear(224, 1)
        init_param(self.modules())
    def forward(self, input_tensor, seq_len):
        
        encoder_hidden = torch.Tensor().to(device)
        
        for it in range(max(seq_len)):
          #print(it)
          if it == 0:
            enout_tmp, hidden_tmp = self.gru(input_tensor[:, it:it+1, :])
          else:
            enout_tmp, hidden_tmp = self.gru(input_tensor[:, it:it+1, :], hidden_tmp)
          encoder_hidden = torch.cat((encoder_hidden, enout_tmp),1)
         
        #print(encoder_hidden.shape)
        '''
        hidden = torch.empty((1, len(seq_len), encoder_hidden.shape[-1])).to(device)
        print(hidden.shape)
        count = 0
        for ith_len in seq_len:
            hidden[0, count, :] = encoder_hidden[count, ith_len - 1, :]
            count += 1
        print(hidden.shape)
        '''
        pooled_feature = self.gap(encoder_hidden)
        #print(pooled_feature.shape)
        encoder_hidden = self.drop_out(pooled_feature).squeeze(2)
        #print(encoder_hidden.shape)
        out = self.fc(encoder_hidden)
        return out[:,0]


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

class Freq_attention(nn.Module):
    def __init__(self, inchannel):
        super(Freq_attention, self).__init__()
        self.in_channels = inchannel
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
       #if T % 2 == 1:
       #    x1 = x1[:,:,1:]
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
       RAM = RAM[:,:-1,:]
       #print(RAM.shape)
       #RAM_i = torch.zeros(N,T,T).double().to(x.device)
       #RAM_i[:,1:T,1:T] = RAM
       #RAM_i[:,0,:1], x[:1,0] = RAM[:,0,:1], RAM[:,:1,0]
       x_hat = torch.einsum('nct, ntp->nct',x, RAM)
       return x_hat
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
    
class MLP(nn.Module):
    def __init__(self, input_size, inchannel, NL_mode='concatenate', NL_dimention=1, channel_kernel=21, seq_kernel=59, drop_out=0.2, hidden1 = 512, hidden2 = 128, stream='1s', class_loss=False):
        super(MLP, self).__init__()
        #self.maxpool =  nn.MaxPool2d((1,input_size))
        '''
        hin = 192
        win = 5600
        self.averagepool =  nn.AdaptiveAvgPool2d((inchannel,1))
        self.maxpool = nn.MaxPool2d(kernel_size=(channel_kernel,seq_kernel), stride=(channel_kernel,seq_kernel-1))
        h0 = ((hin + (2*0) - 1 * (channel_kernel-1) - 1) // channel_kernel) + 1
        w0 = (((win + (2*0) - 1 * (seq_kernel-1-1) - 1) // (seq_kernel-1)) + 1)+1
        if stream == '2s':
            h0 = h0*2
        self.features = nn.Sequential(            
            nn.Linear(h0*w0, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, 1)           
        )
        
        '''
        ## Start of Vanila block ==============================================
        #self.maxpool =  nn.MaxPool2d((1,5600))
        #self.maxpool =  nn.MaxPool2d((1, 875))
        #self.averagepool =  nn.AdaptiveAvgPool2d((inchannel,1))
        #self.NLnet = NLBlockND(in_channels=192, mode=NL_mode, dimension=NL_dimention, bn_layer=True)
        ## End of Vanila block ================================================
       
        ## Start of NL-DDM Mu block ==============================================
        #self.maxpool =  nn.MaxPool2d((1,5600))
        #self.Feat_maxpool =  nn.MaxPool2d((1,25))
        #self.maxpool =  nn.MaxPool2d((1, 875))
        #self.averagepool =  nn.AdaptiveAvgPool2d((inchannel,1))
        #self.NLnet = NLBlockND(in_channels=192, mode='embedded', dimension=1, bn_layer=True)
        ## End of DDM-NL Mu block ================================================        
        
        ## Start of NL-DDM Delta block ==============================================
        #self.maxpool =  nn.MaxPool2d((1,5600))
        self.Feat_maxpool =  nn.MaxPool2d(2)
        self.maxpool =  nn.MaxPool2d((1, 5244))
        self.averagepool =  nn.AdaptiveAvgPool2d((inchannel,1))
        self.NLnet = NLBlockND(in_channels=192, mode='embedded', dimension=2, bn_layer=True)
        ## End of DDM-NL Delta block ================================================      
        
        ## Start of DDM block =================================================
        #self.Freq_attention = Freq_attention(inchannel)
        ## End of DDM block ===================================================
        
        ## Start of Attention block ===========================================
        #self.inavgpool =  nn.AdaptiveAvgPool2d((875,1))
        #self.MultiHeadAttention = MultiHeadAttention(heads=7, d_model=875)
        ## End of Attention block =============================================
        
        #self.conv_att = nn.Conv1d(inchannel, inchannel, kernel_size=1, bias=True)
        self.act =  nn.ReLU()
        self.features = nn.Sequential(            
            nn.Linear(inchannel*2, 1),
            #nn.ReLU(inplace=True),
            #nn.Linear(hidden1, 1),            
            #nn.Sigmoid()    # label smoothing changes score range to 0 - 1 so sigmoid activation helps keep prediction in the same range            
        )
        
        # placeholder for the gradients
        self.gradients = None
        
        self.class_loss = class_loss
        if self.class_loss:
            self.fc_class = nn.Linear(inchannel, 2)
        init_param_fc(self.modules())
    
    def activations_hook(self, grad):
        ''' for activation map '''
        self.gradients = grad
        
    def get_activations(self, x):
        ''' for activation '''
        N, C, T, V = x.size()
        #temp = self.Feat_maxpool(x).squeeze(dim=3)  # torch.Size([1, 192, 875, 1])
        #print(temp.shape)
        #x3 = self.NLnet(temp).view(N,C,-1)
        return x
    
    def get_activations_gradient(self):
        ''' for activation map '''
        return self.gradients

    
    def forward(self, input_tensor, vid_feat=None):
        N, C, T, V = input_tensor.size()
        if torch.isnan(input_tensor).any():
            print('Nan found in MLP')
        
        if self.class_loss:
            #out =  self.avgpool(x)
            out_channels = input_tensor.size(1)
            out = input_tensor.view(N, out_channels, -1)
            #print(out.shape)
            out = out.mean(2)   # Global Average Pooling (Spatial+Temporal)
            #print(out.shape)
            #out = out.mean(1)   # Average pool number of bodies in the sequence
            clss = self.fc_class(out)
        else:
            clss = 0
        '''
        ## Start of Vanila block ==============================================
        N, C, T, V = input_tensor.size()
        temp = input_tensor.mean(3)#.squeeze(dim=3)  
        x3 = self.NLnet(temp)
        #print(x3.shape)
   
        temp = input_tensor.mean(3)#.squeeze(dim=3)  
        
        input_tensor = temp
        input_tensor.register_hook(self.activations_hook)
        #input_tensor = self.act(temp+x3).view(N,C,-1)
        #print(input_tensor.shape)
        x1 = self.averagepool(input_tensor).squeeze(dim=2) # torch.Size([1, 192])
        #print(x1.shape)
        x2 = self.maxpool(input_tensor).squeeze(dim=2)
        #print(x2.shape)
        ## End of Vanila block ================================================
        '''
        ## Start of NL-DDM block ==============================================
        N, C, T, V = input_tensor.size()
        temp = self.Feat_maxpool(input_tensor)#.squeeze(dim=3)  # torch.Size([1, 192, 875, 1])
        #print(temp.shape)
        x3 = self.NLnet(temp).view(N,C,-1)
        #print(x3.shape)
        
        x3.register_hook(self.activations_hook)
        
        x1 = self.averagepool(x3).squeeze(dim=2) # torch.Size([1, 192])
        #print(x1.shape)
        x2 = self.maxpool(x3).squeeze(dim=2)    
        #x2 = x3.mean(2)#.squeeze(dim=2)    
        #print(x2.shape)
        #input_tensor = temp
        #input_tensor = self.act(temp+x3)

        #x1 = self.averagepool(input_tensor).squeeze(dim=2) # torch.Size([1, 192])
        #print(x1.shape)
        #x2 = self.maxpool(input_tensor).squeeze(dim=2)    
        ## End of DDM-NL block ================================================
       
        
        ## Start of DDM block =================================================
        #input_ddm = self.Freq_attention(input_tensor)
        #x_att = input_tensor + input_ddm
        #print(input_ddm.shape)
        ## End of DDM block ===================================================
        
        ## Start of Attention block ===========================================
        #input_tensor = self.inavgpool(input_tensor).view(N,C,T)
        #x_att = self.MultiHeadAttention(input_tensor, input_tensor, input_tensor)
        #x_att = self.act(input_tensor + x_att)
        #print(x_att.shape)
        #x_att = self.act(self.conv_att(x_att))        
        # x1 = x_att.mean(2)
        #x1 = x_att.mean(2)
        #x2 = self.averagepool(x_att).squeeze(dim=2)
        ## End of Attention block =============================================
        
        
        x = torch.cat((x1,x2), dim = 1)
        #x = self.act(x+x3)
        #print(x.shape)
        if torch.is_tensor(vid_feat):
            x = torch.cat((x,vid_feat), dim=1)            
            #print(x.shape)
            
        #x = x.view(N,-1)
        #print(x.shape)
        #print(x.shape)
        out = self.features(x).squeeze(dim=1)
        #print(out)
        return out, clss
        
        
if __name__ == "__main__":
    # For debugging purposes
#     cd /home/uniwa/students3/students/22905553/linux/phd_codes/Olympics/Olympic_code/
    import sys
    sys.path.append('..')
    import thop
    from thop import clever_format
    NL_mode = 'concatenate'
    NL_dimention = 1
    seq_model = MLP(input_size=875, inchannel=192, NL_mode=NL_mode, NL_dimention=NL_dimention, channel_kernel=16, seq_kernel=59, drop_out=0.2, hidden1 = 192, hidden2 = 96, stream='1s', class_loss=True)
    x = torch.randn(4, 192, 875, 25)
    vid_feat = None#torch.randn(1, 12, 97)
    out, clss = seq_model.forward(x,vid_feat)
    
    
    macs, params = thop.profile(seq_model, inputs=(x,[0],), verbose=False)
    macs, params = clever_format([macs, params], "%.2f")
    print( macs, params)