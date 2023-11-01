# cd /home/uniwa/students3/students/22905553/linux/phd_codes/Olympics/Olympic_code/
import sys
sys.path.insert(0, '')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import *
from model.adjGraph import adjGraph
from model.joint_pose_module import Xspace_time_Pose_Module as XPM

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x
    
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
    def __init__(self, dim, dim1, att_type, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(                        
                norm_data(dim),
                cnn1x1(dim, dim1, bias=bias),
                nn.ReLU()
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, dim1, bias=bias),
                nn.ReLU()
            )
        #self.attention =  Attention_Layer(dim1,  att_type=att_type)

    def forward(self, x):
        x = self.cnn(x)
        #print(x.shape)
        return x#self.attention(x)


def init_param(modules):
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
                
class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 max_frame,
                 graph_args,
                 act_type, 
                 bias,
                 edge,
                 block_size,
                 class_loss = False,
                 xpm_window_size = 25,
                 xpm_stride = 15,
                 xpm_num_scales = 6,
                 xpm = False):
        super(Model, self).__init__()
        
        self.class_loss =  class_loss
        temporal_window_size = 3
        max_graph_distance = 2
        keep_prob = 0.9
        self.xpm = xpm
        #A = torch.rand(3,25,25).cuda()#.to(num_class.dtype).to(num_class.device)
        self.graph_hop = adjGraph(**graph_args)
        A = torch.tensor(self.graph_hop.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        # channels
        D_embed = 48
        c1 = D_embed*2
        c2 = c1 * 2     
        #c3 = c2 * 2    
       
        
        
        self.joint_embed = embed(2, D_embed, att_type='stja', norm=True, bias=bias)
        self.dif_embed = embed(2, D_embed, att_type='stja', norm=True, bias=bias) #601
        #self.attention =  Attention_Layer(D_embed,  max_frame, act_type, att_type='stja')
        if self.xpm:
            self.XPM1 = XPM(D_embed, c1, xpm_num_scales, xpm_window_size, xpm_stride, frame=250)
        
        self.sgcn1 = SpatialGraphConv(D_embed, c1, max_graph_distance, bias, edge, A, act_type, residual=True)
        self.tcn11 = SepTemporal_Block(c1, temporal_window_size, bias, act_type, keep_prob, block_size, expand_ratio=0, stride=1, residual=True)
        self.tcn12 = SepTemporal_Block(c1, temporal_window_size+2, bias, act_type, keep_prob, block_size, expand_ratio=0, stride=2, residual=True)
        
        self.sgcn2 = SpatialGraphConv(c1, c2, max_graph_distance, bias, edge, A, act_type, residual=True)
        self.tcn21 = SepTemporal_Block(c2, temporal_window_size, bias, act_type, keep_prob, block_size, expand_ratio=0, stride=2, residual=True)
        self.tcn22 = SepTemporal_Block(c2, temporal_window_size+2, bias, act_type, keep_prob, block_size, expand_ratio=0, stride=4, residual=True)
        
        if self.xpm:
            self.XPM2 = XPM(c1, c2, xpm_num_scales, xpm_window_size, xpm_stride, frame=32)
            
        if class_loss:
            self.fc = nn.Linear(c2, num_class)
        init_param(self.modules())
    
    
    
    def forward(self, x):        
        if torch.isnan(x).any():
            print('Nan found in GCN')
            
        N, C, T, V = x.size()
       
        # Dynamic Representation        
        pos = x.permute(0, 1, 3, 2).contiguous()  # N, C, V, T
        #print(pos.shape)
        dif = pos[:, :, :, 1:] - pos[:, :, :, 0:-1] #  
        dif = torch.cat([dif.new(N, dif.size(1), V, 1).zero_(), dif], dim=-1)
        #dif2 = pos[:, :, :, 2:] - pos[:, :, :, 0:-2] #  
        #dif2 = torch.cat([dif2.new(N, dif.size(1), V, 2).zero_(), dif2], dim=-1)
        #dif3 = torch.cat([dif, dif2], dim=1)
        #print(dif3.shape)
        
        pos = self.joint_embed(pos)        
        dif = self.dif_embed(dif)
        dy = pos + dif
        dy = dy.permute(0,1,3,2).contiguous() # N, C, T, V   # torch.Size([1, 48, 500, 25])
        #print(dy.shape)
        #dy = self.attention(dy)
        #dy.register_hook(lambda g: print(g))
        #########################
        
        #print(xpm_out.shape)
        out = self.tcn12(self.tcn11(self.sgcn1(dy)))    # torch.Size([1, 96, 250, 25])
        if self.xpm:
            xpm_out = self.XPM1(dy)
            #print(xpm_out.shape)
            out = F.relu(out+xpm_out, inplace=True)
        #print(out.shape)
        if self.xpm:
            xpm_out = self.XPM2(out)
        out = self.tcn22(self.tcn21(self.sgcn2(out)))   # torch.Size([1, 192, 32, 25])
        if self.xpm:
            #print(xpm_out.shape)
            out = F.relu(out+xpm_out, inplace=True)
        
        print(out.shape)
        out_channel = out.size(1)
        features = out.reshape(N, out_channel, -1)   
        
        if self.class_loss:
            out = features.mean(2)        
            out = self.fc(out)
        
        
        return features, out
        

if __name__ == "__main__":
    # For debugging purposes
#     cd /home/uniwa/students3/students/22905553/linux/phd_codes/Olympics/Olympic_code/
    import sys
    sys.path.append('..')
    import thop
    from thop import clever_format
     
    model = Model(
        num_class=2,
        num_point=25,
        max_frame=3500,
        graph_args = {'layout': 'olympic','strategy': 'spatial','max_hop':3},
        act_type = 'relu',
        bias = True,
        edge = True,
        block_size=41,
        class_loss = False,
        xpm_window_size = 25,
        xpm_stride = 2,
        xpm_num_scales = 6, 
        xpm = False
    )
    #model = model#.cuda()
    
    #N, C, T, V, M = 6, 3, 300, 25, 2
    x = torch.randn(1, 2, 500, 25)#.cuda()
   
    features, out = model.forward(x)
    
    
    macs, params = thop.profile(model, inputs=(torch.randn(1,2,500,25),), verbose=False)
    macs, params = clever_format([macs, params], "%.2f")

    print( macs, params)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    