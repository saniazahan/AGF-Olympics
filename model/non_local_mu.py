import torch
from torch import nn
from torch.nn import functional as F
import einops

class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't 
           include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError(
                '`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels, kernel_size=1)

        self.act = nn.ReLU()
        self.sep_head = nn.Conv1d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
            
            self.W_z_DD = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z_DD[1].weight, 0)
            nn.init.constant_(self.W_z_DD[1].bias, 0)
            
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels,
                               out_channels=self.in_channels, kernel_size=1)
            
            self.W_z_DD = conv_nd(in_channels=self.inter_channels,
                               out_channels=self.in_channels, kernel_size=1)
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z_DD.weight, 0)
            nn.init.constant_(self.W_z_DD.bias, 0)
            
            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels,
                                 out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels,
                               out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels *
                          2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal 
        # function in the original Caffe2 implementation
        # torch.Size([1, 96, 875])
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # torch.Size([1, 875, 96])
        # print(g_x.shape)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            #temp1 = self.sep_head(x.mean(3))
            #temp2 = self.sep_head(x.mean(2))
            #temp_f = torch.einsum('nct,ncv->nctv', (temp1, temp2)
            #                      ).contiguous().view(batch_size, self.inter_channels, -1)
            # print(temp_f.shape)
            # torch.Size([1, 96, 5244])
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            # print(theta_x.shape)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)

            f = torch.matmul(theta_x, phi_x)
            # print(f.shape)
            #f = self.act(f+temp_f)

        elif self.mode == "concatenate":
            # torch.Size([1, 96, 875, 1])
            theta_x = self.theta(x).view(
                batch_size, self.inter_channels, -1, 1)

            # torch.Size([1, 96, 1, 875])
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            # torch.Size([1, 192, 875, 875])
            concat = torch.cat([theta_x, phi_x], dim=1)

            f = self.W_f(concat)    # torch.Size([1, 1, 875, 875])

            # torch.Size([1, 875, 875])
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
            #temp_f_div_C = F.softmax(temp_f, dim=-1)

        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N
            #temp_f_div_C = temp_f / temp_f.size(-1)
        
        ## Start DD - discriminative distillation
        DD_y = torch.tensor(0.).to(x.device)
        dif = g_x[:, 1:, :] - g_x[:, 0:-1, :] #  
        dif = torch.cat([dif.new(batch_size, 1, dif.size(2)).zero_(), dif], dim=-2)
        dif = torch.sum(dif, dim=2)
       
        split_val = torch.quantile(dif, 0.5, dim=1).to(x.device)
        #print(split_val)
        
        d = torch.stack([(dif[i] >= split_val[i]).int() for i in range(batch_size)]).to(x.device)#.nonzero()[:,1]
        d = einops.repeat(d, 'm n -> m n k', k=96)
        
        y_DD = g_x*d
        
        y_DD = y_DD.permute(0, 2, 1).contiguous()
        y_DD = y_DD.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y_DD = self.W_z_DD(y_DD)  # torch.Size([1, 192, 875])
        #print(W_y_DD.shape)
        ## End DD - discriminative distillation
        
        #print(y_DD.shape)
        y = torch.matmul(f_div_C, g_x)  # torch.Size([1, 875, 96])
        
        #y = y+y_DD
        #print(y.shape)
        # torch.Size([1, 875, 96])
        #temp_y = torch.einsum('nct,ntc->ntc', (temp_f_div_C, g_x)).contiguous()

        #y = temp_y + y
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])

        W_y = self.W_z(y)  # torch.Size([1, 192, 875])
        
        W_y = self.act(W_y + W_y_DD)
        # residual connection
        z = W_y + x

        return z


if __name__ == '__main__':
    import torch
    import thop
    from thop import clever_format

    for bn_layer in [True]:  # [True, False]:
        '''
        img = torch.zeros(2, 3, 20)
        net = NLBlockND(in_channels=3, mode='concatenate', dimension=1, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.zeros(2, 3, 20, 20)
        net = NLBlockND(in_channels=3, mode='concatenate', dimension=2, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.randn(2, 3, 8, 20, 20)
        net = NLBlockND(in_channels=3, mode='concatenate', dimension=3, bn_layer=bn_layer)
        out = net(img)
        print(out.size())
        '''

        x = torch.randn(2, 192, 875)
        net = NLBlockND(in_channels=192, mode='embedded',
                        dimension=1, bn_layer=bn_layer)
        out = net(x)
        # print(out.size())

        macs, params = thop.profile(net, inputs=(x,), verbose=False)
        macs, params = clever_format([macs, params], "%.2f")
        print( macs, params)