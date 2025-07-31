'''
@article{cheng2022spatial,
  title={Spatial-Angular Versatile Convolution for Light Field Reconstruction},
  author={Cheng, Zhen and Liu, Yutong and Xiong, Zhiwei},
  journal={IEEE Transactions on Computational Imaging},
  volume={8},
  pages={1131--1144},
  year={2022},
  publisher={IEEE}
}
'''
import torch
import torch.nn as nn
from option import args
from typing import Tuple, Callable
import importlib
import torch.nn.functional as F
import math
from utils.utils import LF_rgb2ycbcr


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        if args.angRes_in == 2 and args.angRes_out == 7:
            block_num = 16
            block_mode = "SAV_para"
            fn = 45
            self.net = LFASR_2_to_7_net(block_num, block_mode, fn)
        elif args.angRes_in == 2 and args.angRes_out == 8:
            block_num = 16
            block_mode = "SAV_para"
            fn = 45
            self.net = LFASR_2_to_8_net(block_num, block_mode, fn)

    def forward(self, lr, info=None):
        lr = LF_rgb2ycbcr(lr)[:, 0:1, :, :, :, :]
        [sr_mid, sr] = self.net(lr)

        out = {}
        out['SR_mid'] = sr_mid
        out['SR'] = sr
        return out


class LFASR_2_to_7_net(nn.Module):
    def __init__(self, block_num, block_mode="SAS", fn=64, init_gain=1.0):
        super(LFASR_2_to_7_net, self).__init__()
        self.new_ind = indexes()
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.feature_conv = convNd(in_channels=1,
                                   out_channels=fn,
                                   num_dims=4,
                                   kernel_size=(3, 3, 3, 3),
                                   stride=(1, 1, 1, 1),
                                   padding=(1, 1, 1, 1),
                                   kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in', 'leaky_relu'),
                                   bias_initializer=lambda x: nn.init.constant_(x, 0.0))
        sas_para = SAS_para()
        sac_para = SAC_para()
        sas_para.act = 'lrelu'
        sas_para.fn = fn
        sac_para.act = 'lrelu'
        sac_para.symmetry = True
        sac_para.max_k_size = 3
        sac_para.fn = fn

        if block_mode == "SAS":
            alt_blocks = [SAS_conv(act='lrelu', fn=fn) for _ in range(block_num)]
        elif block_mode == "SAC":
            alt_blocks = [SAC_conv(act='lrelu', fn=fn) for _ in range(block_num)]
        elif block_mode == "SAV_catres":
            alt_blocks = [SAV_concat(SAS_para=sas_para, SAC_para=sac_para, residual_connection=True) for _ in range(block_num)]
        elif block_mode == "SAV_cat":
            alt_blocks = [SAV_concat(SAS_para=sas_para, SAC_para=sac_para, residual_connection=False) for _ in
                          range(block_num)]
        elif block_mode == "SAV_para":
            alt_blocks = [SAV_parallel(SAS_para=sas_para, SAC_para=sac_para, feature_concat=False) for _ in
                          range(block_num)]
        else:
            raise Exception("Not implemented block mode!")

        self.alt_blocks = nn.Sequential(*alt_blocks)

        self.synthesis_layer = convNd(in_channels=fn,
                                      out_channels=45,
                                      num_dims=4,
                                      kernel_size=(2, 2, 3, 3),
                                      stride=(1, 1, 1, 1),
                                      padding=(0, 0, 1, 1),
                                      kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in',
                                                                                           'leaky_relu'),
                                      bias_initializer=lambda x: nn.init.constant_(x, 0.0))

        self.ang_dim_reduction1 = convNd(in_channels=1,
                                         out_channels=16,
                                         num_dims=4,
                                         kernel_size=(3, 3, 3, 3),
                                         stride=(2, 2, 1, 1),
                                         padding=(0, 0, 1, 1),
                                         kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in',
                                                                                              'leaky_relu'),
                                         bias_initializer=lambda x: nn.init.constant_(x, 0.0))
        self.ang_dim_reduction2 = convNd(in_channels=16,
                                         out_channels=64,
                                         num_dims=4,
                                         kernel_size=(2, 2, 3, 3),
                                         stride=(1, 1, 1, 1),
                                         padding=(0, 0, 1, 1),
                                         kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in',
                                                                                              'leaky_relu'),
                                         bias_initializer=lambda x: nn.init.constant_(x, 0.0))
        self.residual_predict = convNd(in_channels=64,
                                       out_channels=45,
                                       num_dims=4,
                                       kernel_size=(2, 2, 3, 3),
                                       stride=(1, 1, 1, 1),
                                       padding=(0, 0, 1, 1),
                                       kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in',
                                                                                            'leaky_relu'),
                                       bias_initializer=lambda x: nn.init.constant_(x, 0.0))

    def forward(self, lf_input):
        # lf_input: [B, 1, 2, 2, H, W]

        B, H, W = lf_input.shape[0], lf_input.shape[4], lf_input.shape[5]
        feat = self.relu(self.feature_conv(lf_input)) # [B, 64, 2, 2, H, W]
        feat_syn = self.alt_blocks(feat) # [B, 64, 2, 2, H, W]
        new_views = self.synthesis_layer(feat_syn) # [B, 45, 1, 1, H, W]

        # concat, re-organization and reshape
        lf_input = lf_input.view(B, 4, 1, 1, H, W) # [B, 25, 1, 1, H, W]
        lf_recon = torch.cat((new_views, lf_input), dim=1) # [B, 49, 1, 1, H, W]

        new_ind = torch.LongTensor(self.new_ind.indexes_for_2_to_7).to(lf_input.device)

        lf_recon = lf_recon.index_select(1, new_ind) # [B, 49, 1, 1, H, W]
        lf_recon = lf_recon.view(B, 1, 7, 7, H, W) # [B, 1, 7, 7, H, W]

        ## refinement
        feat_ang_reduce1 = self.relu(self.ang_dim_reduction1(lf_recon)) # [B, 16, 4, 4, H, W]
        feat_ang_reduce2 = self.relu(self.ang_dim_reduction2(feat_ang_reduce1)) # [B, 64, 2, 2, H, W]
        residual = self.residual_predict(feat_ang_reduce2) # [B, 45, 1, 1, H, W]

        ## final reconstruction
        new_views_refine = residual + new_views # [B, 45, 1, 1, H, W]
        lf_recon_refine = torch.cat((new_views_refine, lf_input), dim=1) # [B, 49, 1, 1, H, W]
        lf_recon_refine = lf_recon_refine.index_select(1, new_ind)
        lf_recon_refine = lf_recon_refine.view(B, 1, 7, 7, H, W) # [B, 1, 7, 7, H, W]
        return lf_recon, lf_recon_refine



class LFASR_2_to_8_net(nn.Module):
    def __init__(self, block_num, block_mode="SAS", fn=64, init_gain=1.0):
        super(LFASR_2_to_8_net, self).__init__()

        self.new_ind = indexes()

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.feature_conv = convNd(in_channels=1,
                                   out_channels=fn,
                                   num_dims=4,
                                   kernel_size=(3, 3, 3, 3),
                                   stride=(1, 1, 1, 1),
                                   padding=(1, 1, 1, 1),
                                   kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in', 'leaky_relu'),
                                   bias_initializer=lambda x: nn.init.constant_(x, 0.0))

        sas_para = SAS_para()
        sac_para = SAC_para()
        sas_para.act = 'lrelu'
        sas_para.fn = fn
        sac_para.act = 'lrelu'
        sac_para.symmetry = True
        sac_para.max_k_size = 3
        sac_para.fn = fn

        if block_mode == "SAS":
            alt_blocks = [SAS_conv(act='lrelu', fn=fn) for _ in range(block_num)]
        elif block_mode == "SAC":
            alt_blocks = [SAC_conv(act='lrelu', fn=fn) for _ in range(block_num)]
        elif block_mode == "SAV_catres":
            alt_blocks = [SAV_concat(SAS_para=sas_para, SAC_para=sac_para, residual_connection=True) for _ in range(block_num)]
        elif block_mode == "SAV_cat":
            alt_blocks = [SAV_concat(SAS_para=sas_para, SAC_para=sac_para, residual_connection=False) for _ in
                          range(block_num)]
        elif block_mode == "SAV_para":
            alt_blocks = [SAV_parallel(SAS_para=sas_para, SAC_para=sac_para, feature_concat=False) for _ in
                          range(block_num)]
        else:
            raise Exception("Not implemented block mode!")

        self.alt_blocks = nn.Sequential(*alt_blocks)

        self.synthesis_layer = convNd(in_channels=fn,
                                      out_channels=60,
                                      num_dims=4,
                                      kernel_size=(2, 2, 3, 3),
                                      stride=(1, 1, 1, 1),
                                      padding=(0, 0, 1, 1),
                                      kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in',
                                                                                           'leaky_relu'),
                                      bias_initializer=lambda x: nn.init.constant_(x, 0.0))

        self.ang_dim_reduction1 = convNd(in_channels=1,
                                         out_channels=16,
                                         num_dims=4,
                                         kernel_size=(2, 2, 3, 3),
                                         stride=(2, 2, 1, 1),
                                         padding=(0, 0, 1, 1),
                                         kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in',
                                                                                              'leaky_relu'),
                                         bias_initializer=lambda x: nn.init.constant_(x, 0.0))
        self.ang_dim_reduction2 = convNd(in_channels=16,
                                         out_channels=64,
                                         num_dims=4,
                                         kernel_size=(2, 2, 3, 3),
                                         stride=(2, 2, 1, 1),
                                         padding=(0, 0, 1, 1),
                                         kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in',
                                                                                              'leaky_relu'),
                                         bias_initializer=lambda x: nn.init.constant_(x, 0.0))
        self.residual_predict = convNd(in_channels=64,
                                       out_channels=60,
                                       num_dims=4,
                                       kernel_size=(2, 2, 3, 3),
                                       stride=(1, 1, 1, 1),
                                       padding=(0, 0, 1, 1),
                                       kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in',
                                                                                            'leaky_relu'),
                                       bias_initializer=lambda x: nn.init.constant_(x, 0.0))

    def forward(self, lf_input):
        # lf_input: [B, 1, 2, 2, H, W]

        B, H, W = lf_input.shape[0], lf_input.shape[4], lf_input.shape[5]
        feat = self.relu(self.feature_conv(lf_input)) # [B, 64, 2, 2, H, W]
        feat_syn = self.alt_blocks(feat) # [B, 64, 2, 2, H, W]
        new_views = self.synthesis_layer(feat_syn) # [B, 60, 1, 1, H, W]

        # concat, re-organization and reshape
        lf_input = lf_input.view(B, 4, 1, 1, H, W) # [B, 4, 1, 1, H, W]
        lf_recon = torch.cat((new_views, lf_input), dim=1) # [B, 64, 1, 1, H, W]

        new_ind = torch.LongTensor(self.new_ind.indexes_for_2_to_8).to(lf_input.device)

        lf_recon = lf_recon.index_select(1, new_ind) # [B, 64, 1, 1, H, W]
        lf_recon = lf_recon.view(B, 1, 8, 8, H, W) # [B, 1, 8, 8, H, W]

        ## refinement
        feat_ang_reduce1 = self.relu(self.ang_dim_reduction1(lf_recon)) # [B, 16, 4, 4, H, W]
        feat_ang_reduce2 = self.relu(self.ang_dim_reduction2(feat_ang_reduce1)) # [B, 64, 2, 2, H, W]
        residual = self.residual_predict(feat_ang_reduce2) # [B, 60, 1, 1, H, W]

        ## final reconstruction
        new_views_refine = residual + new_views # [B, 60, 1, 1, H, W]
        lf_recon_refine = torch.cat((new_views_refine, lf_input), dim=1) # [B, 64, 1, 1, H, W]
        lf_recon_refine = lf_recon_refine.index_select(1, new_ind)
        lf_recon_refine = lf_recon_refine.view(B, 1, 8, 8, H, W) # [B, 1, 8, 8, H, W]
        return lf_recon, lf_recon_refine


class SAS_para:
    def __init__(self):
        self.act = 'lrelu'
        self.fn = 64


class SAC_para:
    def __init__(self):
        self.act = 'lrelu'
        self.symmetry = True
        self.max_k_size = 3
        self.fn = 64


class SAS_conv(nn.Module):
    def __init__(self, act='relu', fn=64):
        super(SAS_conv, self).__init__()

        # self.an = an
        self.init_indicator = 'relu'
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
            self.init_indicator = 'relu'
            a = 0
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.init_indicator = 'leaky_relu'
            a = 0.2
        else:
            raise Exception("Wrong activation function!")

        self.spaconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1)
        self.angconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        N, c, U, V, h, w = x.shape  # [N,c,U,V,h,w]
        # N = N // (self.an * self.an)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(N*U*V, c, h, w)

        out = self.act(self.spaconv(x))  # [N*U*V,c,h,w]
        out = out.view(N, U*V, c, h * w)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N * h * w, c, U, V)  # [N*h*w,c,U,V]

        out = self.act(self.angconv(out))  # [N*h*w,c,U,V]
        out = out.view(N, h * w, c, U*V)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N, U, V, c, h, w)  # [N,U,V,c,h,w]
        out = out.permute(0, 3, 1, 2, 4, 5).contiguous() # [N,c,U,V,h,w]
        return out


class SAC_conv(nn.Module):
    def __init__(self, act='relu', symmetry=True, max_k_size=3, fn=64):
        super(SAC_conv, self).__init__()

        # self.an = an
        self.init_indicator = 'relu'
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
            self.init_indicator = 'relu'
            a = 0
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.init_indicator = 'leaky_relu'
            a = 0.2
        else:
            raise Exception("Wrong activation function!")

        if symmetry:
            k_size_ang = max_k_size
            k_size_spa = max_k_size
        else:
            k_size_ang = max_k_size - 2
            k_size_spa = max_k_size

        self.verconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=(k_size_ang, k_size_spa),
                                 stride=(1,1), padding=(k_size_ang // 2, k_size_spa // 2))
        self.horconv = nn.Conv2d(in_channels=fn, out_channels=fn, kernel_size=(k_size_ang, k_size_spa),
                                 stride=(1,1), padding=(k_size_ang // 2, k_size_spa // 2))

    def forward(self, x):
        N, c, U, V, h, w = x.shape  # [N,c,U,V,h,w]
        # N = N // (self.an * self.an)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N*V*w, c, U, h)

        out = self.act(self.verconv(x))  # [N*V*w,c,U,h]
        out = out.view(N, V * w, c, U * h)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N * U * h, c, V, w)  # [N*U*h,c,V,w]

        out = self.act(self.horconv(out))  # [N*U*h,c,V,w]
        out = out.view(N, U * h, c, V * w)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N, V, w, c, U, h)  # [N,V,w,c,U,h]
        out = out.permute(0, 3, 4, 1, 5, 2).contiguous() # [N,c,U,V,h,w]
        return out


class SAV_concat(nn.Module):
    def __init__(self, SAS_para, SAC_para, residual_connection=True):
        """
        parameters for building SAS-SAC block
        :param SAS_para: {act, fn}
        :param SAC_para: {act, symmetry, max_k_size, fn}
        :param residual_connection: True or False for residual connection
        """
        super(SAV_concat, self).__init__()
        self.res_connect = residual_connection
        self.SAS_conv = SAS_conv(act=SAS_para.act, fn=SAS_para.fn)
        self.SAC_conv = SAC_conv(act=SAC_para.act, symmetry=SAC_para.symmetry, max_k_size=SAC_para.max_k_size,
                                 fn=SAC_para.fn)

    def forward(self, lf_input):
        feat = self.SAS_conv(lf_input)
        res = self.SAC_conv(feat)
        if self.res_connect:
            res += lf_input
        return res


class SAV_parallel(nn.Module):
    def __init__(self, SAS_para, SAC_para, feature_concat=True):
        super(SAV_parallel, self).__init__()
        self.feature_concat = feature_concat
        self.SAS_conv = SAS_conv(act=SAS_para.act, fn=SAS_para.fn)
        self.SAC_conv = SAC_conv(act=SAC_para.act, symmetry=SAC_para.symmetry, max_k_size=SAC_para.max_k_size,
                                 fn=SAC_para.fn)
        if self.feature_concat:
            self.channel_reduce = convNd(in_channels=2 * SAS_para.fn,
                                         out_channels=SAS_para.fn,
                                         num_dims=4,
                                         kernel_size=(1, 1, 1, 1),
                                         stride=(1, 1, 1, 1),
                                         padding=(0, 0, 0, 0),
                                         kernel_initializer=lambda x: nn.init.kaiming_normal_(x, 0.2, 'fan_in',
                                                                                              'leaky_relu'),
                                         bias_initializer=lambda x: nn.init.constant_(x, 0.0))

    def forward(self, lf_input):
        sas_feat = self.SAS_conv(lf_input)
        sac_feat = self.SAC_conv(lf_input)  # [N,c,U,V,h,w]

        if self.feature_concat:
            concat_feat = torch.cat((sas_feat, sac_feat), dim=1)  # [N,2c,U,V,h,w]
            res = self.channel_reduce(concat_feat)
            res += lf_input
        else:
            res = sas_feat + sac_feat + lf_input
        return res


class convNd(nn.Module):
    """Some Information about convNd"""

    def __init__(self, in_channels: int,
                 out_channels: int,
                 num_dims: int,
                 kernel_size: Tuple,
                 stride,
                 padding,
                 is_transposed=False,
                 padding_mode='zeros',
                 output_padding=0,
                 dilation: int = 1,
                 groups: int = 1,
                 rank: int = 0,
                 use_bias: bool = True,
                 bias_initializer: Callable = None,
                 kernel_initializer: Callable = None):
        super(convNd, self).__init__()

        # ---------------------------------------------------------------------
        # Assertions for constructor arguments
        # ---------------------------------------------------------------------
        if not isinstance(kernel_size, Tuple):
            kernel_size = tuple(kernel_size for _ in range(num_dims))
        if not isinstance(stride, Tuple):
            stride = tuple(stride for _ in range(num_dims))
        if not isinstance(padding, Tuple):
            padding = tuple(padding for _ in range(num_dims))
        if not isinstance(output_padding, Tuple):
            output_padding = tuple(output_padding for _ in range(num_dims))
        if not isinstance(dilation, Tuple):
            dilation = tuple(dilation for _ in range(num_dims))

        # This parameter defines which Pytorch convolution to use as a base, for 3 Conv2D is used
        if rank == 0 and num_dims <= 3:
            max_dims = num_dims - 1
        else:
            max_dims = 3

        if is_transposed:
            self.conv_f = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)[max_dims - 1]
        else:
            self.conv_f = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[max_dims - 1]

        assert len(kernel_size) == num_dims, \
            'nD kernel size expected!'
        assert len(stride) == num_dims, \
            'nD stride size expected!'
        assert len(padding) == num_dims, \
            'nD padding size expected!'
        assert len(output_padding) == num_dims, \
            'nD output_padding size expected!'
        assert sum(dilation) == num_dims, \
            'Dilation rate other than 1 not yet implemented!'

        # ---------------------------------------------------------------------
        # Store constructor arguments
        # ---------------------------------------------------------------------
        self.rank = rank
        self.is_transposed = is_transposed
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_dims = num_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

        # ---------------------------------------------------------------------
        # Construct 3D convolutional layers
        # ---------------------------------------------------------------------
        if self.bias_initializer is not None:
            if self.use_bias:
                self.bias_initializer(self.bias)
        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv_layers = torch.nn.ModuleList()

        # Compute the next dimension, so for a conv4D, get index 3
        next_dim_len = self.kernel_size[0]

        for _ in range(next_dim_len):
            if self.num_dims - 1 > max_dims:
                # Initialize a Conv_n-1_D layer
                conv_layer = convNd(in_channels=self.in_channels,
                                    out_channels=self.out_channels,
                                    use_bias=self.use_bias,
                                    num_dims=self.num_dims - 1,
                                    rank=self.rank - 1,
                                    is_transposed=self.is_transposed,
                                    kernel_size=self.kernel_size[1:],
                                    stride=self.stride[1:],
                                    groups=self.groups,
                                    dilation=self.dilation[1:],
                                    padding=self.padding[1:],
                                    padding_mode=self.padding_mode,
                                    output_padding=self.output_padding[1:],
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer)

            else:
                # Initialize a Conv layer
                # bias should only be applied by the top most layer, so we disable bias in the internal convs
                conv_layer = self.conv_f(in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         bias=False,
                                         kernel_size=self.kernel_size[1:],
                                         dilation=self.dilation[1:],
                                         stride=self.stride[1:],
                                         padding=self.padding[1:],
                                         # padding_mode=self.padding_mode,
                                         groups=self.groups)
                if self.is_transposed:
                    conv_layer.output_padding = self.output_padding[1:]

                # Apply initializer functions to weight and bias tensor
                if self.kernel_initializer is not None:
                    self.kernel_initializer(conv_layer.weight)

            # Store the layer
            self.conv_layers.append(conv_layer)

    # -------------------------------------------------------------------------

    def forward(self, input):

        # Pad the input if is not transposed convolution
        if not self.is_transposed:
            padding = list(self.padding)
            # Pad input if this is the parent convolution ie rank=0
            if self.rank == 0:
                inputShape = list(input.shape)
                inputShape[2] += 2 * self.padding[0]
                padSize = (0, 0, self.padding[0], self.padding[0])
                padding[0] = 0
                if self.padding_mode == 'zeros':
                    input = F.pad(input.view(input.shape[0], input.shape[1], input.shape[2], -1), padSize, 'constant',
                                  0).view(inputShape)
                else:
                    input = F.pad(input.view(input.shape[0], input.shape[1], input.shape[2], -1), padSize,
                                  self.padding_mode).view(inputShape)

        # Define shortcut names for dimensions of input and kernel
        (b, c_i) = tuple(input.shape[0:2])
        size_i = tuple(input.shape[2:])
        size_k = self.kernel_size

        if not self.is_transposed:
            # Compute the size of the output tensor based on the zero padding
            size_o = tuple(
                [math.floor((size_i[x] + 2 * padding[x] - size_k[x]) / self.stride[x] + 1) for x in range(len(size_i))])
            # Compute size of the output without stride
            size_ons = tuple([size_i[x] - size_k[x] + 1 for x in range(len(size_i))])
        else:
            # Compute the size of the output tensor based on the zero padding
            size_o = tuple(
                [(size_i[x] - 1) * self.stride[x] - 2 * self.padding[x] + (size_k[x] - 1) + 1 + self.output_padding[x]
                 for x in range(len(size_i))])

        # Output tensors for each 3D frame
        frame_results = size_o[0] * [torch.zeros((b, self.out_channels) + size_o[1:], device=input.device)]
        empty_frames = size_o[0] * [None]

        # Convolve each kernel frame i with each input frame j
        for i in range(size_k[0]):
            # iterate inputs first dimmension
            for j in range(size_i[0]):

                # Add results to this output frame
                if self.is_transposed:
                    out_frame = i + j * self.stride[0] - self.padding[0]
                else:
                    out_frame = j - (i - size_k[0] // 2) - (size_i[0] - size_ons[0]) // 2 - (1 - size_k[0] % 2)
                    k_center_position = out_frame % self.stride[0]
                    out_frame = math.floor(out_frame / self.stride[0])
                    if k_center_position != 0:
                        continue

                if out_frame < 0 or out_frame >= size_o[0]:
                    continue

                # Prepate input for next dimmension
                conv_input = input.view(b, c_i, size_i[0], -1)
                conv_input = conv_input[:, :, j, :].view((b, c_i) + size_i[1:])

                # Convolve
                frame_conv = \
                    self.conv_layers[i](conv_input)

                if empty_frames[out_frame] is None:
                    frame_results[out_frame] = frame_conv
                    empty_frames[out_frame] = 1
                else:
                    frame_results[out_frame] += frame_conv

        result = torch.stack(frame_results, dim=2)

        if self.use_bias:
            resultShape = result.shape
            result = result.view(b, resultShape[1], -1)
            for k in range(self.out_channels):
                result[:, k, :] += self.bias[k]
            return result.view(resultShape)
        else:
            return result


class indexes:
    indexes_for_2_to_8 = [60, 0, 1, 2, 3, 4, 5, 61,
                          6, 7, 8, 9, 10, 11, 12, 13,
                          14, 15, 16, 17, 18, 19, 20, 21,
                          22, 23, 24, 25, 26, 27, 28, 29,
                          30, 31, 32, 33, 34, 35, 36, 37,
                          38, 39, 40, 41, 42, 43, 44, 45,
                          46, 47, 48, 49, 50, 51, 52, 53,
                          62, 54, 55, 56, 57, 58, 59, 63]
    indexes_for_2_to_7 = [45, 0, 1, 2, 3, 4, 46,
                          5, 6, 7, 8, 9, 10, 11,
                          12, 13, 14, 15, 16, 17, 18,
                          19, 20, 21, 22, 23, 24, 25,
                          26, 27, 28, 29, 30, 31, 32,
                          33, 34, 35, 36, 37, 38, 39,
                          47, 40, 41, 42, 43, 44, 48]
    indexes_for_2_to_8_extra1 = [0, 1, 2, 3, 4, 5, 6, 7,
                                 8, 60, 9, 10, 11, 12, 61, 13,
                                 14, 15, 16, 17, 18, 19, 20, 21,
                                 22, 23, 24, 25, 26, 27, 28, 29,
                                 30, 31, 32, 33, 34, 35, 36, 37,
                                 38, 39, 40, 41, 42, 43, 44, 45,
                                 46, 62, 47, 48, 49, 50, 63, 51,
                                 52, 53, 54, 55, 56, 57, 58, 59]
    indexes_for_2_to_8_extra2 = [0, 1, 2, 3, 4, 5, 6, 7,
                                 8, 9, 10, 11, 12, 13, 14, 15,
                                 16, 17, 60, 18, 19, 61, 20, 21,
                                 22, 23, 24, 25, 26, 27, 28, 29,
                                 30, 31, 32, 33, 34, 35, 36, 37,
                                 38, 39, 62, 40, 41, 63, 42, 43,
                                 44, 45, 46, 47, 48, 49, 50, 51,
                                 52, 53, 54, 55, 56, 57, 58, 59]


def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, criterion_data=None):
        HR = LF_rgb2ycbcr(HR)[:, 0:1, :, :, :, :]
        loss = self.criterion_Loss(out['SR'], HR) + self.criterion_Loss(out['SR_mid'], HR)

        return loss

if __name__ == "__main__":
    args.model_name = 'LFASR_SAV'
    args.angRes_in = 2
    args.angRes_out = 7

    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    MODEL_PATH = args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args).to(device)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of Parameters: %.4fM' % (total / 1e6))

    input = torch.randn([2, 3, args.angRes_in, args.angRes_in, 64, 64]).to(device)
    GPU_cost = torch.cuda.memory_allocated(0)
    out = net(input, [])