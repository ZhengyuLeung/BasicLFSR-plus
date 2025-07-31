'''
@article{LF-EASR,
  title={Efficient light field angular super-resolution with sub-aperture feature learning and macro-pixel upsampling},
  author={Liu, Gaosheng and Yue, Huanjing and Wu, Jiamin and Yang, Jingyu},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}
'''
import torch
import torch.nn as nn
import torch.nn.functional as functional
from einops import rearrange


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        channel = 64
        self.FeaExtract = InitFeaExtract(channel)
        self.D3Unet = UNet(channel, channel, channel)
        self.Out = nn.Conv2d(channel, 3, 1, 1, 0, bias=False)
        self.Angular_UpSample = Upsample(channel, self.angRes_in, self.angRes_out)
        self.Resup = Interpolation(self.angRes_in, self.angRes_out)

    def forward(self, lf, info):
        Bicubic_up = self.Resup(lf)

        buffer_mv_initial = self.FeaExtract(lf)
        buffer_mv = self.D3Unet(rearrange(buffer_mv_initial, 'b c u v h w -> b c (u v) h w'))
        HAR = self.Angular_UpSample(buffer_mv)
        y = self.Out(rearrange(HAR, 'b c u v h w -> (b u v) c h w'))
        y = rearrange(y, '(b u v) c h w -> b c u v h w', u=self.angRes_out, v=self.angRes_out)
        y = y + Bicubic_up

        out = {}
        out['SR'] = y
        return out


class Conv2d_refpad(nn.Module):
    def __init__(self, inchannel, outchannel, kernel=3):
        super(Conv2d_refpad, self).__init__()
        pad = kernel // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=kernel, padding=0, bias=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channel, angRes_in, angRes_out):
        super(Upsample, self).__init__()
        self.an = angRes_in
        self.an_out = angRes_out
        self.angconv = nn.Sequential(
            nn.Conv2d(in_channels=channel * 2, out_channels=channel * 2, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True))
        self.upsp = nn.Sequential(
            nn.Conv2d(channel * 2, channel * 2, kernel_size=angRes_in, stride=angRes_in, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel * 2, channel * angRes_out * angRes_out, kernel_size=1, padding=0, bias=False),
            nn.PixelShuffle(angRes_out),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = rearrange(x, 'b (u v) c h w -> (b h w) c u v', u=self.an, v=self.an)
        up_in = self.angconv(x)

        out = self.upsp(up_in)
        out = rearrange(out, '(b h w) c u v -> b c u v h w', h=h, w=w)
        return out


class Interpolation(nn.Module):
    def __init__(self, angRes_in, angRes_out):
        super(Interpolation, self).__init__()
        self.an = angRes_in
        self.an_out = angRes_out

    def forward(self, x):
        [b, c, u, v, h, w] = x.size()
        x = rearrange(x, 'b c u v h w -> (b h w) c u v')
        out = functional.interpolate(x, size=[self.an_out, self.an_out], mode='bicubic', align_corners=False)
        out = rearrange(out, '(b h w) c u v -> b c u v h w', h=h, w=w)
        return out


class D3Resblock(nn.Module):
    def __init__(self, channel):
        super(D3Resblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1),
                      bias=False),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv_2 = nn.Conv3d(channel, channel, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1),
                                dilation=(1, 1, 1), bias=False)
        # self.gating = SEGating(channel)

    def __call__(self, x_init):
        x = self.conv(x_init)
        x = self.conv_2(x)
        return x + x_init


class SEGating(nn.Module):

    def __init__(self, inplanes, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attn_layer = nn.Sequential(
            nn.Conv3d(inplanes, inplanes, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        y = self.attn_layer(out)
        return x * y


class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.1, inplace=True)

        # Down sampling
        self.down_1 = D3Resblock(self.in_dim)
        self.pool_1 = stride_conv_3d(self.num_filters, self.num_filters * 2, activation)
        self.down_2 = D3Resblock(self.num_filters * 2)
        self.pool_2 = stride_conv_3d(self.num_filters * 2, self.num_filters * 3, activation)

        # Bridge
        self.bridge_1 = D3Resblock(self.num_filters * 3)
        # self.bridge_2 = D3Resblock(self.num_filters * 3)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 3, self.num_filters * 2, activation)
        self.up_1 = D3Resblock(self.num_filters * 2)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 1, activation)
        self.up_2 = D3Resblock(self.num_filters * 1)

        # Output
        self.out_2D = nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        # Bridge
        bridge = self.bridge_1(pool_2)

        # Up sampling
        trans_1 = self.trans_1(bridge)
        addition_1 = trans_1 + down_2
        up_1 = self.up_1(addition_1)
        trans_2 = self.trans_2(up_1)
        addition_2 = trans_2 + down_1
        up_2 = self.up_2(addition_2)

        # Output
        out = torch.cat((up_2, x), 1).permute(0, 2, 1, 3, 4)
        b, n, c, h, w = out.shape
        out = self.out_2D(out.contiguous().view(b * n, c, h, w)).view(b, n, c, h, w)  # -> [1, 3, 128, 128, 128]
        return out


def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
        activation)


def stride_conv_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
        activation)


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=(1, 2, 2), padding=(1, 1, 1),
                           output_padding=(0, 1, 1), bias=False),
        activation)


class InitFeaExtract(nn.Module):
    def __init__(self, channel):
        super(InitFeaExtract, self).__init__()
        self.FEconv = nn.Sequential(
            nn.Conv2d(3, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        [b, c, u, v, h, w] = x.size()
        x = rearrange(x, 'b c u v h w -> (b u v) c h w')
        buffer = self.FEconv(x)
        buffer = rearrange(buffer, '(b u v) c h w -> b c u v h w', u=u, v=v)
        return buffer


def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, criterion_data=None):
        SR = out['SR']
        # HR = LF_rgb2ycbcr(HR)[:, 0:1, :, :, :, :]
        loss = self.criterion_Loss(SR, HR)

        return loss