'''
@inproceedings{EDSR,
  title={Enhanced deep residual networks for single image super-resolution},
  author={Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Mu Lee, Kyoung},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages={136--144},
  year={2017}
}
'''
import torch
import torch.nn as nn
from utils.utils import LF_rgb2ycbcr, LF_ycbcr2rgb, LF_interpolate
from einops import rearrange


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.scale = args.scale_factor
        self.init_feature = nn.Conv2d(1, 256, 3, 1, 1)
        self.body = ResidualGroup(256, 32)
        if args.scale_factor == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(256, 256 * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(256, 1, 3, 1, 1))
        if args.scale_factor == 4:
            self.upscale = nn.Sequential(
                nn.Conv2d(256, 256 * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(256, 256 * 4, 1, 1, 0),
                nn.PixelShuffle(2),
                nn.Conv2d(256, 1, 3, 1, 1))


    def forward(self, lr, info=None):
        [b, c, u, v, h, w] = lr.size()
        # rgb2ycbcr
        lr_ycbcr = LF_rgb2ycbcr(lr)
        sr_ycbcr = LF_interpolate(lr_ycbcr, scale_factor=self.scale, mode='bicubic')
        x = rearrange(lr_ycbcr[:, 0:1, :, :, :, :], 'b c u v h w -> b c (u h) (v w)')

        buffer = self.init_feature(x)
        buffer = self.body(buffer)
        y = self.upscale(buffer)
        y = rearrange(y, 'b c (u h) (v w) -> b c u v h w', u=u, v=v)

        # ycbcr2rgb
        sr_ycbcr[:, 0:1, :, :, :, :] = y
        out = {}
        out['SR'] = LF_ycbcr2rgb(sr_ycbcr)
        return out


class ResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            ResB(n_feat) \
            for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResB(nn.Module):
    def __init__(self, n_feat):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, 3, 1, 1),
        )

    def forward(self, x):
        res = 0.1 * self.body(x)
        res = res + x
        return res


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, degrade_info=None):
        loss = self.criterion_Loss(out['SR'], HR)

        return loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    pass

