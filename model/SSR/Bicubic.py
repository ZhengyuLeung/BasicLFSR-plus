import torch
import torch.nn as nn
from utils.utils import LF_interpolate


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.scale = args.scale_factor
        self.conv_init = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, lr, degrade_info=None):
        # Bicubic
        lr_upscale = LF_interpolate(lr, scale_factor=self.scale, mode='bicubic')

        out = {}
        out['SR'] = lr_upscale

        return out


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, degrade_info=None):
        loss = self.criterion_Loss(out['SR'], HR)

        return loss


def weights_init(m):
    pass

