'''
@InProceedings{Yeung2018,
author = {Yeung, Henry Wing Fung and Hou, Junhui and Chen, Jie and Chung, Yuk Ying and Chen, Xiaoming},
title = {Fast Light Field Reconstruction With Deep Coarse-To-Fine Modeling of Spatial-Angular Clues},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
'''
import torch
import torch.nn as nn
from einops import rearrange
from option import args
import importlib


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.angFactor = (self.angRes_out - 1) // (self.angRes_in - 1)
        channels = 64
        depth = 8

        # View Synthesis Netowrk
        self.pre_conv = nn.Sequential(
            Conv4d(3, channels, channels, k=(3, 3, 3, 3), p=(1, 1, 1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Alternating Filtering
        layers = []
        for i in range(depth):
            layers.append(Conv4d(channels, channels, channels, k=(3, 3, 3, 3), p=(1, 1, 1, 1)))
            layers.append(nn.LeakyReLU(0.2, True))
        self.deep_conv = nn.Sequential(*layers)

        # residual prediction layer
        self.prediction = nn.Sequential(
            Conv4d(channels, channels, channel_out=3*self.angRes_out**2,
                   k=(self.angRes_in, self.angRes_in, 3, 3), p=(0, 0, 1, 1))
        )

        # View Refinement Network
        self.ang_dim_reduction = nn.Sequential(
            Conv4d(3, channels//4, channels//4, k=(3, 3, 3, 3), p=(0, 0, 1, 1)),
            nn.LeakyReLU(0.2, True),
            Conv4d(channels//4, channels, channels, k=(3, 3, 3, 3), p=(0, 0, 1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Fine Details Recovery
        self.fine_details_recovery = nn.Sequential(
            Conv4d(channels, channels, 3*self.angRes_out**2, k=(self.angRes_out-4, self.angRes_out-4, 3, 3),
                   p=(0, 0, 1, 1)),
            nn.LeakyReLU(0.2, True),
        )
        # self.conv_post = nn.Sequential(
        #     Conv4d(16, channels, channels, k=(3, 3, 3, 3)),
        #     nn.LeakyReLU(0.2, True)
        # )

    def forward(self, lf, info):
        # View Synthesis Netowrk
        buffer = self.pre_conv(lf)  # Feature Extraction
        buffer = self.deep_conv(buffer)  # Alternating Filtering
        buffer = self.prediction(buffer)  # Novel SAIs Synthesis
        buffer = rearrange(buffer, 'b (c u v) 1 1 h w -> b c u v h w', c=3, u=self.angRes_out, v=self.angRes_out)

        # Reshape & Concat
        buffer[:, :, ::self.angFactor, ::self.angFactor, :, :] = lf

        # View Refinement Network
        buffer = self.ang_dim_reduction(buffer)

        # Fine Details Recovery
        buffer = self.fine_details_recovery(buffer)
        sr = rearrange(buffer, 'b (c u v) 1 1 h w -> b c u v h w', c=3, u=self.angRes_out, v=self.angRes_out)

        out = {}
        out['SR'] = sr
        return out


class Conv4d(nn.Module):
    def __init__(self, chnnal_in, channels, channel_out, k, p):
        super(Conv4d, self).__init__()
        self.pre_conv4d_spa = nn.Sequential(
            nn.Conv2d(chnnal_in, channels, kernel_size=(k[2], k[3]), padding=(p[2], p[3]), bias=True),
        )
        self.pre_conv4d_ang = nn.Sequential(
            nn.Conv2d(channels, channel_out, kernel_size=(k[0], k[1]), padding=(p[0], p[1]), bias=True),
        )

    def forward(self, buffer):
        [b, c, u, v, h, w] = buffer.size()
        buffer = rearrange(buffer, 'b c u v h w -> (b u v) c h w')
        buffer = self.pre_conv4d_spa(buffer)
        buffer = rearrange(buffer, '(b u v) c h w -> (b h w) c u v', u=u, v=v)
        buffer = self.pre_conv4d_ang(buffer)
        buffer = rearrange(buffer, '(b h w) c u v -> b c u v h w', h=h, w=w)
        return buffer


def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, criterion_data=None):
        SR = out['SR']
        loss = self.criterion_Loss(SR, HR)

        return loss


if __name__ == "__main__":
    args.model_name = 'Yeung2018'
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

    input = torch.randn([2, 3, args.angRes_in, args.angRes_in, 64, 128]).to(device)
    GPU_cost = torch.cuda.memory_allocated(0)
    out = net(input, [])