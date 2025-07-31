'''
@article{P4DCNN,
  title={High-fidelity view synthesis for light field imaging with extended pseudo 4DCNN},
  author={Wang, Yunlong and Liu, Fei and Zhang, Kunbo and Wang, Zilei and Sun, Zhenan and Tan, Tieniu},
  journal={IEEE Transactions on Computational Imaging},
  volume={6},
  pages={830--842},
  year={2020},
  publisher={IEEE}
}
'''
import torch
import torch.nn as nn
from einops import rearrange
from option import args
import importlib
import torch.nn.functional as F


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.angFactor = (self.angRes_out - 1) // (self.angRes_in - 1)

        self.Layer1 = nn.ConvTranspose2d(3, 3, kernel_size=(5, 5), stride=(self.angFactor, 1), padding=(2, 2))
        self.Layer2 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.ReLU(True),
            nn.Conv3d(64, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(16, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(64, 3, kernel_size=(3, 9, 9), padding=(1, 4, 4)),
        )
        self.Layer3 = nn.ConvTranspose2d(3, 3, kernel_size=(5, 5), stride=(self.angFactor, 1), padding=(2, 2))
        self.Layer4 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.ReLU(True),
            nn.Conv3d(64, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(16, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(64, 3, kernel_size=(3, 9, 9), padding=(1, 4, 4)),
        )

    def forward(self, x, info):
        [b, c, u, v, h, w] = x.size()

        x = rearrange(x, 'b c u v h w -> (b u h) c v w')
        bilinear_x = self.Layer1(x)
        bilinear_x = rearrange(bilinear_x, '(b u h) c v w -> (b u) c v h w', b=b, h=h)
        x = self.Layer2(bilinear_x) + bilinear_x

        x = rearrange(x, '(b u) c v h w -> (b v w) c u h', b=b)
        bilinear_x = self.Layer3(x)
        bilinear_x = rearrange(bilinear_x, '(b v w) c u h -> (b v) c u h w', b=b, w=w)
        x = self.Layer4(bilinear_x) + bilinear_x

        x = rearrange(x, '(b v) c u h w -> b c u v h w', b=b)

        out = {}
        out['SR'] = x
        return out


class Interpolation(nn.Module):
    def __init__(self, angRes_out):
        super(Interpolation, self).__init__()
        self.angRes_out = angRes_out

    def forward(self, buffer):
        [b, c, u, h] = buffer.size()
        buffer = F.interpolate(buffer, size=[self.angRes_out, h], mode='bilinear')
        return buffer


class CNN3D(nn.Module):
    def __init__(self, channels):
        super(CNN3D, self).__init__()
        self.Layers = nn.Sequential(
            nn.Conv3d(3, channels, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=True),
            nn.ReLU(True),
            nn.Conv3d(channels, channels // 2, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=True),
            nn.ReLU(True),
            nn.Conv3d(channels // 2, 3, kernel_size=(3, 9, 9), padding=(1, 4, 4), bias=True)
        )

    def forward(self, buffer):
        buffer = self.Layers(buffer) + buffer
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
    args.model_name = 'P4DCNN'
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