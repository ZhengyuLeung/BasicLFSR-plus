'''
@inproceedings{Pseudo4DCNN,
  title={End-to-end view synthesis for light field imaging with pseudo 4DCNN},
  author={Wang, Yunlong and Liu, Fei and Wang, Zilei and Hou, Guangqi and Sun, Zhenan and Tan, Tieniu},
  booktitle={Proceedings of the european conference on computer vision (ECCV)},
  pages={333--348},
  year={2018}
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
        channels = 64
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.angFactor = (self.angRes_out - 1) // (self.angRes_in - 1)

        self.row_interpolation = Interpolation(self.angRes_out)
        self.col_interpolation = Interpolation(self.angRes_out)
        self.row_restoration = CNN3D(channels)
        self.col_restoration = CNN3D(channels)

    def forward(self, lf, info):
        [b, c, u, v, h, w] = lf.size()

        lf = rearrange(lf, 'b c u v h w -> (b u h) c v w')
        lf = self.row_interpolation(lf)
        lf = rearrange(lf, '(b u h) c v w -> (b u) c v h w', b=b, h=h)
        lf = self.row_restoration(lf)
        lf = rearrange(lf, '(b u) c v h w -> b c u v h w', b=b)

        lf = rearrange(lf, 'b c u v h w -> (b v w) c u h')
        lf = self.col_interpolation(lf)
        lf = rearrange(lf, '(b v w) c u h -> (b v) c u h w', b=b, w=w)
        lf = self.col_restoration(lf)
        lf = rearrange(lf, '(b v) c u h w -> b c u v h w', b=b)

        out = {}
        out['SR'] = lf
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
    args.model_name = 'Pseudo4DCNN'
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