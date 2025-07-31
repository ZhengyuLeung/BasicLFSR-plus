'''
@article{Kalantari2016,
  title={Learning-based view synthesis for light field cameras},
  author={Kalantari, Nima Khademi and Wang, Ting-Chun and Ramamoorthi, Ravi},
  journal={ACM Transactions on Graphics (TOG)},
  volume={35},
  number={6},
  pages={1--10},
  year={2016},
  publisher={ACM New York, NY, USA}
}
'''
import torch
import torch.nn as nn
from einops import rearrange
from option import args
import importlib
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.DepthNet = DepthNetModel()
        self.ColorNet = ColorNetModel(channel_in=3*self.angRes_in**2)

    def forward(self, lf, info=None):
        depth_features = prepare_depth_features(lf)

        depth_features = rearrange(depth_features, 'b c u v h w -> (b u v) c h w')
        depthRes = self.DepthNet(depth_features)
        depthRes = rearrange(depthRes, '(b u v) c h w -> b c u v h w', u=self.angRes_in, v=self.angRes_in)
        depthRes = depthRes / (self.angRes_out - 1)

        # prepare_color_features
        warpedImages = warp_images(depthRes, lf, self.angRes_out)

        colorRes = self.ColorNet(rearrange(warpedImages, 'b c u v h w -> (b u v) c h w'))
        colorRes = rearrange(colorRes, '(b u v) c h w -> b c u v h w', u=self.angRes_out, v=self.angRes_out)

        out = {}
        out['SR'] = colorRes
        return out


def isnan(x):
    return x != x


def warp_images(disparity, input, angRes_out):
    [b, c, u, v, h, w] = input.size()
    # disparity = disparity.view(b, 1, 1, 1, h, w)
    angFactor = (angRes_out - 1) // (u - 1)

    # hw dimensions
    grid_H, grid_W = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid_H = grid_H.view(1, 1, 1, 1, h, w).to(input.device)
    grid_W = grid_W.view(1, 1, 1, 1, h, w).to(input.device)

    #  uv dimensions
    grid_U, grid_V = torch.meshgrid([torch.arange(angRes_out), torch.arange(angRes_out)])
    grid_U = grid_U.view(1, 1, angRes_out, angRes_out, 1, 1).to(input.device)
    grid_V = grid_V.view(1, 1, angRes_out, angRes_out, 1, 1).to(input.device)

    warped_images = []
    for i in range(u):
        for j in range(v):
            curInput = input[:, :, i, j, :, :]

            deltaU = grid_U - grid_U[:, :, ::angFactor, ::angFactor, :, :][:, :, i, j, :, :]  # - refPos[1]
            deltaV = grid_V - grid_V[:, :, ::angFactor, ::angFactor, :, :][:, :, i, j, :, :]  # - refPos[0]
            curH = grid_H - deltaU * disparity[:, :, i:i+1, j:j+1, :, :]
            curW = grid_W - deltaV * disparity[:, :, i:i+1, j:j+1, :, :]
            curH = rearrange(curH, 'b 1 u v h w -> b (u h) (v w) 1') / h * 2 - 1
            curW = rearrange(curW, 'b 1 u v h w -> b (u h) (v w) 1') / w * 2 - 1

            tmp = F.grid_sample(curInput, grid=torch.cat([curW, curH], dim=3).float(), padding_mode='zeros',
                                mode='bicubic', align_corners=True)
            tmp = rearrange(tmp, 'b c (u h) (v w) -> b c u v h w', h=h, w=w)
            warped_images.append(tmp)
            pass
        pass

    warped_images = torch.cat(warped_images, dim=1)
    return warped_images


def prepare_depth_features(inputLF):
    depthResolution = 100
    deltaDisparity = 21
    [b, c, u, v, h, w] = inputLF.size()

    # convert the input rgb light field to grayscale
    grayLF = (inputLF * torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1, 1, 1).to(inputLF.device))
    grayLF = rearrange(grayLF.mean(1, keepdims=True), 'b c u v h w -> c b u v h w')

    #
    defocusStack = torch.zeros((b, u, v, h, w, depthResolution)).to(inputLF.device)
    correspStack = torch.zeros((b, u, v, h, w, depthResolution)).to(inputLF.device)
    delta = 2 * deltaDisparity / (depthResolution - 1)

    # hw dimensions
    grid_H, grid_W = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid_H = grid_H.view(1, 1, 1, 1, h, w).to(inputLF.device)
    grid_W = grid_W.view(1, 1, 1, 1, h, w).to(inputLF.device)

    #  uv dimensions
    grid_U, grid_V = torch.meshgrid([torch.arange(u), torch.arange(v)])
    grid_U = grid_U.view(1, 1, u, v, 1, 1).to(inputLF.device)
    grid_V = grid_V.view(1, 1, u, v, 1, 1).to(inputLF.device)

    indDepth = 0
    for curDepth in np.arange(- deltaDisparity, deltaDisparity + delta, delta):
        shearedLF = []
        for i in range(0, u):
            for j in range(0, v):
                deltaU = grid_U - grid_U[:, :, i, j, :, :]
                deltaV = grid_V - grid_V[:, :, i, j, :, :]
                curH = grid_H + curDepth * deltaU
                curW = grid_W + curDepth * deltaV
                curH = rearrange(curH, 'b 1 u v h w -> b (u h) (v w) 1') / h * 2 - 1
                curW = rearrange(curW, 'b 1 u v h w -> b (u h) (v w) 1') / w * 2 - 1

                tmp = F.grid_sample(grayLF[:, :, i, j, :, :], grid=torch.cat([curW, curH], dim=3).float(),
                                    padding_mode='zeros', mode='bicubic', align_corners=True)
                tmp = rearrange(tmp, 'c b (u h) (v w) -> b c u v h w', h=h, w=w)
                shearedLF.append(tmp)

        shearedLF = torch.cat(shearedLF, dim=1)
        shearedLF = shearedLF.masked_fill(shearedLF == 0, torch.nan)

        defocusStack[:, :, :, :, :, indDepth] = torch.nanmean(shearedLF, dim=1)
        correspStack[:, :, :, :, :, indDepth] = torch.var(shearedLF, dim=1)
        indDepth = indDepth + 1

    featuresStack = torch.cat([defocusStack, correspStack], dim=-1).permute(0, 5, 1, 2, 3, 4)
    featuresStack[torch.isnan(featuresStack)] = 0
    return featuresStack


def defocus_response(input):
    curMean = torch.nanmean(input, dim=1)
    curMean[torch.isnan(curMean)] = 0
    return curMean


def corresp_response(input):
    # import warnings
    # warnings.filterwarnings("ignore")
    inputVar = np.nanvar(input, 3, ddof=1)
    inputVar[np.isnan(inputVar)] = 0
    output = np.sqrt(inputVar)
    return output


class DepthNetModel(nn.Module):
    def __init__(self):
        super(DepthNetModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(200, 100, kernel_size=7, padding=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(100, 100, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(100, 50, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(50, 1, kernel_size=1, stride=1),
        )

    def forward(self, features):
        out = self.layer(features)
        return out


class ColorNetModel(nn.Module):
    def __init__(self, channel_in):
        super(ColorNetModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel_in, 100, 7, padding=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(100, 100, 5, padding=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(100, 50, 3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(50, 3, 1, stride=1),
        )

    def forward(self, features):
        out = self.layer(features)
        return out


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
    args.model_name = 'Kalantari'
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
    GPU_cost1 = (torch.cuda.memory_allocated(0) - GPU_cost) / 1024 / 1024 / 1024  # GB
    print('   GPU consumption: %.4fGB' % (GPU_cost1))

    input = torch.randn([1, 3, args.angRes_in, args.angRes_in, 32, 32]).to(device)
    flops = FlopCountAnalysis(net, input).total()
    # print('   Model Parameters:', parameter_count_table(net))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))

    # start = time.time()
    # for _ in range(100):
    #     input = torch.randn([1, 3, args.angRes_in, args.angRes_in, 32, 32]).to(device)
    #     output = net(input)
    # end = time.time()
    # print('   Time used: %.4fs' % ((end - start) / 100))