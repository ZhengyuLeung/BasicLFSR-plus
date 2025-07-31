'''
@article{FS-GAF,
  title={Deep coarse-to-fine dense light field reconstruction with flexible sampling and geometry-aware fusion},
  author={Jin, Jing and Hou, Junhui and Chen, Jie and Zeng, Huanqiang and Kwong, Sam and Yu, Jingyi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={4},
  pages={1819--1836},
  year={2020},
  publisher={IEEE}
}
'''
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from utils.utils import LF_rgb2ycbcr


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out

        if 'RE_HCI' in args.data_list_for_train:
            psv_range = 4
        elif 'RE_Lytro' in args.data_list_for_train:
            psv_range = 2

        # coarse SAI synthesis network
        self.net_view = Net_view(num_source=self.angRes_in**2, psv_step=50, angRes_out=self.angRes_out, psv_range=psv_range)
        # efficient LF refinement network
        self.net_refine = Net_refine(layer_num=4, angRes_out=self.angRes_out)

    def forward(self, lr, info):
        lr = LF_rgb2ycbcr(lr)[:, 0:1, :, :, :, :]
        lr = rearrange(lr, 'b 1 u v h w -> b (u v) h w')
        ind_source = torch.tensor([0, 6, 42, 48])

        # coarse view synthesis
        disp_lf, inter_lf = self.net_view(lr, ind_source)
        lf = self.net_refine(inter_lf)

        out = {}
        out['SR'] = rearrange(lf, 'b (u v) h w -> b 1 u v h w', u=self.angRes_out, v=self.angRes_out)
        # out['disp_lf'] = disp_lf
        out['inter_lf'] = rearrange(inter_lf, 'b (u v) h w -> b 1 u v h w', u=self.angRes_out, v=self.angRes_out)

        return out


class Net_view(nn.Module):
    def __init__(self, num_source, psv_step, angRes_out, psv_range):
        super(Net_view, self).__init__()
        self.angRes_out = angRes_out
        self.psv_step = psv_step
        self.psv_range = psv_range
        self.conv_perPlane = nn.Sequential(
            nn.Conv2d(num_source, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )

        self.conv_crossPlane = nn.Sequential(
            nn.Conv2d(4 * psv_step, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(200, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(200, psv_step, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )

        self.conv_disp = nn.Sequential(
            nn.Conv2d(psv_step, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1 + num_source, kernel_size=3, stride=1, padding=1),
        )

        self.softmax_d1 = nn.Softmax(dim=1)
        self.softmax_d2 = nn.Softmax(dim=2)

    def forward(self, img_source, ind_source):
        an = self.angRes_out
        an2 = self.angRes_out ** 2

        N, num_source, h, w = img_source.shape  # [N,4,c,h,w]

        D = self.psv_step
        disp_range = torch.linspace(-1 * self.psv_range, self.psv_range, steps=D).type_as(img_source)  # [D]
        crop_size = 0
        h_c = h - 2 * crop_size
        w_c = w - 2 * crop_size

        if self.training:
            # PSV
            psv_input = img_source.contiguous().view(N * num_source, 1, h, w).repeat(D * an2, 1, 1, 1)  # [N*an2*D*4,1,h,w]
            grid = construct_psv_grid(an, D, num_source, ind_source, disp_range, N, h, w)  # [N*an2*D*4,h,w,2]
            PSV = F.grid_sample(psv_input, grid, align_corners=False).view(N, an2, D, num_source, h, w)  # [N*an2*D*4,1,h,w]-->[N,an2,D,4,h,w]
            PSV = crop_boundary(PSV, crop_size)

            # disparity & confidence estimation
            perPlane_out = self.conv_perPlane(PSV.view(N * an2 * D, num_source, h_c, w_c))  # [N*an2*D,4,h,w]
            crossPlane_out = self.conv_crossPlane(perPlane_out.view(N * an2, D * 4, h_c, w_c))  # [N*an2,D,h,w]
            disp_out = self.conv_disp(crossPlane_out)  # [N*an2,5,h,w]
            disp_target = disp_out[:, 0, :, :].view(N, an2, h_c, w_c)  # disparity for each view
            disp_target = F.pad(disp_target, pad=[crop_size, crop_size, crop_size, crop_size], mode='constant', value=0)
            conf_source = disp_out[:, 1:, :, :].view(N, an2, num_source, h_c, w_c)  # confidence of source views for each view
            conf_source = self.softmax_d2(conf_source)

            # intermediate LF
            warp_img_input = img_source.view(N * num_source, 1, h, w).repeat(an2, 1, 1, 1)  # [N*an2*4,1,h,w]
            grid = construct_syn_grid(an, num_source, ind_source, disp_target, N, h, w)  # [N*an2*4,h,w,2]
            warped_img = F.grid_sample(warp_img_input, grid, align_corners=False).view(N, an2, num_source, h, w)  # {N,an2,4,h,w]
            warped_img = crop_boundary(warped_img, crop_size)

            inter_lf = torch.sum(warped_img * conf_source, dim=2)  # [N,an2,h,w]
            return disp_target, inter_lf

        else:
            inter_lf = torch.zeros((N, an2, h_c, w_c)).type_as(img_source)
            for k_t in range(0, an2):  # for each target view
                ind_t = torch.arange(an2)[k_t]
                # disparity & confidence estimation
                PSV = torch.zeros((N, D, num_source, h, w)).type_as(img_source)
                for step in range(0, D):
                    for k_s in range(0, num_source):
                        ind_s = ind_source[k_s]
                        disp = disp_range[step]
                        PSV[:, step, k_s] = warping(disp, ind_s, ind_t, img_source[:, k_s], an)

                PSV = crop_boundary(PSV, crop_size)

                perPlane_out = self.conv_perPlane(PSV.view(N * D, num_source, h_c, w_c))  # [N*D,4,h,w]
                crossPlane_out = self.conv_crossPlane(perPlane_out.view(N, D * num_source, h_c, w_c))  # [N,D,h,w]
                disp_out = self.conv_disp(crossPlane_out)  # [N,5,h,w]
                disp_target = disp_out[:, 0, :, :]  # [N,h,w] disparity for each view
                # disp_target = F.pad(disp_target, pad=[crop_size, crop_size, crop_size, crop_size], mode='constant', value=0)
                conf_source = disp_out[:, 1:, :, :]  # [N,4,h_c,w_c] confidence of source views for each view
                conf_source_norm = self.softmax_d1(conf_source)

                # warping source views
                warped_img = torch.zeros(N, num_source, h, w).type_as(img_source)
                for k_s in range(0, num_source):
                    ind_s = ind_source[k_s]
                    disp = disp_target
                    warped_img[:, k_s] = warping(disp, ind_s, ind_t, img_source[:, k_s], an)
                warped_img = crop_boundary(warped_img, crop_size)

                inter_view = torch.sum(warped_img * conf_source_norm, dim=1)  # [N,h,w]
                inter_lf[:, k_t] = inter_view

            return None, inter_lf


def construct_psv_grid(an, D, num_source, ind_source, disp_range, N, h, w):
    grid = []
    for k_t in range(0, an*an):
        for step in range(0, D):
            for k_s in range(0, num_source):
                ind_s = ind_source[k_s].type_as(disp_range)
                ind_t = torch.arange(an*an)[k_t].type_as(disp_range)
                ind_s_h = torch.floor(ind_s / an)
                ind_s_w = ind_s % an
                ind_t_h = torch.floor(ind_t / an)
                ind_t_w = ind_t % an
                disp = disp_range[step]

                XX = torch.arange(0, w).view(1, 1, w).expand(N, h, w).type_as(disp_range)  # [N,h,w]
                YY = torch.arange(0, h).view(1, h, 1).expand(N, h, w).type_as(disp_range)

                grid_w_t = XX + disp * (ind_t_w - ind_s_w)
                grid_h_t = YY + disp * (ind_t_h - ind_s_h)

                grid_w_t_norm = 2.0 * grid_w_t / (w - 1) - 1.0
                grid_h_t_norm = 2.0 * grid_h_t / (h - 1) - 1.0

                grid_t = torch.stack((grid_w_t_norm, grid_h_t_norm), dim=3)  # [N,h,w,2]
                grid.append(grid_t)

    grid = torch.cat(grid, 0)  # [N*an2*D*4,h,w,2]
    return grid


def construct_syn_grid(an, num_source, ind_source, disp_target, N, h, w):
    grid = []
    for k_t in range(0, an*an):
        for k_s in range(0, num_source):
            ind_s = ind_source[k_s].type_as(disp_target)
            ind_t = torch.arange(an*an)[k_t].type_as(disp_target)
            ind_s_h = torch.floor(ind_s / an)
            ind_s_w = ind_s % an
            ind_t_h = torch.floor(ind_t / an)
            ind_t_w = ind_t % an
            disp = disp_target[:, torch.arange(an*an)[k_t], :, :]

            XX = torch.arange(0, w).view(1, 1, w).expand(N, h, w).type_as(disp_target)  # [N,h,w]
            YY = torch.arange(0, h).view(1, h, 1).expand(N, h, w).type_as(disp_target)
            grid_w_t = XX + disp * (ind_t_w - ind_s_w)
            grid_h_t = YY + disp * (ind_t_h - ind_s_h)
            grid_w_t_norm = 2.0 * grid_w_t / (w - 1) - 1.0
            grid_h_t_norm = 2.0 * grid_h_t / (h - 1) - 1.0
            grid_t = torch.stack((grid_w_t_norm, grid_h_t_norm), dim=3)  # [N,h,w,2]
            grid.append(grid_t)

    grid = torch.cat(grid, 0)  # [N*an2*4,h,w,2]
    return grid


class Net_refine(nn.Module):
    def __init__(self, layer_num, angRes_out):
        super(Net_refine, self).__init__()
        self.lf_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.lf_altblock = make_Altlayer(layer_num=layer_num, an=angRes_out, ch=64)
        self.lf_res_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, inter_lf):
        N, an2, h, w = inter_lf.shape
        feat = self.lf_conv0(inter_lf.view(N * an2, 1, h, w))  # [N*an2,64,h,w]
        feat = self.lf_altblock(feat)  # [N*an2,64,h,w]
        res = self.lf_res_conv(feat).view(N, an2, h, w)  # [N*an2,1,h,w]-->[N,an2,h,w]

        lf = inter_lf + res  # [N,an2,h,w]

        return lf


def make_Altlayer(layer_num, an, ch):
    layers = []
    for i in range(layer_num):
        layers.append(AltFilter(an, ch))
    return nn.Sequential(*layers)


class AltFilter(nn.Module):
    def __init__(self, an, ch):
        super(AltFilter, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.spaconv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=(3, 3), stride=1, padding=2, dilation=2)
        self.angconv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=(3, 3), stride=1, padding=(1, 1))

        self.an = an
        self.an2 = an * an

    def forward(self, x):
        N, c, h, w = x.shape  # [N*81,c,h,w]
        N = N // self.an2

        out = self.relu(self.spaconv(x))  # [N*81,c,h,w]

        out = out.view(N, self.an2, c, h * w)
        out = torch.transpose(out, 1, 3).contiguous()  # [N,h*w,c,81]
        out = out.view(N * h * w, c, self.an, self.an)  # [N*h*w,c,9,9]

        out = self.relu(self.angconv(out))  # [N*h*w,c,9,9]

        out = out.view(N, h * w, c, self.an2)
        out = torch.transpose(out, 1, 3).contiguous()
        out = out.view(N * self.an2, c, h, w)  # [N*81,c,h,w]

        return out

def warping(disp, ind_source, ind_target, img_source, an):
    '''warping one source image/map to the target'''

    # disp:       [scale] or [N,h,w]
    # ind_souce:  (int)
    # ind_target: (int)
    # img_source: [N,h,w]
    # an:         angular number
    # ==> out:    [N,1,h,w]

    N, h, w = img_source.shape
    ind_source = ind_source.type_as(disp)
    ind_target = ind_target.type_as(disp)

    # coordinate for source and target
    ind_h_source = torch.floor(ind_source / an)
    ind_w_source = ind_source % an

    ind_h_target = torch.floor(ind_target / an)
    ind_w_target = ind_target % an

    # generate grid
    XX = torch.arange(0, w).view(1, 1, w).expand(N, h, w).type_as(img_source)  # [N,h,w]
    YY = torch.arange(0, h).view(1, h, 1).expand(N, h, w).type_as(img_source)

    grid_w = XX + disp * (ind_w_target - ind_w_source)
    grid_h = YY + disp * (ind_h_target - ind_h_source)

    grid_w_norm = 2.0 * grid_w / (w - 1) - 1.0
    grid_h_norm = 2.0 * grid_h / (h - 1) - 1.0

    grid = torch.stack((grid_w_norm, grid_h_norm), dim=3)  # [N,h,w,2]

    # inverse warp
    img_source = torch.unsqueeze(img_source, 0)
    img_target = F.grid_sample(img_source, grid, align_corners=False)  # [N,1,h,w]
    img_target = torch.squeeze(img_target, 1)  # [N,h,w]

    return img_target


def crop_boundary(I, crop_size):
    '''crop the boundary (the last 2 dimensions) of a tensor'''
    if crop_size == 0:
        return I

    if crop_size > 0:
        size = list(I.shape)
        I_crop = I.view(-1, size[-2], size[-1])
        I_crop = I_crop[:, crop_size:-crop_size, crop_size:-crop_size]
        size[-1] -= crop_size * 2
        size[-2] -= crop_size * 2
        I_crop = I_crop.view(size)
        return I_crop


def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, criterion_data=None):
        loss = self.criterion_Loss(out['SR'], HR) + self.criterion_Loss(out['inter_lf'], HR)

        return loss

