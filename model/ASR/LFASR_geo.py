'''
@inproceedings{LFASR-geo,
  title={Learning light field angular super-resolution via a geometry-aware network},
  author={Jin, Jing and Hou, Junhui and Yuan, Hui and Kwong, Sam},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={34},
  number={07},
  pages={11141--11148},
  year={2020}
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
        an2 = args.angRes_out ** 2
        num_source = self.angRes_in ** 2

        # disparity
        self.disp_estimator = nn.Sequential(
            nn.Conv2d(num_source, 16, kernel_size=7, stride=1, dilation=2, padding=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=7, stride=1, dilation=2, padding=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, an2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(an2, an2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(an2, an2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(an2, an2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(an2, an2, kernel_size=3, stride=1, padding=1),
        )

        # LF
        self.lf_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=num_source, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.lf_altblock = make_Altlayer(layer_num=4, an=self.angRes_out, ch=64)
        if self.angRes_out == 9:
            self.lf_res_conv = nn.Sequential(
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(5, 3, 3), stride=(4, 1, 1), padding=(0, 1, 1)),
                # 81->20
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(5, 3, 3), stride=(3, 1, 1), padding=(0, 1, 1)),
                # 20->6
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=81, kernel_size=(6, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                # 6-->1
            )

        elif self.angRes_out == 8:
            self.lf_res_conv = nn.Sequential(
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(4, 3, 3), stride=(4, 1, 1), padding=(0, 1, 1)),
                # 64-->16
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(4, 3, 3), stride=(4, 1, 1), padding=(0, 1, 1)),
                # 16-->4
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(4, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                # 4-->1
            )

        elif self.angRes_out == 7:
            self.lf_res_conv = nn.Sequential(
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(5, 3, 3), stride=(4, 1, 1), padding=(0, 1, 1)),
                # 49-->12
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(4, 3, 3), stride=(4, 1, 1), padding=(0, 1, 1)),
                # 12-->3
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=64, out_channels=49, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                # 3-->1
            )

    def forward(self, lr, info):
        lr = LF_rgb2ycbcr(lr)[:, 0:1, :, :, :, :]
        lr = rearrange(lr, 'b 1 u v h w -> b (u v) h w')
        ind_source = torch.tensor([0, 6, 42, 48])

        an = self.angRes_out
        an2 = self.angRes_out**2

        # ind_source
        N, num_source, h, w = lr.shape  # [N,num_source,h,w]
        ind_source = torch.squeeze(ind_source)  # [num_source]

        #################### disparity estimation ##############################
        disp_target = self.disp_estimator(lr)  # [N,an2,h,w]

        #################### intermediate LF ##############################
        warp_img_input = lr.contiguous().view(N * num_source, 1, h, w).repeat(an2, 1, 1, 1)  # [N*an2*4,1,h,w]

        grid = []
        for k_t in range(0, an2):
            for k_s in range(0, num_source):
                ind_s = ind_source[k_s].type_as(lr)
                ind_t = torch.arange(an2)[k_t].type_as(lr)
                ind_s_h = torch.floor(ind_s / an)
                ind_s_w = ind_s % an
                ind_t_h = torch.floor(ind_t / an)
                ind_t_w = ind_t % an
                disp = disp_target[:, k_t, :, :]

                XX = torch.arange(0, w).view(1, 1, w).expand(N, h, w).type_as(lr)  # [N,h,w]
                YY = torch.arange(0, h).view(1, h, 1).expand(N, h, w).type_as(lr)
                grid_w_t = XX + disp * (ind_t_w - ind_s_w)
                grid_h_t = YY + disp * (ind_t_h - ind_s_h)
                grid_w_t_norm = 2.0 * grid_w_t / (w - 1) - 1.0
                grid_h_t_norm = 2.0 * grid_h_t / (h - 1) - 1.0
                grid_t = torch.stack((grid_w_t_norm, grid_h_t_norm), dim=3)  # [N,h,w,2]
                grid.append(grid_t)
        grid = torch.cat(grid, 0)  # [N*an2*4,h,w,2]

        warped_img = F.grid_sample(warp_img_input, grid, align_corners=False).view(N, an2, num_source, h, w)

        ################# refine LF ###########################
        feat = self.lf_conv0(warped_img.view(N * an2, num_source, h, w))  # [N*an2,64,h,w]
        feat = self.lf_altblock(feat)  # [N*an2,64,h,w]
        feat = torch.transpose(feat.view(N, an2, 64, h, w), 1, 2)  # [N,64,an2,h,w]
        res = self.lf_res_conv(feat)  # [N,an2,1,h,w]

        lf = warped_img[:, :, 0, :, :] + torch.squeeze(res, 2)  # [N,an2,h,w]

        out = {}
        out['SR'] = rearrange(lf, 'b (u v) h w -> b 1 u v h w', u=self.angRes_out, v=self.angRes_out)
        out['disp_target'] = disp_target
        out['warped_img'] = rearrange(warped_img, 'b (uu vv) (u v) h w -> b 1 uu vv (u v) h w',
                                      uu=self.angRes_out, vv=self.angRes_out, u=self.angRes_in, v=self.angRes_in)

        return out


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


def weights_init(m):
    pass


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    @staticmethod
    def smooth_loss(pred_map):
        def gradient(pred):
            D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy
        # [N,an2,h,w]
        loss = 0
        weight = 1.
        dx, dy = gradient(pred_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean()) * weight
        return loss

    def epi_loss(self, pred, label):
        def gradient(pred):
            D_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy
        # epi loss

        def lf2epi(lf):
            epi_h = rearrange(lf, 'b c u v h w -> (b u h) c v w')
            epi_v = rearrange(lf, 'b c u v h w -> (b v w) c u h')
            return epi_h, epi_v

        epi_h_pred, epi_v_pred = lf2epi(pred)
        dx_h_pred, dy_h_pred = gradient(epi_h_pred)
        dx_v_pred, dy_v_pred = gradient(epi_v_pred)

        epi_h_label, epi_v_label = lf2epi(label)
        dx_h_label, dy_h_label = gradient(epi_h_label)
        dx_v_label, dy_v_label = gradient(epi_v_label)

        return self.criterion_Loss(dx_h_pred, dx_h_label) + self.criterion_Loss(dy_h_pred, dy_h_label) + \
               self.criterion_Loss(dx_v_pred, dx_v_label) + self.criterion_Loss(dy_v_pred, dy_v_label)

    def forward(self, out, HR, criterion_data=None):
        pred_views = out['warped_img']
        disp_target = out['disp_target']
        HR = LF_rgb2ycbcr(HR)[:, 0:1, :, :, :, :]
        smooth = 0.001
        epi = 1.0
        loss = self.criterion_Loss(out['SR'], HR) + smooth * self.smooth_loss(disp_target)
        for i in range(pred_views.shape[4]):
            loss += self.criterion_Loss(pred_views[:, :, :, :, i, :, :], HR)
        loss += epi * self.epi_loss(out['SR'], HR)
        return loss

