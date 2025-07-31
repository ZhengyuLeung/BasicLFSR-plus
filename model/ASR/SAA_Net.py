'''
@article{SAA_Net,
  title={Spatial-angular attention network for light field reconstruction},
  author={Wu, Gaochang and Wang, Yingqian and Liu, Yebin and Fang, Lu and Chai, Tianyou},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={8999--9013},
  year={2021},
  publisher={IEEE}
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
        self.model_column = model(args.angRes_in, args.angRes_out)
        self.model_row = model(args.angRes_in, args.angRes_out)
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.angFactor = (self.angRes_out - 1) // (self.angRes_in - 1)

    def forward(self, lf, info):
        [b, c, u, v, h, w] = lf.size()

        # Column reconstruction
        lf = rearrange(lf, 'b c u v h w -> (b u) c v h w')
        lf = self.model_column(lf)

        # Row reconstruction
        lf = rearrange(lf, '(b u) c v h w -> (b v) c u h w', b=b)
        lf = self.model_row(lf)
        lf = rearrange(lf, '(b v) c u h w -> b c u v h w', b=b)

        out = {}
        out['SR'] = lf
        return out


class model(nn.Module):
    def __init__(self, angRes_in, angRes_out):
        super(model, self).__init__()
        self.angRes_in = angRes_in
        self.angRes_out = angRes_out
        chn_base = 24

        # Group 1
        self.Group1_conv1 = nn.Sequential(
            nn.Conv3d(3, chn_base, kernel_size=(3, 1, 3), padding=(1, 0, 1)),
            nn.ReLU(True),
            nn.Conv3d(chn_base, chn_base, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(True)
        )
        self.Group1_deconv = nn.Sequential(
            nn.ConvTranspose3d(chn_base, chn_base, kernel_size=(self.angRes_out, 1, 3), padding=(0, 0, 1),
                               stride=(self.angRes_out // 2, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(chn_base, chn_base, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.ReLU(True),
        )
        self.Group1_conv2 = nn.Sequential(
            nn.Conv3d(chn_base, chn_base * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 2, 2)),
            nn.ReLU(True),
        )

        # Group 2
        self.Group2_conv1 = nn.Sequential(
            nn.Conv3d(chn_base * 2, chn_base * 2, kernel_size=(3, 1, 3), padding=(1, 0, 1)),
            nn.ReLU(True),
            nn.Conv3d(chn_base * 2, chn_base * 2, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(True)
        )
        self.Group2_deconv = nn.Sequential(
            nn.ConvTranspose3d(chn_base * 2, chn_base * 2, kernel_size=(self.angRes_out, 1, 3), padding=(0, 0, 1),
                               stride=(self.angRes_out // 2, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(chn_base * 2, chn_base * 2, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.ReLU(True),
        )
        self.Group2_conv2 = nn.Sequential(
            nn.Conv3d(chn_base * 2, chn_base * 4, kernel_size=(1, 1, 3), padding=(0, 0, 1), stride=(1, 1, 2)),
            nn.ReLU(True),
        )

        # Layer 3, shrinking
        self.Layer3 = nn.Sequential(
            nn.Conv3d(chn_base * 4, chn_base * 2, kernel_size=(1, 1, 1), padding=(0, 0, 0), stride=(1, 1, 1)),
            nn.ReLU(True),
        )

        # Group 4, Mapping
        self.Group4 = nn.Sequential(
            nn.Conv3d(chn_base * 2, chn_base * 2, kernel_size=(3, 1, 3), padding=(1, 0, 1), stride=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(chn_base * 2, chn_base * 2, kernel_size=(3, 3, 1), padding=(1, 1, 0), stride=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(chn_base * 2, chn_base * 2, kernel_size=(3, 1, 3), padding=(1, 0, 1), stride=(1, 1, 1)),
            nn.ReLU(True),
            nn.Conv3d(chn_base * 2, chn_base * 2, kernel_size=(3, 3, 1), padding=(1, 1, 0), stride=(1, 1, 1)),
            nn.ReLU(True),
        )

        # Group 5, Attention
        self.SAAM = SAAM(chn_base * 2, self.angRes_in, self.angRes_out)

        # Layer 6, Expanding
        self.Layer6_conv = nn.Sequential(
            nn.Conv3d(chn_base * self.angRes_in, chn_base * 4, kernel_size=(1, 1, 1)),
            nn.ReLU(True),
        )

        # Group 7
        self.Group7_deconv = nn.Sequential(
            nn.ConvTranspose3d(chn_base * 4, chn_base * 2, kernel_size=(1, 1, 4), padding=(0, 0, 1),
                               stride=(1, 1, 2)),
            nn.ReLU(True),
        )
        self.Group7_conv = nn.Sequential(
            nn.Conv3d(chn_base * 4, chn_base * 2, kernel_size=(3, 1, 3), padding=(1, 0, 1)),
            nn.ReLU(True),
            nn.Conv3d(chn_base * 2, chn_base * 2, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(True),
        )

        # Group 8
        self.Group8_deconv = nn.Sequential(
            nn.ConvTranspose3d(chn_base * 2, chn_base, kernel_size=(1, 4, 4), padding=(0, 1, 1),
                               stride=(1, 2, 2)),
            nn.ReLU(True),
        )
        self.Group8_conv = nn.Sequential(
            nn.Conv3d(chn_base * 2, chn_base, kernel_size=(3, 1, 3), padding=(1, 0, 1)),
            nn.ReLU(True),
            nn.Conv3d(chn_base, chn_base, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(True),
        )

        # Group 9
        self.Group9_conv = nn.Conv3d(chn_base, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))

    def forward(self, buffer):
        # Group 1
        buffer = self.Group1_conv1(buffer)
        buffer1 = self.Group1_deconv(buffer)
        buffer1 = buffer1[:, :, 0:-3, :, :]  # [bu, _, v, h, w]
        buffer = self.Group1_conv2(buffer)  # [bu, _, v, h//2, w//2]

        # Group 2
        buffer = self.Group2_conv1(buffer)
        buffer2 = self.Group2_deconv(buffer)
        buffer2 = buffer2[:, :, 0:-3, :, :]  # [bu, _, v_out, h//2, w//2]
        buffer = self.Group2_conv2(buffer)  # [bu, _, v, h//2, w//4]

        # Layer 3, shrinking
        buffer = self.Layer3(buffer)  # [bu, _, v, h//2, w//4]

        # Group 4, Mapping
        buffer = self.Group4(buffer)  # [bu, _, v, h//2, w//4]

        # Group 5, Attention
        buffer = self.SAAM(buffer)

        # Layer 6, Expanding
        buffer = self.Layer6_conv(buffer)

        # Group 7
        buffer = self.Group7_deconv(buffer)
        buffer = torch.cat([buffer, buffer2], dim=1)
        buffer = self.Group7_conv(buffer)

        # Group 8
        buffer = self.Group8_deconv(buffer)
        buffer = torch.cat([buffer, buffer1], dim=1)
        buffer = self.Group8_conv(buffer)

        # Group 9
        sr = self.Group9_conv(buffer)
        return sr


class SAAM(nn.Module):
    def __init__(self, chn, angRes_in, angRes_out):
        super(SAAM, self).__init__()
        self.angRes_out = angRes_out
        self.q = nn.Conv3d(chn, chn // 8, kernel_size=(1, 1, 1), bias=False)
        self.k = nn.Conv3d(chn, chn // 8, kernel_size=(1, 1, 1), bias=False)
        self.v = nn.Conv3d(chn, chn // 8 * angRes_out, kernel_size=(1, 1, 1), bias=False)

        self.conv3d = nn.Conv3d(chn, chn // 8 * angRes_out, kernel_size=(1, 1, 1), bias=False)
        self.out = nn.Sequential(
            nn.Conv3d(chn // 8 * angRes_in, chn, kernel_size=(1, 1, 7), padding=(0, 0, 3), bias=False),
            nn.ReLU(True),
            nn.Conv3d(chn, chn, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.ReLU(True),
        )


    def forward(self, buffer):
        [bu, c, v, h, w] = buffer.size()
        query = self.q(buffer)
        key = self.k(buffer)
        value = self.v(buffer)

        query = rearrange(query, 'bu c v h w -> bu c h (v w)')
        key = rearrange(key, 'bu c v h w -> bu c h (v w)')
        value = rearrange(value, 'bu c v h w -> bu c h (v w)')

        attn_EPI = torch.einsum('bchm, bchn -> bhmn', query, key)
        attn_EPI = torch.softmax(attn_EPI, dim=-1)

        out = torch.einsum('bhmn, bchn -> bchm', attn_EPI, value)
        out = rearrange(out, 'bu c h (v w) -> bu c v h w', v=v, w=w)

        buffer = self.conv3d(buffer)
        out = out + buffer

        out = rearrange(out, 'bu (c angRes_out) v h w -> bu (c v) angRes_out h w',
                        angRes_out=self.angRes_out)
        out = self.out(out)
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
    args.model_name = 'SAA_Net'
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