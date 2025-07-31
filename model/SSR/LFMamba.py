import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from einops import rearrange, repeat
from typing import Optional, Callable, Any
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from functools import partial
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.vision_transformer import _load_weights
from utils.utils import LF_rgb2ycbcr, LF_ycbcr2rgb, LF_interpolate

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channel = 64
        self.channels = channel
        self.angRes = args.angRes_in
        self.scale = args.scale_factor
        depth = 2
        #################### Initial Feature Extraction #####################
        self.conv_init0 = nn.Sequential(nn.Conv3d(1, channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False))
        self.conv_init = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ############# Deep Spatial-Angular Correlation Learning #############
        self.altblock = nn.Sequential(
            SpaAngFilter(dim=channel, angRes=self.angRes, depth=depth),
            SpaAngFilter(dim=channel, angRes=self.angRes, depth=depth),
            SpaAngFilter(dim=channel, angRes=self.angRes, depth=depth),
        )
        self.epiblock = nn.Sequential(
            EPISSM(dim=channel, angRes=self.angRes, depth=depth),
            EPISSM(dim=channel, angRes=self.angRes, depth=depth),
            EPISSM(dim=channel, angRes=self.angRes, depth=depth),
        )

        ########################### UP-Sampling #############################
        self.upsampling = nn.Sequential(
            nn.Conv2d(channel * 3, channel * self.scale ** 2, kernel_size=1, padding=0, bias=False),
            nn.PixelShuffle(self.scale),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel, 1, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, lr, info=None):
        [b, c, u, v, h, w] = lr.size()

        # rgb2ycbcr
        lr_ycbcr = LF_rgb2ycbcr(lr)
        sr_ycbcr = LF_interpolate(lr_ycbcr, scale_factor=self.scale, mode='bicubic')
        lr = lr_ycbcr[:, 0:1, :, :, :, :]

        # Initial Feature Extraction
        x = rearrange(lr, 'b c u v h w -> b c (u v) h w')
        buffer = self.conv_init0(x)
        buffer_init = self.conv_init(buffer) + buffer

        for m in self.modules():
            m.h = lr.size(-2)
            m.w = lr.size(-1)

        # Deep Spatial-Angular Correlation Learning
        buffer_1 = self.altblock(buffer_init) + buffer_init
        buffer_2 = self.epiblock(buffer_1) + buffer_1
        buffer = torch.cat([buffer_init, buffer_1, buffer_2], dim=1)

        # UP-Sampling
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) (v w)', u=u, v=v)
        y = self.upsampling(buffer)

        y = rearrange(y, 'b c (u h) (v w) -> b c u v h w', u=u, v=v)
        sr_ycbcr[:, 0:1, :, :, :, :] += y

        out = {}
        out['SR'] = LF_ycbcr2rgb(sr_ycbcr)

        return out


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr:  # a larger compression ratio is used for light-SR
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class ESS2D(nn.Module):
    """
        Efficient SS2D
        Keep K = 1, which means the learnable parameters Bs, Cs are trained across the whole channel
        x: [B, K, D, L]
     """

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1

        x_hw = x[:, :C // 4].view(B, -1, L)
        x_wh = x[:, C // 4:C // 2].permute(0, 1, 3, 2).contiguous().view(B, -1, L)
        x_hw_r = x[:, C // 2:3 * C // 4].view(B, -1, L).flip(-1)
        x_wh_r = x[:, 3 * C // 4:].permute(0, 1, 3, 2).contiguous().view(B, -1, L).flip(-1)

        xs = torch.cat([x_hw, x_wh, x_hw_r, x_wh_r], dim=1).view(B, 1, -1, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        y_hw = out_y[:, 0, :C // 4].view(B, -1, H, W)
        y_wh = out_y[:, 0, C // 4:C // 2].view(B, -1, W, H).permute(0, 1, 3, 2).contiguous()
        y_hw_r = out_y[:, 0, C // 2:3 * C // 4].flip(-1).view(B, -1, H, W)
        y_wh_r = out_y[:, 0, 3 * C // 4:].flip(-1).view(B, -1, W, H).permute(0, 1, 3, 2).contiguous()

        y = torch.cat([y_hw, y_wh, y_hw_r, y_wh_r], dim=1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        assert y.dtype == torch.float32
        y = y.permute(0, 2, 3, 1)  # [b, h, w, c]
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = ESS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate,
                                    **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3,
                                                                                                        1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class BasicLayer(nn.Module):
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 is_light_sr=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                input_resolution=input_resolution, is_light_sr=is_light_sr))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        # x: [b, l, c]
        h, w = x_size
        x = rearrange(x, 'b c h w -> b (h w) c')
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class ResidualGroup(nn.Module):
    """Residual State Space Group (RSSG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 angRes,
                 dim,
                 input_resolution,
                 depth,
                 d_state=16,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 resi_connection='1conv',
                 is_light_sr=False):
        super(ResidualGroup, self).__init__()
        self.angRes = angRes
        self.dim = dim
        self.input_resolution = input_resolution  # no use
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state=d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr=is_light_sr)

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x):

        b, c, h, w = x.shape
        x_size = [h, w]
        x = self.conv(self.residual_group(x, x_size)) + x

        return x


class SpaSSM(nn.Module):
    def __init__(self, dim, angRes, depth):
        super().__init__()
        self.angRes = angRes
        self.layer = ResidualGroup(angRes=angRes, dim=dim, input_resolution=(16, 16), depth=depth)

    def forward(self, x):
        # x = x + self.spa_position
        x = rearrange(x, 'b c a h w -> (b a) c h w')
        x = self.layer(x)
        x = rearrange(x, '(b a) c h w -> b c a h w', a=self.angRes ** 2)
        return x


class AngSSM(nn.Module):
    def __init__(self, dim, angRes, depth):
        super().__init__()
        self.angRes = angRes
        self.layer = ResidualGroup(angRes=angRes, dim=dim, input_resolution=(5, 5), depth=depth)

    def forward(self, x):
        # x = x + self.ang_position
        x = rearrange(x, 'b c (u v) h w -> (b h w) c u v', u=self.angRes)
        x = self.layer(x)
        x = rearrange(x, '(b h w) c u v -> b c (u v) h w', h=self.h, w=self.w)
        return x


class EPISSM(nn.Module):
    def __init__(self, dim, angRes, depth):
        super().__init__()
        self.angRes = angRes
        self.layer = ResidualGroup(angRes=angRes, dim=dim, input_resolution=(5, 5), depth=depth)

    def forward(self, x):
        x = rearrange(x, 'b c (u v) h w -> (b v w) c u h', u=self.angRes)
        x = self.layer(x)
        x = rearrange(x, '(b v w) c u h -> (b u h) c v w', v=self.angRes, w=self.w)
        x = self.layer(x)
        x = rearrange(x, '(b u h) c v w -> b c (u v) h w', u=self.angRes, h=self.h)
        return x


class SpaAngFilter(nn.Module):
    def __init__(self, dim, angRes, depth):
        super().__init__()
        self.angRes = angRes
        self.spa_block = SpaSSM(angRes=angRes, dim=dim, depth=depth)
        self.ang_block = AngSSM(angRes=angRes, dim=dim, depth=depth)

    def forward(self, x):
        x = self.spa_block(x)
        x = self.ang_block(x)
        return x


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, degrade_info=None):
        loss = self.criterion_Loss(out, HR)

        return loss


def weights_init(m):
    pass

