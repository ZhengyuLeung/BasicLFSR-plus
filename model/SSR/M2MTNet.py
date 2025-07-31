import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Callable, Tuple, TypeVar
from utils.utils import LF_rgb2ycbcr, LF_ycbcr2rgb, LF_interpolate
T = TypeVar('T')
_spatial = Tuple[T, T]


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.scale = args.scale_factor
        self.sz_a = args.angRes_in
        self.patch_size = 32

        self.n_chns_in = 1
        self.n_chns_ft = 48
        self.embed_dim = 128
        self.n_block = 8

        self.head = SpatialConv(in_channels=self.n_chns_in, out_channels=self.n_chns_ft, n=4, bias=False)
        self.body = []
        for _ in range(self.n_block):
            block = CorrBlock(chns_in=self.n_chns_ft, embed_dim=self.embed_dim, patch_size=self.patch_size)
            self.body.append(block)
        self.body = nn.ModuleList(self.body)

        self.tail = [
            nn.Conv2d(self.n_chns_ft, self.n_chns_ft * self.scale ** 2, kernel_size=1, padding=0, bias=False),
            nn.PixelShuffle(self.scale),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.n_chns_ft, self.n_chns_in, kernel_size=3, padding=1, bias=False),
        ]
        self.tail = nn.Sequential(*self.tail)

    def forward(self, lr, info=None):
        [b, c, u, v, h, w] = lr.size()

        # rgb2ycbcr
        lr_ycbcr = LF_rgb2ycbcr(lr)
        sr_ycbcr = LF_interpolate(lr_ycbcr, scale_factor=self.scale, mode='bicubic')
        inp = (lr_ycbcr[:, 0:1, :, :, :, :])

        x = self.head(inp)
        res = x
        for block in self.body:
            res = res + block(res)
        x = x + res

        x = rearrange(x, "b c u v h w -> b c (u h) (v w)")
        x = self.tail(x)
        x = rearrange(x, 'b c (u h) (v w) -> b c u v h w', u=u, v=v)

        # ycbcr2rgb
        sr_ycbcr[:, 0:1, :, :, :, :] += x
        out = {}
        out['SR'] = LF_ycbcr2rgb(sr_ycbcr)

        return out


class M2MTAttention(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, factor: int, patch_size, heads: int = 1,
                 sz_a: Tuple[int, int] = (5, 5)) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.sz_a = sz_a
        self.factor = factor
        in_channels_ = in_channels * sz_a[0] * sz_a[1]
        self.embed_dim = embed_dim

        self.norm = nn.LayerNorm(embed_dim)
        self.ff_in = nn.Linear(in_channels_, embed_dim, bias=True)
        nn.init.kaiming_uniform_(self.ff_in.weight, a=math.sqrt(5))
        self.qkv = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=True),
            nn.Linear(embed_dim, embed_dim // factor**2, bias=True),
            nn.Linear(embed_dim, embed_dim // factor**2, bias=True)
        ])
        for w in self.qkv:
            nn.init.kaiming_uniform_(w.weight, a=math.sqrt(5))
        self.ff_out = nn.Linear(embed_dim, in_channels_, bias=True)
        nn.init.kaiming_uniform_(self.ff_out.weight, a=math.sqrt(5))

        self.heads = heads
        self.scale = (in_channels_ // heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp: torch.Tensor):
        B, C, N, H, W = inp.shape
        x = inp
        x = rearrange(x, 'b c n h w -> b (h w) (c n)')
        x = self.ff_in(x)
        x_norm = self.norm(x)
        q = self.qkv[0](x_norm)
        k = self.qkv[1](x_norm)
        v = self.qkv[2](x)
        q = rearrange(q, 'b hw1 (c1 head) -> b head hw1 c1', head=self.heads)
        k = rearrange(k, 'b (hw2 f) (c2 head) -> b head hw2 (c2 f)', head=self.heads, f=self.factor**2)
        v = rearrange(v, 'b (hw2 f) (c2 head) -> b head hw2 (c2 f)', head=self.heads, f=self.factor**2)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        x = attn @ v
        x = self.ff_out(x)
        x = rearrange(x, 'b head (h w) (c n) -> b (head c) n h w', n=N, h=H, w=W)
        return x


class M2MT(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=(32, 32), heads=1):
        super(M2MT, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attention = M2MTAttention(in_channels=in_channels, embed_dim=embed_dim, factor=1,
                                       patch_size=patch_size, heads=heads)
        self.sz_a = (5, 5)

    def forward(self, inp):
        buffer = rearrange(inp, 'b c u v h w -> b c (u v) h w', u=self.sz_a[0], v=self.sz_a[1])
        buffer = self.conv(buffer) + buffer

        buffer = self.attention(buffer) + buffer

        buffer = rearrange(buffer, 'b c (u v) h w -> b c u v h w', u=self.sz_a[0], v=self.sz_a[1])
        return buffer


class CorrBlock(nn.Module):
    def __init__(self, chns_in: int, embed_dim: int, patch_size) -> None:
        super(CorrBlock, self).__init__()
        self.body = [
            M2MT(in_channels=chns_in, embed_dim=embed_dim, patch_size=patch_size, heads=1),
            DTransformer(in_channels=chns_in, connection="uv")
        ]
        self.body = nn.Sequential(*self.body)

    def forward(self, x):
        x = self.body(x)
        return x


class DConv(nn.Module):
    available_dims = ['u', 'v', 'h', 'w']
    all_dims = ['b', 'c'] + available_dims

    def __init__(self,
                 in_channels: int, out_channels: int, connection: str,
                 kernel_size: _spatial[int] = (3, 3),
                 stride: _spatial[int] = (1, 1),
                 padding: _spatial[int] = (1, 1),
                 n: int = 1,
                 act: Callable[[], nn.Module] = lambda: nn.LeakyReLU(0.2, inplace=True),
                 *args, **kwargs):
        super(DConv, self).__init__()
        assert all(dim in self.available_dims for dim in connection)
        self.connection = connection
        self.dims_preserve, self.dims_active = [], []
        for dim in self.available_dims:
            _dst = self.dims_active if dim in connection else self.dims_preserve
            _dst.append(dim)
        self.pattern0 = f"{' '.join(['b', 'c'] + self.available_dims)}"
        self.pattern1 = f"({' '.join(['b'] + self.dims_preserve)}) {' '.join(['c'] + self.dims_active)}"

        self.body = []
        for i in range(n):
            self.body.append(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                          *args, **kwargs))
            if act:
                self.body.append(act())
        self.body = nn.Sequential(*self.body)

    def forward(self, inp: torch.Tensor):
        szs_preserve = {dim: sz for dim, sz in zip(self.all_dims, inp.shape) if dim in self.dims_preserve}
        x = rearrange(inp, f'{self.pattern0} -> {self.pattern1}')
        x = self.body(x)
        x = rearrange(x, f'{self.pattern1} -> {self.pattern0}', **szs_preserve)

        return x


def SpatialConv(*args, **kwargs) -> DConv:
    return DConv(connection='hw', *args, **kwargs)


def AngularConv(*args, **kwargs) -> DConv:
    return DConv(connection='uv', *args, **kwargs)


def lf2token(lf: torch.Tensor) -> torch.Tensor:
    token = rearrange(lf, 'b c d d3 d4 -> d (b d3 d4) c')
    return token


def token2lf(token: torch.Tensor, d1: int, d2: int, d3: int, d4: int) -> torch.Tensor:
    lf = rearrange(token, '(d3 d4) (b d1 d2) (c) -> b c d1 d2 d3 d4', d1=d1, d2=d2, d3=d3, d4=d4)
    return lf


class DTransformer(nn.Module):
    available_dims = ['u', 'v', 'h', 'w']
    all_dims = ['b', 'c'] + available_dims

    def __init__(self, in_channels: int, connection: str, kernel_sz: int = 3):
        super(DTransformer, self).__init__()
        assert all(dim in self.available_dims for dim in connection)
        self.connection = connection
        self.kernel_sz = kernel_sz
        self.dims_preserve, self.dims_active = [], []
        for dim in self.available_dims:
            _dst = self.dims_active if dim in connection else self.dims_preserve
            _dst.append(dim)
        self.pattern0 = f"{' '.join(['b', 'c'] + self.available_dims)}"
        self.pattern1 = f"{' '.join(['b', 'c'] + self.dims_preserve + self.dims_active)}"

        # if "w" in self.dims_active or "h" in self.dims_active:
        self.linear_in = nn.Linear(in_channels, in_channels, bias=True)
        self.norm = nn.LayerNorm(in_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=8,
            bias=True
        )
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, in_channels * 2, bias=True),
            nn.ReLU(True),
            nn.Linear(in_channels * 2, in_channels, bias=True),
        )
        self.linear_out = nn.Linear(in_channels, in_channels, bias=True)
        self.conv = SpatialConv(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                n=3, bias=True)

    def forward(self, inp: torch.Tensor):
        szs_preserve = {dim: sz for dim, sz in zip(self.all_dims, inp.shape) if dim in self.dims_preserve}
        x = rearrange(inp, f'{self.pattern0} -> {self.pattern1}')
        _, c, d1, d2, d3, d4 = x.shape

        x = rearrange(x, 'b c d1 d2 d3 d4 -> b c (d3 d4) d1 d2')
        token = lf2token(x)
        token = self.linear_in(token)

        token_norm = self.norm(token)
        token = self.attention(query=token_norm,
                               key=token_norm,
                               value=token,
                               need_weights=False)[0] + token
        token = self.feed_forward(token) + token
        token = self.linear_out(token)

        x = token2lf(token, d1=d1, d2=d2, d3=d3, d4=d4)
        x = rearrange(x, f'{self.pattern1} -> {self.pattern0}', **szs_preserve)
        x = self.conv(x) + x
        return x


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, out, HR, degrade_info=None):
        loss = self.criterion_Loss(out['SR'], HR)

        return loss


def weights_init(m):
    pass
