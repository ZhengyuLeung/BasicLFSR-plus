import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
import math
from utils.utils import LF_rgb2ycbcr, LF_ycbcr2rgb


class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        channels = 64
        self.channels = channels
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.factor = args.scale_factor
        self.task = args.task

        ##################### Initial Convolution #####################
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        ################ Alternate AngTrans & SpaTrans ################
        self.altblock = nn.Sequential(
            AltFilter(self.angRes_in, self.channels),
            AltFilter(self.angRes_in, self.channels),
            AltFilter(self.angRes_in, self.channels),
            AltFilter(self.angRes_in, self.channels),
            AltFilter(self.angRes_in, self.channels),
        )

        ####################### UP Sampling ###########################
        if args.task == 'SSR':
            self.upsampling = nn.Sequential(
                nn.Conv2d(channels, channels * self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
                nn.PixelShuffle(self.factor),
                nn.LeakyReLU(0.2),
                nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
            )
        elif args.task == 'ASR':
            self.upsampling = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=self.angRes_in, stride=self.angRes_in, padding=0, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, channels * self.angRes_out ** 2, kernel_size=1, padding=0, bias=False),
                nn.PixelShuffle(self.angRes_out),
                nn.Conv2d(channels, 1, kernel_size=3, dilation=self.angRes_out, padding=self.angRes_out, bias=False)
            )

    def forward(self, lr, info=None):
        lr = LF_rgb2ycbcr(lr)[:, 0:1, :, :, :, :]
        # Initial Convolution
        buffer = rearrange(lr, 'b c u v h w -> b c (u v) h w', u=self.angRes_in, v=self.angRes_in)
        buffer = self.conv_init0(buffer)
        buffer = self.conv_init(buffer) + buffer  # [B, C, A^2, h, w]

        # Alternate AngTrans & SpaTrans
        buffer = self.altblock(buffer) + buffer

        # Up-Sampling
        if self.task == 'SSR':
            # Bicubic
            lr_upscale = interpolate(lr, self.angRes_in, scale_factor=self.factor, mode='bicubic')
            buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) (v w)', u=self.angRes_in, v=self.angRes_in)
            out = {}
            out['SR'] = self.upsampling(buffer) + lr_upscale
        elif self.task == 'ASR':
            buffer = rearrange(buffer, 'b c (u v) h w -> b c u v h w', u=self.angRes_in, v=self.angRes_in)
            buffer = rearrange(buffer, 'b c u v h w -> b c (h u) (w v)')
            buffer = self.upsampling(buffer)
            out = {}
            out['SR'] = rearrange(buffer, 'b c (h u) (w v) -> b c u v h w', u=self.angRes_out, v=self.angRes_out)

        return out


class MultiHeadSelfAtten(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(MultiHeadSelfAtten, self).__init__()
        self.num_heads = num_heads
        self.qk_ff = nn.Linear(embed_dim, 2*embed_dim, bias=False)
        self.v_ff = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_ff = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.kaiming_uniform_(self.qk_ff.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_ff.weight, a=math.sqrt(5))

        self.norm = nn.LayerNorm(embed_dim)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, attn_mask):
        token_norm = self.norm(token)
        qk = self.qk_ff(token_norm)
        qk = rearrange(qk, 'tgt_len B (Nh c) -> (B Nh) tgt_len c', Nh=self.num_heads)
        qk = qk / math.sqrt(qk.size(2)//2)
        [query, key] = torch.chunk(qk, 2, dim=2)

        v = self.v_ff(token)
        v = rearrange(v, 'tgt_len B (Nh c) -> (B Nh) tgt_len c', Nh=self.num_heads)
        value = v / math.sqrt(v.size(2))

        # (B, Nt, E) x (B, Ns, E) -> (B, Nt, Ns)
        attn = torch.einsum('b t c, b s c -> b t s', query, key) + attn_mask
        attn = self.dropout(self.softmax(attn))
        attn_output_weights = rearrange(attn, '(B Nh) Nt Ns -> B Nh Nt Ns', Nh=self.num_heads).mean(dim=1)

        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.einsum('b t s, b s c -> b t c', attn, value)
        output = rearrange(output, '(B Nh) tgt_len c -> tgt_len B (Nh c)', Nh=self.num_heads)
        output = self.out_ff(output)

        return output, attn_output_weights


class SpaTrans(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8, dropout=0.):
        super(SpaTrans, self).__init__()
        self.spa_dim = spa_dim
        self.linear_in = nn.Linear(channels, self.spa_dim, bias=False)

        self.norm = nn.LayerNorm(self.spa_dim)
        self.attention = nn.MultiheadAttention(self.spa_dim, num_heads, dropout, bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None

        # self.attention = MultiHeadSelfAtten(self.spa_dim, num_heads, dropout)

        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.spa_dim),
            nn.Linear(self.spa_dim, self.spa_dim*2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.spa_dim*2, self.spa_dim, bias=False),
            nn.Dropout(dropout)
        )
        self.linear_out = nn.Linear(self.spa_dim, channels, bias=False)

    def gen_mask(self, h:int, w:int, k_h:int, k_w:int):
        attn_mask = torch.zeros([h, w, h, w])
        k_h_left = k_h//2
        k_h_right = k_h - k_h_left
        k_w_left = k_w//2
        k_w_right = k_w - k_w_left
        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[max(0, i-k_h_left):min(h, i+k_h_right), max(0, j-k_w_left):min(w,j+k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, buffer):
        [_, _, a, h, w] = buffer.size()
        attn_mask = self.gen_mask(h, w, self.kernel_search[0], self.kernel_search[1]).to(buffer.device)

        spa_token = rearrange(buffer, 'b c a h w -> (h w) (b a) c')
        spa_token = self.linear_in(spa_token)

        spa_token_norm = self.norm(spa_token)
        spa_token = self.attention(query=spa_token_norm,
                                   key=spa_token_norm,
                                   value=spa_token,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + spa_token

        # spa_token = self.attention(spa_token, attn_mask)[0] + spa_token
        spa_token = self.feed_forward(spa_token) + spa_token
        spa_token = self.linear_out(spa_token)
        buffer = rearrange(spa_token, '(h w) (b a) c -> b c a h w', h=h, w=w, a=a)

        return buffer


class AltFilter(nn.Module):
    def __init__(self, angRes, channels):
        super(AltFilter, self).__init__()
        self.angRes = angRes
        self.epi_trans = SpaTrans(channels, channels*2)
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        )

    def forward(self, buffer):
        shortcut = buffer
        [_, _, _, h, w] = buffer.size()
        self.epi_trans.kernel_search = [10, 21]

        # # Horizon
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (v w) u h', u=self.angRes, v=self.angRes)
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, 'b c (v w) u h -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer = self.conv(buffer) + shortcut

        # Vertical
        buffer = rearrange(buffer, 'b c (u v) h w -> b c (u h) v w', u=self.angRes, v=self.angRes)
        buffer = self.epi_trans(buffer)
        buffer = rearrange(buffer, 'b c (u h) v w -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer = self.conv(buffer) + shortcut
        return buffer


def interpolate(x, angRes, scale_factor, mode):
    x = rearrange(x, 'b c (u h) (v w) -> (b u v) c h w', u=angRes, v=angRes)
    x_upscale = F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)
    x_upscale = rearrange(x_upscale, '(b u v) c h w -> b c (u h) (v w)', u=angRes, v=angRes)
    return x_upscale


class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR, info=None):
        loss = self.criterion_Loss(SR['SR'], HR)

        return loss


def weights_init(m):
    pass
