import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
import sys
import os
script_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_directory)
from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import warnings

class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SS2D(nn.Module):
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
            dropout=0.1,
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

        self.conv2d_2 = nn.Conv2d(
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
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

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
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        d = self.act(self.conv2d_2(d))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4 + d.contiguous().view(B, -1, H * W) ## Local enhancement
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
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
            drop_rate: float = 0.1,
            d_state: int = 16,
            expand: float = 2.,
            img_size: int = 224,
            patch_size: int = 4,
            embed_dim: int = 64,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)

        self.ss2d = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.ffn = FeedForward(hidden_dim, expand, bias=True)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=embed_dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=embed_dim, norm_layer=None)

        self.conv2d = nn.Conv2d(int(hidden_dim*expand), int(hidden_dim*expand), kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False)

    def forward(self, inputs):
        input, illum = inputs
        input_size = (input.shape[2], input.shape[3])
        input = self.patch_embed(input) 
        input = self.pos_drop(input)
        illum = F.gelu(self.conv2d(illum))
        B, L, C = input.shape
        input = input.view(B, *input_size, C).contiguous()  # [B,H,W,C]
        x = input + self.drop_path(self.ss2d(self.ln_1(input),illum))
        x = x.view(B, -1, C).contiguous()
        x = self.patch_unembed(x, input_size) + self.ffn(self.patch_unembed(self.ln_2(x), input_size),illum)
        return (x,illum)

class FeedForward(nn.Module): ## Implicit Retinex-Aware
    def __init__(self, dim, expand, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*expand)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features, bias=bias)
        self.dwconv3 = nn.Conv2d(hidden_features, 2, kernel_size=3, padding=1, bias=bias)
        self.dwconv4 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.act = nn.Sigmoid()

    def forward(self, x_in,illum):
        x = self.project_in(x_in)
        attn1 = self.dwconv(x) 
        attn2 = self.dwconv2(attn1)
        illum1,illum2 = self.dwconv3(illum).chunk(2, dim=1)
        attn = attn1*self.act(illum1)+attn2*self.act(illum2)
        x = x + attn*x
        x = F.gelu(self.dwconv4(x))
        x = self.project_out(x)
        return x
        
@ARCH_REGISTRY.register()
class MambaLLIE(nn.Module):
    def __init__(self, nf=32,
                img_size=128,
                patch_size=1,
                embed_dim=32,
                depths=(1,2,2,2,2,2),  
                d_state = 32,
                mlp_ratio=2.,
                norm_layer=nn.LayerNorm,
                num_layer=3):
        super(MambaLLIE, self).__init__()

        self.nf = nf
        self.depths = depths

        self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=False)
        self.conv_first_1_fea = nn.Conv2d(5,int(nf*mlp_ratio),3,1,1)
        self.VSSB_1 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf,norm_layer=norm_layer,d_state=d_state,expand=mlp_ratio,img_size=img_size,patch_size=patch_size,embed_dim=embed_dim) for i in range(self.depths[0])])
        
        self.conv_first_2 = nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False)
        self.conv_first_2_fea = nn.Conv2d(5,int(nf*2*mlp_ratio),3,1,1)
        self.VSSB_2 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf*2,norm_layer=norm_layer,d_state=d_state,expand=mlp_ratio,img_size=img_size//2,patch_size=patch_size,embed_dim=embed_dim*2) for i in range(self.depths[1])])
        
        self.conv_first_3 = nn.Conv2d(nf*2, nf * 4, 4, 2, 1, bias=False)
        self.conv_first_3_fea = nn.Conv2d(5,int(nf*4*mlp_ratio),3,1,1)
        self.VSSB_3 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf*4,norm_layer=norm_layer,d_state=d_state,expand=mlp_ratio,img_size=img_size//4,patch_size=patch_size,embed_dim=embed_dim*4) for i in range(self.depths[2])])

        self.conv_first_4 = nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False)
        self.conv_first_4_fea = nn.Conv2d(5,int(nf*4*mlp_ratio),3,1,1)
        self.VSSB_4 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf*4,norm_layer=norm_layer,d_state=d_state,expand=mlp_ratio,img_size=img_size//4,patch_size=patch_size,embed_dim=embed_dim*4) for i in range(self.depths[3])])

        self.upconv1 = nn.ConvTranspose2d(nf*4, nf*4 // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0)
        self.conv_first_5 = nn.Conv2d(nf*4, nf*4 // 2, 3, 1, 1, bias=False)
        self.conv_first_5_fea = nn.Conv2d(5,int(nf*2*mlp_ratio),3,1,1)
        self.VSSB_5 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf*2,norm_layer=norm_layer,d_state=d_state,expand=mlp_ratio,img_size=img_size//2,patch_size=patch_size,embed_dim=embed_dim*2) for i in range(self.depths[4])])
        
        self.upconv2 = nn.ConvTranspose2d(nf*2, nf*2 // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0)
        self.conv_first_6 = nn.Conv2d(nf*2, nf*2 // 2, 3, 1, 1, bias=False)
        self.conv_first_6_fea = nn.Conv2d(5,int(nf*mlp_ratio),3,1,1)
        self.VSSB_6 = nn.Sequential(*[VSSBlock(
            hidden_dim=nf,norm_layer=norm_layer,d_state=d_state,expand=mlp_ratio,img_size=img_size,patch_size=patch_size,embed_dim=embed_dim) for i in range(self.depths[5])])

        self.out_embed = nn.Conv2d(nf, 3, 3, 1, 1)

    def forward(self, x_in):

        x_max = torch.max(x_in, dim=1, keepdim=True)[0]
        x_mean = torch.mean(x_in, dim=1, keepdim=True)
        x_in_cat = torch.cat((x_in,x_max,x_mean), dim=1)

        x_2 = F.avg_pool2d(x_in_cat, kernel_size=2, stride=2)
        x_4 = F.avg_pool2d(x_in_cat, kernel_size=4, stride=4)
        
        x_conv_1 = self.conv_first_1(x_in)
        illum_conv_1 = self.conv_first_1_fea(x_in_cat)
        vssb_fea_1 = self.VSSB_1((x_conv_1,illum_conv_1))[0]
        
        x_conv_2 = self.conv_first_2(vssb_fea_1)
        illum_conv_2 = self.conv_first_2_fea(x_2)
        vssb_fea_2 = self.VSSB_2((x_conv_2,illum_conv_2))[0]

        x_conv_3 = self.conv_first_3(vssb_fea_2)
        illum_conv_3 = self.conv_first_3_fea(x_4)
        vssb_fea_3 = self.VSSB_3((x_conv_3,illum_conv_3))[0]

        x_conv_4 = self.conv_first_4(vssb_fea_3)
        illum_conv_4 = self.conv_first_4_fea(x_4)
        vssb_fea_4 = self.VSSB_4((x_conv_4,illum_conv_4))[0]

        up_feat_1 = self.upconv1(vssb_fea_4)
        x_cat_1 = torch.cat([up_feat_1, vssb_fea_2], dim=1)
        vssb_fea_5 = self.conv_first_5(x_cat_1)
        illum_conv_5 = self.conv_first_5_fea(x_2)
        vssb_fea_5 = self.VSSB_5((vssb_fea_5,illum_conv_5))[0]

        up_feat_2 = self.upconv2(vssb_fea_5)
        x_cat_2 = torch.cat([up_feat_2, vssb_fea_1], dim=1)
        vssb_fea_6 = self.conv_first_6(x_cat_2)
        illum_conv_6 = self.conv_first_6_fea(x_in_cat)
        vssb_fea_6 = self.VSSB_6((vssb_fea_6,illum_conv_6))[0]

        out = self.out_embed(vssb_fea_6) + x_in

        return out

