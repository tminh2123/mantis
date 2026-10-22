# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_


# =====================================================
# KAN Linear Layer
# =====================================================
class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Linear layer with B-spline expansion.
    Combines a standard linear projection with learnable spline basis functions.
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.grid_eps = grid_eps

        # Grid definition
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        ).expand(in_features, -1)
        self.register_buffer("grid", grid.contiguous())

        # Weights
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order], noise
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions.
        Args:
            x: (B, in_features)
        Returns:
            bases: (B, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)

        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute spline coefficients via least squares fitting.
        Args:
            x: (N, in_features)
            y: (N, in_features, out_features)
        Returns:
            coeff: (out_features, in_features, grid_size + spline_order)
        """
        A = self.b_splines(x).transpose(0, 1)   # (in, N, coeff)
        B = y.transpose(0, 1)                   # (in, N, out)
        sol = torch.linalg.lstsq(A, B).solution # (in, coeff, out)
        return sol.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_out = F.linear(self.base_activation(x), self.base_weight)
        spline_out = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_out + spline_out

    def regularization_loss(self, reg_act=1.0, reg_entropy=1.0) -> torch.Tensor:
        """Simple surrogate regularization on spline weights"""
        l1 = self.spline_weight.abs().mean(-1)  # (out, in)
        act_loss = l1.sum()
        p = l1 / (act_loss + 1e-8)
        entropy_loss = -torch.sum(p * p.log())
        return reg_act * act_loss + reg_entropy * entropy_loss


# =====================================================
# KAN Wrapper
# =====================================================
class KAN(nn.Module):
    """Stack of KANLinear layers"""
    def __init__(self, layers_hidden, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            KANLinear(in_f, out_f, **kwargs)
            for in_f, out_f in zip(layers_hidden, layers_hidden[1:])
        )

    def forward(self, x, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, reg_act=1.0, reg_entropy=1.0):
        return sum(layer.regularization_loss(reg_act, reg_entropy) for layer in self.layers)


# =====================================================
# KAN MLP Layer
# =====================================================
class KANLayer(nn.Module):
    """MLP-like block with KANLinear + Depthwise Convs"""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if not no_kan:
            self.fc1 = KANLinear(in_features, hidden_features)
            self.fc2 = KANLinear(hidden_features, out_features)
            self.fc3 = KANLinear(hidden_features, out_features)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.view(B * N, C)).view(B, N, -1)
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.view(B * N, -1)).view(B, N, -1)
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.view(B * N, -1)).view(B, N, -1)
        x = self.dwconv_3(x, H, W)
        return x


# =====================================================
# KAN Block (Residual)
# =====================================================
class KANBlock(nn.Module):
    """Residual block with KANLayer"""
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.norm = norm_layer(dim)
        self.layer = KANLayer(dim, hidden_features=dim, act_layer=act_layer, drop=drop, no_kan=no_kan)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        return x + self.drop_path(self.layer(self.norm(x), H, W))


# =====================================================
# Depthwise Convolutions
# =====================================================
class DWConv(nn.Module):
    """Standard depthwise convolution for (B, N, C) tensors"""
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                groups=dim, bias=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class DW_bn_relu(nn.Module):
    """Depthwise conv + BN + ReLU"""
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                groups=dim, bias=True)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.relu(self.bn(self.dwconv(x)))
        return x.flatten(2).transpose(1, 2)


# =====================================================
# Patch Embedding
# =====================================================
class PatchEmbed(nn.Module):
    """
    Convert image to patch embeddings.
    Args:
        img_size: image resolution (int or tuple)
        patch_size: patch size (int or tuple)
        stride: stride for conv projection
        in_chans: number of input channels
        embed_dim: embedding dimension
    """
    def __init__(self, img_size=224, patch_size=7, stride=4,
                 in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)                          # (B, embed_dim, H', W')
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)          # (B, N, embed_dim)
        x = self.norm(x)
        return x, H, W


# =====================================================
# Convolutional Blocks
# =====================================================
class ConvLayer(nn.Module):
    """Conv-BN-ReLU ×2 block"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class D_ConvLayer(nn.Module):
    """Variant: first conv keeps channels, second conv changes them"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

