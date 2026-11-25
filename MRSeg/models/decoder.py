"""
Multi-scale decoder for MRSeg with gated skip connections.
"""
import torch
import torch.nn as nn


class GatedFuse(nn.Module):
    """
    Gated skip-connection fusion.

    gate = tanh(gate_conv([up_feat, skip_feat]))
    out  = up_feat + skip_feat + skip_feat * gate
    """

    def __init__(self, up_channels: int = 192, skip_channels: int = 192):
        super().__init__()
        self.skip_proj = (
            nn.Conv2d(skip_channels, up_channels, 1)
            if skip_channels != up_channels
            else nn.Identity()
        )
        self.gate_conv = nn.Conv2d(up_channels * 2, up_channels, 1)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)

    def forward(self, x_up: torch.Tensor, f_skip: torch.Tensor) -> torch.Tensor:
        f = self.skip_proj(f_skip)
        gate = torch.tanh(self.gate_conv(torch.cat([x_up, f], dim=1)))
        f_tilde = f + f * gate
        return x_up + f_tilde


class MRSegDecoder(nn.Module):
    """
    Hierarchical decoder:

    f4'' (768, H/32) → proj4 → up → + proj3(f3'')   → x1 (192, H/16)
    x1               → up → gated_fuse(f2 192)       → x2 (192, H/8)
    x2               → up → gated_fuse(proj1(f1) 96) → x3 (192, H/4)
    x3               → up → + proj_skip1(skip1 32)   → x4 (192, H/2)
    x4               → up → + proj_skip2(skip2 16)   → x5 (192, H)
    x5               → coarse_head + detail_head      → pred (1, H)
    """

    def __init__(self, dim: int = 192):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.proj4 = nn.Conv2d(768, dim, 1)
        self.proj3 = nn.Conv2d(384, dim, 1)
        self.proj1 = nn.Conv2d(96, dim, 1)
        self.proj_skip1 = nn.Conv2d(32, dim, 1)
        self.proj_skip2 = nn.Conv2d(16, dim, 1)

        # f2 is already 192 channels (matches decoder dim); no proj needed
        self.gated2 = GatedFuse(dim, dim)
        self.gated1 = GatedFuse(dim, dim)

        # Prediction heads
        self.coarse_head = nn.Conv2d(dim, 1, 1)
        self.detail_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.detail_head = nn.Conv2d(dim, 1, 1)

    def forward(
        self,
        f1: torch.Tensor,    # (B, 96,  H/4,  W/4)
        f2: torch.Tensor,    # (B, 192, H/8,  W/8)
        f3pp: torch.Tensor,  # (B, 384, H/16, W/16)
        f4pp: torch.Tensor,  # (B, 768, H/32, W/32)
        skip1: torch.Tensor, # (B, 32,  H/2,  W/2)
        skip2: torch.Tensor, # (B, 16,  H,    W)
    ) -> torch.Tensor:
        x1 = self.up(self.proj4(f4pp)) + self.proj3(f3pp)   # (B, 192, H/16)
        x2 = self.gated2(self.up(x1), f2)                   # (B, 192, H/8)
        x3 = self.gated1(self.up(x2), self.proj1(f1))       # (B, 192, H/4)
        x4 = self.up(x3) + self.proj_skip1(skip1)           # (B, 192, H/2)
        x5 = self.up(x4) + self.proj_skip2(skip2)           # (B, 192, H)

        coarse = self.coarse_head(x5)
        detail = self.detail_head(self.detail_dw(x5))
        return coarse + detail   # (B, 1, H, W)
