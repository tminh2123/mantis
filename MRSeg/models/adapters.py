"""
Adapter modules for MRSeg.

MedImageAdapter — grayscale→RGB adapter with stem skips
Router          — TopK=2 sparse routing over M=4 adapters
PairAdapter     — M=4 parallel low-rank adapters for f3, f4, text
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedImageAdapter(nn.Module):
    """
    Converts 1-channel grayscale to I3 (3-channel) via a lightweight residual path.
    Also produces two stem feature maps for decoder skip connections.

    Returns: i3 (B,3,H,W), skip1 (B,32,H/2,W/2), skip2 (B,16,H,W)
    """

    def __init__(self, adapter_channels: int = 64):
        super().__init__()
        C = adapter_channels

        # Grayscale → 3-channel residual path
        self.a_conv1 = nn.Conv2d(1, C, 1)
        self.a_dw = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.a_norm = nn.GroupNorm(1, C)       # ≈ InstanceNorm
        self.a_act = nn.GELU()
        self.a_conv2 = nn.Conv2d(C, 3, 1)
        nn.init.zeros_(self.a_conv2.weight)
        nn.init.zeros_(self.a_conv2.bias)

        # Stem branches for skip connections
        self.stem_half = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )
        self.stem_full = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )

    def forward(self, x_gray: torch.Tensor):
        i_base = x_gray.repeat(1, 3, 1, 1)
        a_out = self.a_conv2(self.a_act(self.a_norm(self.a_dw(self.a_conv1(x_gray)))))
        i3 = i_base + a_out
        skip1 = self.stem_half(i3)   # (B, 32, H/2, W/2)
        skip2 = self.stem_full(i3)   # (B, 16, H, W)
        return i3, skip1, skip2


class Router(nn.Module):
    """
    Produces sparse routing weights alpha (B, M) with TopK=2 normalization.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 768,
        num_adapters: int = 4,
        topk: int = 2,
    ):
        super().__init__()
        self.topk = topk
        self.num_adapters = num_adapters

        # Single linear: (768+768) → 4 = ~6K params
        self.gate = nn.Linear(visual_dim + text_dim, num_adapters)

    def forward(self, f4: torch.Tensor, z_text: torch.Tensor) -> torch.Tensor:
        """
        f4:     (B, 768, H, W)
        z_text: (B, L, 768)
        Returns alpha: (B, M) sparse, sums to 1 within kept entries
        """
        v_pool = F.adaptive_avg_pool2d(f4, 1).flatten(1)   # (B, 768)
        t_mean = z_text.mean(1)                             # (B, 768)
        r = self.gate(torch.cat([v_pool, t_mean], dim=1))  # (B, M)
        s = F.softmax(r, dim=1)                            # (B, M)

        # TopKNorm: keep top-k, zero the rest, renormalize
        topk_vals, topk_idx = s.topk(self.topk, dim=1)
        alpha = torch.zeros_like(s)
        alpha.scatter_(1, topk_idx, topk_vals)
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)

        return alpha   # (B, M)


class PairAdapter(nn.Module):
    """
    M=4 parallel low-rank adapters applied to f3, f4, and text jointly.
    Each adapter: Linear A → DW-Conv (visual) → Linear B with scale=eta/rank.
    Zero-init on all B matrices so adapters start as identity.
    """

    def __init__(
        self,
        f3_channels: int = 384,
        f4_channels: int = 768,
        text_dim: int = 768,
        num_adapters: int = 4,
        rank: int = 8,
        eta: float = 16.0,
    ):
        super().__init__()
        self.M = num_adapters
        self.rank = rank
        self.eta = eta

        # --- f3 adapter bank ---
        self.f3_A = nn.ModuleList([nn.Linear(f3_channels, rank, bias=False) for _ in range(num_adapters)])
        self.f3_dw = nn.ModuleList([nn.Conv2d(rank, rank, 3, padding=1, groups=rank) for _ in range(num_adapters)])
        self.f3_B = nn.ModuleList([nn.Linear(rank, f3_channels, bias=False) for _ in range(num_adapters)])
        self.f3_ln = nn.LayerNorm(f3_channels)
        for m in range(num_adapters):
            nn.init.kaiming_uniform_(self.f3_A[m].weight, a=5 ** 0.5)
            nn.init.zeros_(self.f3_B[m].weight)

        # --- f4 adapter bank ---
        self.f4_A = nn.ModuleList([nn.Linear(f4_channels, rank, bias=False) for _ in range(num_adapters)])
        self.f4_dw = nn.ModuleList([nn.Conv2d(rank, rank, 3, padding=1, groups=rank) for _ in range(num_adapters)])
        self.f4_B = nn.ModuleList([nn.Linear(rank, f4_channels, bias=False) for _ in range(num_adapters)])
        self.f4_ln = nn.LayerNorm(f4_channels)
        for m in range(num_adapters):
            nn.init.kaiming_uniform_(self.f4_A[m].weight, a=5 ** 0.5)
            nn.init.zeros_(self.f4_B[m].weight)

        # --- text adapter bank ---
        self.t_A = nn.ModuleList([nn.Linear(text_dim, rank, bias=False) for _ in range(num_adapters)])
        self.t_B = nn.ModuleList([nn.Linear(rank, text_dim, bias=False) for _ in range(num_adapters)])
        self.t_ln = nn.LayerNorm(text_dim)
        for m in range(num_adapters):
            nn.init.kaiming_uniform_(self.t_A[m].weight, a=5 ** 0.5)
            nn.init.zeros_(self.t_B[m].weight)

    def _visual_bank(
        self,
        h: torch.Tensor,       # (B, C, H, W)
        A_list, dw_list, B_list,
        ln: nn.LayerNorm,
        alpha: torch.Tensor,   # (B, M)
    ) -> torch.Tensor:
        B, C, H, W = h.shape
        h_flat = h.flatten(2).permute(0, 2, 1)     # (B, N, C)
        h_ln = ln(h_flat)                           # (B, N, C)

        delta = torch.zeros_like(h_flat)
        for m in range(self.M):
            a = A_list[m](h_ln)                     # (B, N, rank)
            a = a.permute(0, 2, 1).view(B, self.rank, H, W)
            a = dw_list[m](a)                       # (B, rank, H, W)
            a = a.flatten(2).permute(0, 2, 1)       # (B, N, rank)
            b = B_list[m](a)                        # (B, N, C)
            w = alpha[:, m].view(B, 1, 1)           # (B, 1, 1)
            delta = delta + w * b

        h_out = h_flat + (self.eta / self.rank) * delta   # (B, N, C)
        return h_out.permute(0, 2, 1).view(B, C, H, W)

    def _text_bank(
        self,
        t: torch.Tensor,       # (B, L, D)
        A_list, B_list,
        ln: nn.LayerNorm,
        alpha: torch.Tensor,   # (B, M)
    ) -> torch.Tensor:
        t_ln = ln(t)   # (B, L, D)

        delta = torch.zeros_like(t)
        for m in range(self.M):
            a = A_list[m](t_ln)                     # (B, L, rank)
            b = B_list[m](a)                        # (B, L, D)
            w = alpha[:, m].view(-1, 1, 1)
            delta = delta + w * b

        return t + (self.eta / self.rank) * delta

    def forward(
        self,
        f3: torch.Tensor,
        f4: torch.Tensor,
        z_text: torch.Tensor,
        alpha: torch.Tensor,
    ):
        f3p = self._visual_bank(f3, self.f3_A, self.f3_dw, self.f3_B, self.f3_ln, alpha)
        f4p = self._visual_bank(f4, self.f4_A, self.f4_dw, self.f4_B, self.f4_ln, alpha)
        tp = self._text_bank(z_text, self.t_A, self.t_B, self.t_ln, alpha)
        return f3p, f4p, tp
