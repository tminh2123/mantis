"""
RegionBridge for MRSeg.

Converts text tokens into R spatial region queries, refines them via
self-attention + cross-attention, then back-projects onto the visual feature map.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RefinementBlock(nn.Module):
    """One self-attn + cross-attn + FFN block for region queries."""

    def __init__(self, dim: int = 192, text_dim: int = 768, num_heads: int = 4):
        super().__init__()
        self.self_ln = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.cross_ln = nn.LayerNorm(dim)
        self.kv_proj = nn.Linear(text_dim, dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.ffn_ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, g: torch.Tensor, t_prime: torch.Tensor) -> torch.Tensor:
        """
        g:       (B, R, D) region queries
        t_prime: (B, L, text_dim)
        """
        # Self-attention
        g_n = self.self_ln(g)
        g = g + self.self_attn(g_n, g_n, g_n)[0]

        # Cross-attention: G queries, projected t' as K, V
        g_n = self.cross_ln(g)
        kv = self.kv_proj(t_prime)                       # (B, L, D)
        g = g + self.cross_attn(g_n, kv, kv)[0]

        # FFN
        g = g + self.ffn(self.ffn_ln(g))

        return g


class RegionBridge(nn.Module):
    """
    Bridges visual features and text tokens via learnable spatial regions.

    visual_channels: C  (384 for f3, 768 for f4)
    num_regions:     R  (12 for f3, 6 for f4)
    D:               region/query dimension (192)
    """

    def __init__(
        self,
        visual_channels: int,
        num_regions: int,
        D: int = 192,
        text_dim: int = 768,
        num_heads: int = 4,
        num_refine: int = 2,
    ):
        super().__init__()
        self.R = num_regions
        self.D = D

        self.wq = nn.Linear(text_dim, num_regions * D)
        self.wf = nn.Conv2d(visual_channels, D, 1)
        self.z_ln = nn.LayerNorm(D)

        self.refine_blocks = nn.ModuleList([
            RefinementBlock(D, text_dim, num_heads) for _ in range(num_refine)
        ])

        self.out_proj = nn.Conv2d(D, visual_channels, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, F_in: torch.Tensor, t_prime: torch.Tensor) -> torch.Tensor:
        """
        F_in:    (B, C, H, W)
        t_prime: (B, L, text_dim)
        Returns: (B, C, H, W)
        """
        B, C, H, W = F_in.shape
        N = H * W
        R, D = self.R, self.D

        # Build region queries from mean text embedding
        mean_t = t_prime.mean(1)                            # (B, text_dim)
        Q = self.wq(mean_t).view(B, R, D)                  # (B, R, D)

        # Project visual features and flatten spatial
        Z = self.wf(F_in).flatten(2).permute(0, 2, 1)      # (B, N, D)
        Z_ln = self.z_ln(Z)                                 # (B, N, D)

        # Soft assignment: R queries attend to N positions
        A = F.softmax(
            torch.bmm(Q, Z_ln.transpose(1, 2)) / math.sqrt(D),
            dim=-1,
        )   # (B, R, N)

        # Aggregate visual features into region tokens
        G = torch.bmm(A, Z)   # (B, R, D)

        # Refine with text context
        for block in self.refine_blocks:
            G = block(G, t_prime)   # (B, R, D)

        # Back-project onto spatial grid
        recon = torch.bmm(A.transpose(1, 2), G)              # (B, N, D)
        recon = recon.permute(0, 2, 1).view(B, D, H, W)      # (B, D, H, W)

        delta = self.out_proj(recon)                          # (B, C, H, W)
        return F_in + self.gamma * delta
