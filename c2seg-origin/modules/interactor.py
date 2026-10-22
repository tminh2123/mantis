# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


class PatchEmbedding(nn.Module):
    """
    Convert image into patch embeddings + learnable positional embeddings.
    """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, dropout: float = 0.1):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)

        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return None
        x = self.proj(x)                      # [B, C, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)      # [B, N_patches, C]
        return self.dropout(x + self.pos_embed)


class ConvBNReLU(nn.Module):
    """
    1D Convolution + BatchNorm + ReLU
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Interactor(nn.Module):
    """
    Cross-modal interaction between image patches and text features
    using multi-head attention and convolutional refinement.
    """
    def __init__(self, img_size: int, in_channels: int, patch_size: int,
                 embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, dropout)
        self.conv_refine = ConvBNReLU(embed_dim, embed_dim)

        # Split heads equally between text and image
        self.attn_text = nn.MultiheadAttention(embed_dim, num_heads // 2, dropout=dropout, batch_first=True)
        self.attn_img = nn.MultiheadAttention(embed_dim, num_heads // 2, dropout=dropout, batch_first=True)

    def forward(self, img: torch.Tensor, text_feat: torch.Tensor):
        """
        Args:
            img:  [B, C, H, W]
            text_feat: [B, N, C]

        Returns:
            img_feat:  [B, P, C]  (patch features)
            text_feat: [B, N, C]  (text features refined)
        """
        img_feat = self.patch_embed(img)   # [B, P, C]

        # Text attends to image
        text_feat, _ = self.attn_text(text_feat, img_feat, img_feat)
        # Image attends to text
        img_feat, _ = self.attn_img(img_feat, text_feat, text_feat)

        # Convolutional refinement for text
        text_feat = self.conv_refine(text_feat.transpose(1, 2)).transpose(1, 2)

        return img_feat, text_feat

