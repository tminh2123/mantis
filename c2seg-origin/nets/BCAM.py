# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.KCG import KCGate


class Sinusoidal2DPositionEmbedding(nn.Module):
    """
    2D sinusoidal position embedding for image features.
    Generates deterministic positional encodings of shape [1, C, H, W].
    """

    def __init__(self, num_channels: int, height: int, width: int):
        super().__init__()
        assert num_channels % 4 == 0, "Embedding dim must be divisible by 4 for 2D encoding"
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.register_buffer("pos_encoding", self._build_sincos_pos_encoding())

    def _build_sincos_pos_encoding(self) -> torch.Tensor:
        pe = torch.zeros(self.num_channels, self.height, self.width)

        y_pos = torch.arange(self.height, dtype=torch.float32).unsqueeze(1).repeat(1, self.width)
        x_pos = torch.arange(self.width, dtype=torch.float32).unsqueeze(0).repeat(self.height, 1)

        half_dim = self.num_channels // 2
        div_term = torch.exp(torch.arange(0, half_dim, 2).float() * -(math.log(10000.0) / half_dim))

        pe[0::4, :, :] = torch.sin(y_pos[None, :, :] * div_term[:, None, None])  # sin(y)
        pe[1::4, :, :] = torch.cos(y_pos[None, :, :] * div_term[:, None, None])  # cos(y)
        pe[2::4, :, :] = torch.sin(x_pos[None, :, :] * div_term[:, None, None])  # sin(x)
        pe[3::4, :, :] = torch.cos(x_pos[None, :, :] * div_term[:, None, None])  # cos(x)

        return pe.unsqueeze(0)  # [1, C, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_encoding


class BCAModule(nn.Module):
    """
    Bidirectional Cross Attention (BCA) module.
    Performs cross-attention between image and text features.

    Args:
        img_size (int): Input image size (assumed square).
        in_channels (int): Number of feature channels.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """

    def __init__(self, img_size: int, in_channels: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_heads = num_heads

        self.pos_embed = Sinusoidal2DPositionEmbedding(in_channels, img_size, img_size)

        # Projection layers
        self.v_key = self._make_conv(in_channels, in_channels)
        self.v_value = self._make_conv(in_channels, in_channels)
        self.l_key = self._make_conv(512, in_channels)
        self.l_value = self._make_conv(512, in_channels)

        # Gating mechanisms
        self.gate_v = KCGate(in_channels)
        self.gate_l = KCGate(in_channels)

        # Output projection
        self.project = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm1d(in_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    @staticmethod
    def _make_conv(in_ch: int, out_ch: int) -> nn.Sequential:
        """Helper function to build Conv1d + InstanceNorm1d block"""
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.InstanceNorm1d(out_ch)
        )

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            img_feat (Tensor): Image features of shape (B, C, H, W).
            text_feat (Tensor): Text features of shape (B, N, C_txt).

        Returns:
            Tensor: Fused features of shape (B, C, H, W).
        """
        b, c, h, w = img_feat.shape
        hw = h * w
        n_t = text_feat.size(1)

        # Flatten image features and add position encoding
        img_flat = img_feat.flatten(2)  # (B, C, HW)
        v = self.pos_embed(img_feat).flatten(2)  # (B, C, HW)
        l = text_feat.transpose(1, 2)  # (B, C, N)

        # Keys and values
        v_key, v_value = self.v_key(v), self.v_value(v)  # (B, C, HW)
        l_key, l_value = self.l_key(l), self.l_value(l)  # (B, C, N)

        # Reshape for multi-head attention
        head_dim = self.in_channels // self.num_heads
        v_key = v_key.view(b, self.num_heads, head_dim, hw).permute(0, 1, 3, 2)       # (B, H, HW, D)
        l_key = l_key.view(b, self.num_heads, head_dim, n_t)                          # (B, H, D, N)
        v_value = v_value.view(b, self.num_heads, head_dim, hw).permute(0, 1, 3, 2)   # (B, H, HW, D)
        l_value = l_value.view(b, self.num_heads, head_dim, n_t).permute(0, 1, 3, 2)  # (B, H, N, D)

        # Attention maps
        sim_map = torch.matmul(v_key, l_key) * (self.in_channels ** -0.5)  # (B, H, HW, N)
        attn_v2l = F.softmax(sim_map, dim=-1)                      # (B, H, HW, N)
        attn_l2v = F.softmax(sim_map.transpose(-1, -2), dim=-1)    # (B, H, N, HW)

        # Cross attention
        v_out = torch.matmul(attn_v2l, l_value)                    # (B, H, HW, D)
        v_out = v_out.permute(0, 2, 1, 3).reshape(b, hw, c)        # (B, HW, C)
        a_v = self.gate_v(v_out).permute(0, 2, 1)                  # (B, C, HW)

        l_out = torch.einsum("b h i j, b h j k -> b h j k", attn_l2v, v_value)  # (B, H, HW, D)
        l_out = l_out.permute(0, 2, 1, 3).reshape(b, hw, c)                          # (B, HW, C)
        a_l = self.gate_l(l_out).permute(0, 2, 1)                                    # (B, C, HW)

        # Fuse and project
        fused = self.project(img_flat + a_v * a_l).view(b, c, h, w)

        return fused

