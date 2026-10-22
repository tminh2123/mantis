# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class PixLevelModule(nn.Module):
    """
    Pixel-Level Attention Module.
    Combines average-pooled and max-pooled features to generate pixel-wise attention weights.

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int): Hidden expansion ratio for the bottleneck MLP.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 2):
        super().__init__()
        self.conv_avg = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_max = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(3 * reduction_ratio, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W) with pixel-wise attention.
        """
        # Channel-reduced average and max features
        avg_feat = torch.mean(self.conv_avg(x), dim=1, keepdim=True)  # (B, 1, H, W)
        max_feat = torch.max(self.conv_max(x), dim=1, keepdim=True).values  # (B, 1, H, W)

        # Combine features
        combined = torch.cat([avg_feat, max_feat, avg_feat + max_feat], dim=1)  # (B, 3, H, W)

        # Bottleneck MLP across channel dimension (treat last dim as "features")
        out = combined.permute(0, 2, 3, 1)            # (B, H, W, 3)
        out = self.bottleneck(out)                    # (B, H, W, 1)
        out = out.permute(0, 3, 1, 2)                 # (B, 1, H, W)

        # Apply pixel-wise attention
        return x * out

