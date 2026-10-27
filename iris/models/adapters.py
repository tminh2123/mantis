"""
Visual Adapters for frozen pretrained encoders.

Adapters allow task-specific learning while keeping the backbone frozen,
which preserves pretrained representations and reduces overfitting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BottleneckAdapter(nn.Module):
    """
    Bottleneck-style adapter for visual features.
    
    Architecture: x + scale * MLP(x)
    where MLP = Conv1x1(down) -> ReLU -> Conv1x1(up)
    
    Similar to CLIP-Adapter and SAM-Adapter designs.
    Preserves spatial dimensions while adding task-specific adaptation.
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio (default: 4)
        init_scale: Initial scale factor (default: 0.1, starts conservative)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        reduction: int = 4,
        init_scale: float = 0.1,
    ):
        super().__init__()
        
        mid_channels = max(in_channels // reduction, 64)
        
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        
        # Learnable scale - starts small to preserve original features
        self.scale = nn.Parameter(torch.ones(1) * init_scale)
        
        # Initialize adapter to near-zero output
        self._init_weights()
    
    def _init_weights(self):
        """Initialize to near-identity at start."""
        for m in self.adapter:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
        Returns:
            Adapted features [B, C, H, W]
        """
        return x + self.scale * self.adapter(x)


class SpatialAdapter(nn.Module):
    """
    Spatial-aware adapter with 3x3 convolutions.
    
    Better at capturing local spatial patterns than pure 1x1 bottleneck.
    Useful when the pretrained features lack task-specific spatial details.
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio (default: 4)
        init_scale: Initial scale factor (default: 0.1)
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        init_scale: float = 0.1,
    ):
        super().__init__()
        
        mid_channels = max(in_channels // reduction, 64)
        
        self.adapter = nn.Sequential(
            # Down-project
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # Spatial mixing
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # Up-project
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        
        self.scale = nn.Parameter(torch.ones(1) * init_scale)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.adapter:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.adapter(x)


class MultiScaleAdapter(nn.Module):
    """
    Multi-scale adapter that processes features at multiple resolutions.
    
    Captures both fine-grained and coarse patterns.
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio (default: 4)
        init_scale: Initial scale factor (default: 0.1)
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction: int = 4,
        init_scale: float = 0.1,
    ):
        super().__init__()
        
        mid_channels = max(in_channels // reduction, 64)
        
        # Local branch (1x1)
        self.local_adapter = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        # Spatial branch (3x3)
        self.spatial_adapter = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        
        self.scale = nn.Parameter(torch.ones(1) * init_scale)
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.local_adapter, self.spatial_adapter, self.fusion]:
            for m in module:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feat = self.local_adapter(x)
        spatial_feat = self.spatial_adapter(x)
        
        fused = torch.cat([local_feat, spatial_feat], dim=1)
        out = self.fusion(fused)
        
        return x + self.scale * out


def get_adapter(
    adapter_type: str,
    in_channels: int,
    reduction: int = 4,
    init_scale: float = 0.1,
) -> nn.Module:
    """
    Factory function to create adapters.
    
    Args:
        adapter_type: "bottleneck", "spatial", or "multiscale"
        in_channels: Number of input channels
        reduction: Channel reduction ratio
        init_scale: Initial scale factor
    
    Returns:
        Adapter module
    """
    adapters = {
        "bottleneck": BottleneckAdapter,
        "spatial": SpatialAdapter,
        "multiscale": MultiScaleAdapter,
    }
    
    if adapter_type not in adapters:
        raise ValueError(f"Unknown adapter type: {adapter_type}. Choose from {list(adapters.keys())}")
    
    return adapters[adapter_type](
        in_channels=in_channels,
        reduction=reduction,
        init_scale=init_scale,
    )
