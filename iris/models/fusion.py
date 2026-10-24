"""
Fusion Modules for combining visual and text features
- Cross-Attention (from C2Seg BCAM)
- CBAM (Convolutional Block Attention Module)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion between visual and text features.
    Similar to BCAM in C2Seg.
    """
    
    def __init__(self, v_dim: int, t_dim: int = 512, num_heads: int = 8):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = v_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Project text to visual dimension
        self.t_proj = nn.Linear(t_dim, v_dim)
        
        # Q from visual, K/V from text
        self.q_proj = nn.Conv2d(v_dim, v_dim, 1)
        self.k_proj = nn.Linear(v_dim, v_dim)
        self.v_proj = nn.Linear(v_dim, v_dim)
        
        # Output projection
        self.out_proj = nn.Conv2d(v_dim, v_dim, 1)
        
        # Residual gate
        self.gate = nn.Sequential(
            nn.Conv2d(v_dim * 2, v_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v: Visual features [B, C, H, W]
            t: Text features [B, N, 512]
        
        Returns:
            Fused features [B, C, H, W]
        """
        B, C, H, W = v.shape
        
        # Project text
        t = self.t_proj(t)  # [B, N, C]
        
        # Compute Q, K, V
        q = self.q_proj(v)  # [B, C, H, W]
        q = q.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        k = self.k_proj(t)  # [B, N, C]
        v_text = self.v_proj(t)  # [B, N, C]
        
        # Multi-head attention
        q = q.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_text = v_text.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v_text
        out = out.transpose(1, 2).reshape(B, H*W, C)
        out = out.transpose(1, 2).view(B, C, H, W)
        
        out = self.out_proj(out)
        
        # Gated residual
        gate = self.gate(torch.cat([v, out], dim=1))
        out = v + gate * out
        
        return out


class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        
        avg_out = self.fc(self.avg_pool(x).view(B, C))
        max_out = self.fc(self.max_pool(x).view(B, C))
        
        out = torch.sigmoid(avg_out + max_out)
        return out.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        
        out = torch.cat([avg_out, max_out], dim=1)
        out = torch.sigmoid(self.conv(out))
        
        return out


class CBAMFusion(nn.Module):
    """
    CBAM-based fusion with text guidance.
    Uses text features to modulate channel attention.
    """
    
    def __init__(self, v_dim: int, t_dim: int = 512, reduction: int = 16):
        super().__init__()
        
        # Project text to channel weights
        self.text_to_channel = nn.Sequential(
            nn.Linear(t_dim, v_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(v_dim // reduction, v_dim),
            nn.Sigmoid()
        )
        
        # Standard CBAM
        self.channel_att = ChannelAttention(v_dim, reduction)
        self.spatial_att = SpatialAttention()
        
        # Combine text-guided and CBAM attention
        self.combine = nn.Sequential(
            nn.Conv2d(v_dim, v_dim, 1),
            nn.BatchNorm2d(v_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v: Visual features [B, C, H, W]
            t: Text features [B, N, 512]
        
        Returns:
            Fused features [B, C, H, W]
        """
        B, C, H, W = v.shape
        
        # Text-guided channel attention
        t_pooled = t.mean(dim=1)  # [B, 512]
        t_weights = self.text_to_channel(t_pooled)  # [B, C]
        t_weights = t_weights.view(B, C, 1, 1)
        
        # CBAM channel attention
        c_weights = self.channel_att(v)
        
        # Combined channel attention
        channel_att = t_weights * c_weights
        v = v * channel_att
        
        # Spatial attention
        s_weights = self.spatial_att(v)
        v = v * s_weights
        
        # Final projection
        v = self.combine(v)
        
        return v


def get_fusion_module(name: str, v_dim: int, t_dim: int = 512) -> nn.Module:
    """Get fusion module by name."""
    if name == "cross_attention":
        return CrossAttentionFusion(v_dim, t_dim)
    elif name == "cbam":
        return CBAMFusion(v_dim, t_dim)
    else:
        raise ValueError(f"Unknown fusion type: {name}")
