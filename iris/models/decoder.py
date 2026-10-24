"""
Decoder with Deep Supervision
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class ConvBlock(nn.Module):
    """Basic conv block: Conv -> BN -> ReLU."""
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsample + Conv block."""
    
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            ConvBlock(in_ch + skip_ch, out_ch),
            ConvBlock(out_ch, out_ch),
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        
        return x


class SegmentationHead(nn.Module):
    """Output head for segmentation."""
    
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        
        self.conv = nn.Conv2d(in_ch, num_classes, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Decoder(nn.Module):
    """
    UNet-style decoder with optional deep supervision.
    
    Takes multi-scale encoder features and produces segmentation mask(s).
    """
    
    def __init__(
        self,
        encoder_channels: List[int],  # [256, 512, 1024, 2048] for ResNet50
        decoder_channels: List[int] = [256, 128, 64, 32],
        num_classes: int = 1,
        deep_supervision: bool = True,
    ):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        
        # Reverse encoder channels (bottom-up to top-down)
        enc_ch = encoder_channels[::-1]  # [2048, 1024, 512, 256]
        dec_ch = decoder_channels  # [256, 128, 64, 32]
        
        # Initial conv to reduce deepest features
        self.init_conv = ConvBlock(enc_ch[0], dec_ch[0])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in range(len(dec_ch) - 1):
            in_ch = dec_ch[i]
            skip_ch = enc_ch[i + 1] if i + 1 < len(enc_ch) else 0
            out_ch = dec_ch[i + 1]
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch))
        
        # Final upsample to full resolution
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            ConvBlock(dec_ch[-1], dec_ch[-1]),
        )
        
        # Main segmentation head
        self.seg_head = SegmentationHead(dec_ch[-1], num_classes)
        
        # Auxiliary heads for deep supervision
        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                SegmentationHead(ch, num_classes) for ch in dec_ch[:-1]
            ])
    
    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Multi-scale features from encoder
                      [f1, f2, f3, f4] from shallow to deep
        
        Returns:
            Dictionary with:
                - 'pred': Main prediction at full resolution
                - 'aux_preds': List of auxiliary predictions (if deep_supervision)
        """
        # Reverse features (deep to shallow)
        features = features[::-1]  # [f4, f3, f2, f1]
        
        # Initial conv
        x = self.init_conv(features[0])
        
        aux_outputs = []
        
        # Upsampling path
        for i, up_block in enumerate(self.up_blocks):
            if self.deep_supervision:
                aux_outputs.append(x)
            
            skip = features[i + 1] if i + 1 < len(features) else None
            if skip is not None:
                x = up_block(x, skip)
            else:
                x = up_block.up(x)
                x = up_block.conv(torch.cat([x, x], dim=1))  # Self-concat if no skip
        
        # Final upsample
        x = self.final_up(x)
        
        # Main prediction
        pred = self.seg_head(x)
        
        outputs = {'pred': pred}
        
        # Auxiliary predictions
        if self.deep_supervision and self.training:
            aux_preds = []
            for aux_feat, aux_head in zip(aux_outputs, self.aux_heads):
                aux_pred = aux_head(aux_feat)
                aux_preds.append(aux_pred)
            outputs['aux_preds'] = aux_preds
        
        return outputs
