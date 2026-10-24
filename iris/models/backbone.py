"""
Visual Backbone Encoders
- ResNet (baseline)
- Vision Transformer
- MedCLIP (medical pretrained)
"""
import torch
import torch.nn as nn
from typing import List, Tuple
from torchvision import models


class ResNetBackbone(nn.Module):
    """ResNet backbone with multi-scale feature extraction."""
    
    def __init__(self, variant: str = "resnet50", pretrained: bool = True):
        super().__init__()
        
        if variant == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            self.channels = [256, 512, 1024, 2048]
        elif variant == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            self.channels = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unknown ResNet variant: {variant}")
        
        # Extract layers
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features."""
        x = self.stem(x)
        
        f1 = self.layer1(x)   # 1/4
        f2 = self.layer2(f1)  # 1/8
        f3 = self.layer3(f2)  # 1/16
        f4 = self.layer4(f3)  # 1/32
        
        return [f1, f2, f3, f4]


class ViTBackbone(nn.Module):
    """Vision Transformer backbone."""
    
    def __init__(self, variant: str = "vit_base", pretrained: bool = True):
        super().__init__()
        
        # TODO: Implement ViT backbone with multi-scale features
        # Options:
        # - Use intermediate transformer layers as multi-scale
        # - Use FPN-style feature pyramid
        # - Use hierarchical ViT (Swin)
        
        raise NotImplementedError("ViT backbone not yet implemented")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        pass


class MedCLIPBackbone(nn.Module):
    """MedCLIP visual encoder for medical images."""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # TODO: Load MedCLIP visual encoder
        # - Extract multi-scale features from ViT
        # - May need adapter layers for feature pyramid
        
        raise NotImplementedError("MedCLIP backbone not yet implemented")
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        pass


def get_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, List[int]]:
    """
    Get backbone by name.
    
    Returns:
        backbone: nn.Module
        channels: List of channel dimensions for each scale
    """
    if name in ["resnet50", "resnet34"]:
        backbone = ResNetBackbone(name, pretrained)
        return backbone, backbone.channels
    elif name == "vit_base":
        backbone = ViTBackbone(name, pretrained)
        return backbone, backbone.channels
    elif name == "medclip":
        backbone = MedCLIPBackbone(pretrained)
        return backbone, backbone.channels
    else:
        raise ValueError(f"Unknown backbone: {name}")
