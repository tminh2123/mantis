"""
IRIS with MedKLIP Backbone

This version of IRIS uses MedKLIP pretrained encoders:
  - Visual: ResNet50 pretrained on MIMIC-CXR with medical knowledge
  - Text: BioBERT/PubMedBERT with entity translation

The architecture adapts IRIS's fusion and decoder to work with MedKLIP's
different channel dimensions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from .medklip_encoder import MedKLIPVisualEncoder, MedKLIPTextEncoder, translate_entity
from .iris import (
    MLPGate, 
    Sinusoidal2DPositionEmbedding, 
    PixLevelModule,
    ChannelGate,
    GateModule,
    ConvBNReLU,
    conv_stack,
    UpBlockAttention
)


class BCAModuleMedKLIP(nn.Module):
    """
    Bidirectional Cross Attention adapted for MedKLIP dimensions.
    
    MedKLIP text encoder outputs 768-dim (BERT), projected to 512.
    """
    def __init__(
        self, 
        img_size: int, 
        in_channels: int, 
        text_dim: int = 512,  # After projection
        num_heads: int = 8, 
        dropout: float = 0.0,
        expansion: int = 4,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_heads = num_heads

        self.pos_embed = Sinusoidal2DPositionEmbedding(in_channels, img_size, img_size)

        # Projection layers
        self.v_key = self._make_conv(in_channels, in_channels)
        self.v_value = self._make_conv(in_channels, in_channels)
        self.l_key = self._make_conv(text_dim, in_channels)
        self.l_value = self._make_conv(text_dim, in_channels)

        # MLP Gate module
        self.gate = GateModule(in_channels, expansion, dropout)

        # Output projection
        self.project = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm1d(in_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    @staticmethod
    def _make_conv(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.InstanceNorm1d(out_ch)
        )

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat: [B, C, H, W]
            text_feat: [B, N, text_dim]
        Returns:
            Fused features [B, C, H, W]
        """
        b, c, h, w = img_feat.shape
        hw = h * w
        n_t = text_feat.size(1)

        # Flatten image features and add position encoding
        img_flat = img_feat.flatten(2)  # [B, C, HW]
        v = self.pos_embed(img_feat).flatten(2)  # [B, C, HW]
        l = text_feat.transpose(1, 2)  # [B, text_dim, N]

        # Keys and values
        v_key, v_value = self.v_key(v), self.v_value(v)  # [B, C, HW]
        l_key, l_value = self.l_key(l), self.l_value(l)  # [B, C, N]

        # Reshape for multi-head attention
        head_dim = self.in_channels // self.num_heads
        v_key = v_key.view(b, self.num_heads, head_dim, hw).permute(0, 1, 3, 2)
        l_key = l_key.view(b, self.num_heads, head_dim, n_t)
        v_value = v_value.view(b, self.num_heads, head_dim, hw).permute(0, 1, 3, 2)
        l_value = l_value.view(b, self.num_heads, head_dim, n_t).permute(0, 1, 3, 2)

        # Attention maps
        sim_map = torch.matmul(v_key, l_key) * (self.in_channels ** -0.5)
        attn_v2l = F.softmax(sim_map, dim=-1)
        attn_l2v = F.softmax(sim_map.transpose(-1, -2), dim=-1)

        # Cross attention outputs
        v_out = torch.matmul(attn_v2l, l_value)
        f_v = v_out.permute(0, 2, 1, 3).reshape(b, hw, c)

        l_out = torch.einsum("b h i j, b h j k -> b h j k", attn_l2v, v_value)
        f_l = l_out.permute(0, 2, 1, 3).reshape(b, hw, c)

        # MLP Gate fusion
        fused = self.gate(f_l, f_v)  # [B, C, HW]
        
        # Residual connection
        out = self.project(img_flat + fused).view(b, c, h, w)

        return out


class UpBlockMedKLIP(nn.Module):
    """Up-sampling block adapted for MedKLIP ResNet channel dimensions."""
    def __init__(
        self, 
        skip_channels: int, 
        up_channels: int, 
        out_channels: int, 
        num_layers: int = 2
    ):
        super().__init__()
        self.att = PixLevelModule(skip_channels)
        self.up = nn.ConvTranspose2d(up_channels, up_channels, kernel_size=2, stride=2)
        self.convs = conv_stack(skip_channels + up_channels, out_channels, num_layers)

    def forward(self, skip_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        skip_att = self.att(skip_x)
        up_x = self.up(x)
        if up_x.shape[2:] != skip_att.shape[2:]:
            up_x = F.interpolate(up_x, size=skip_att.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([skip_att, up_x], dim=1)
        return self.convs(out)


class IRISMedKLIP(nn.Module):
    """
    IRIS with MedKLIP Backbone.
    
    Uses MedKLIP pretrained visual encoder (ResNet50) and text encoder (BioBERT).
    Adapts the channel dimensions from ResNet50's output to IRIS decoder.
    
    MedKLIP Visual Encoder outputs:
      - stem: 64 channels, H/2
      - layer1: 256 channels, H/4
      - layer2: 512 channels, H/8
      - layer3: 1024 channels, H/16
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 224,
        medklip_visual_path: Optional[str] = None,
        medklip_text_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        max_token_len: int = 77,
        base_channel: int = 64,
        num_heads: int = 8,
        dropout: float = 0.5,
        expansion: int = 4,
        deep_supervision: bool = True,
        freeze_visual: bool = False,
        freeze_text: bool = True,
        use_entity_translation: bool = True,
        text_mask_rate: float = 0.3,
        img_mask_rate: float = 0.3,
    ):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        self.text_mask_rate = text_mask_rate
        self.img_mask_rate = img_mask_rate
        self.use_entity_translation = use_entity_translation
        
        # MedKLIP Visual Encoder (ResNet50)
        self.visual_encoder = MedKLIPVisualEncoder(
            pretrained_path=medklip_visual_path,
            d_model=256,
            freeze=freeze_visual
        )
        
        # MedKLIP Text Encoder (BioBERT)
        self.text_encoder = MedKLIPTextEncoder(
            model_name=medklip_text_model,
            max_length=max_token_len,
            freeze=freeze_text
        )
        
        # Text projection (768 -> 512 for fusion)
        self.text_proj = nn.Linear(self.text_encoder.hidden_size, 512)
        
        # Channel dimensions from MedKLIP ResNet50
        # stem: 64, layer1: 256, layer2: 512, layer3: 1024
        
        # Channel adapters to match IRIS decoder expectations
        # We'll use these channels: 64, 128, 256, 512
        self.adapt_stem = nn.Conv2d(64, base_channel, 1)      # 64 -> base
        self.adapt_l1 = nn.Conv2d(256, base_channel * 2, 1)   # 256 -> 2*base
        self.adapt_l2 = nn.Conv2d(512, base_channel * 4, 1)   # 512 -> 4*base
        self.adapt_l3 = nn.Conv2d(1024, base_channel * 8, 1)  # 1024 -> 8*base
        
        img_scales = [img_size, img_size//2, img_size//4, img_size//8, img_size//16]
        
        # Cross-modal fusion (BCAM at 3 scales)
        self.fuse2 = BCAModuleMedKLIP(img_scales[2], base_channel * 2, 512, num_heads, dropout, expansion)
        self.fuse3 = BCAModuleMedKLIP(img_scales[3], base_channel * 4, 512, num_heads, dropout, expansion)
        self.fuse4 = BCAModuleMedKLIP(img_scales[4], base_channel * 8, 512, num_heads, dropout, expansion)
        
        # Decoder with PixLevelModule
        self.up3 = UpBlockMedKLIP(base_channel * 4, base_channel * 8, base_channel * 4)
        self.up2 = UpBlockMedKLIP(base_channel * 2, base_channel * 4, base_channel * 2)
        self.up1 = UpBlockMedKLIP(base_channel, base_channel * 2, base_channel)
        
        # Final upsample to original resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(base_channel, base_channel, kernel_size=2, stride=2),
            ConvBNReLU(base_channel, base_channel),
        )
        
        # Output heads
        self.out0 = nn.Conv2d(base_channel, num_classes, 1)
        
        if self.deep_supervision:
            self.out1 = nn.Conv2d(base_channel, num_classes, 1)
            self.out2 = nn.Conv2d(base_channel * 2, num_classes, 1)
            self.out3 = nn.Conv2d(base_channel * 4, num_classes, 1)
        
        # Projections for contrastive learning
        self.img_proj = nn.Linear(base_channel * 8, 256)
        self.text_proj_cl = nn.Linear(512, 256)
    
    def _apply_text_mask(self, token_emb: torch.Tensor) -> torch.Tensor:
        """Apply random masking to text tokens during training."""
        if not self.training or self.text_mask_rate <= 0:
            return token_emb
        
        B, N, D = token_emb.shape
        mask = torch.rand(B, N, device=token_emb.device) > self.text_mask_rate
        mask[:, 0] = True  # Keep CLS token
        mask = mask.unsqueeze(-1).expand_as(token_emb)
        
        return token_emb * mask.float()

    def _apply_img_mask(self, img_feat: torch.Tensor) -> torch.Tensor:
        """Apply random patch masking to image features during training."""
        if not self.training or self.img_mask_rate <= 0:
            return img_feat
        
        B, C, H, W = img_feat.shape
        patch_size = max(1, H // 7)
        pH, pW = H // patch_size, W // patch_size
        
        patch_mask = torch.rand(B, 1, pH, pW, device=img_feat.device) > self.img_mask_rate
        mask = F.interpolate(patch_mask.float(), size=(H, W), mode='nearest')
        
        return img_feat * mask

    def forward(
        self, 
        images: torch.Tensor, 
        texts: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: [B, 3, H, W]
            texts: List of text prompts
            
        Returns:
            Dictionary with 'pred' and optionally 'aux_preds'
        """
        # Encode text with entity translation
        l = None
        if texts is not None:
            if self.use_entity_translation:
                texts = [translate_entity(t) if len(t.split()) <= 3 else t for t in texts]
            
            token_emb, _ = self.text_encoder(texts, images.device)
            token_emb = self.text_proj(token_emb)  # [B, N, 512]
            l = self._apply_text_mask(token_emb)
        
        # Encode image with MedKLIP visual encoder
        features = self.visual_encoder(images)
        
        # Adapt channels
        v0 = self.adapt_stem(features['stem'])   # [B, base, H/2, W/2]
        v1 = self.adapt_l1(features['layer1'])   # [B, 2*base, H/4, W/4]
        v2 = self.adapt_l2(features['layer2'])   # [B, 4*base, H/8, W/8]
        v3 = self.adapt_l3(features['layer3'])   # [B, 8*base, H/16, W/16]
        
        # Apply image masking
        v3 = self._apply_img_mask(v3)
        
        # Cross-modal fusion + decoder
        if l is not None:
            f3 = self.fuse4(v3, l)  # [B, 8*base, H/16, W/16]
            
            up3 = self.up3(v2, f3)  # [B, 4*base, H/8, W/8]
            f2 = self.fuse3(up3, l)
            
            up2 = self.up2(v1, f2)  # [B, 2*base, H/4, W/4]
            f1 = self.fuse2(up2, l)
            
            up1 = self.up1(v0, f1)  # [B, base, H/2, W/2]
        else:
            # No text fusion
            up3 = self.up3(v2, v3)
            up2 = self.up2(v1, up3)
            up1 = self.up1(v0, up2)
            f1, f2, f3 = up2, up3, v3
        
        # Final upsample
        up0 = self.final_up(up1)  # [B, base, H, W]
        
        # Output
        pred = self.out0(up0)
        outputs = {'pred': pred}

        if self.deep_supervision and self.training:
            outputs['aux_preds'] = [
                self.out1(up1),
                self.out2(f1),
                self.out3(f2),
            ]
        
        return outputs

    def get_contrastive_embeddings(
        self, 
        images: torch.Tensor, 
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For CaCL pretraining."""
        # Encode image
        features = self.visual_encoder(images)
        v3 = self.adapt_l3(features['layer3'])
        
        # Global image vector
        v_global = F.adaptive_avg_pool2d(v3, 1).flatten(1)
        
        # Encode text
        if self.use_entity_translation:
            texts = [translate_entity(t) if len(t.split()) <= 3 else t for t in texts]
        
        with torch.no_grad():
            _, sent_emb = self.text_encoder(texts, images.device)
        
        sent_emb = self.text_proj(sent_emb)
        
        v = F.normalize(self.img_proj(v_global), dim=-1)
        l = F.normalize(self.text_proj_cl(sent_emb), dim=-1)
        l_raw = F.normalize(sent_emb.detach().float(), dim=-1)
        
        return v, l, l_raw


def build_iris_medklip(config) -> IRISMedKLIP:
    """Build IRIS-MedKLIP model from config."""
    mcfg = config.get('model', config)
    
    return IRISMedKLIP(
        num_classes=mcfg.get('num_classes', 1),
        img_size=config.get('data', {}).get('image_size', 224),
        medklip_visual_path=mcfg.get('medklip_visual_path', None),
        medklip_text_model=mcfg.get('medklip_text_model', 
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
        max_token_len=mcfg.get('max_token_len', 77),
        base_channel=mcfg.get('base_channel', 64),
        num_heads=mcfg.get('num_heads', 8),
        dropout=mcfg.get('dropout', 0.5),
        expansion=mcfg.get('expansion', 4),
        deep_supervision=mcfg.get('deep_supervision', True),
        freeze_visual=mcfg.get('freeze_medklip_visual', False),
        freeze_text=mcfg.get('freeze_medklip_text', True),
        use_entity_translation=mcfg.get('use_entity_translation', True),
        text_mask_rate=mcfg.get('text_mask_rate', 0.3),
        img_mask_rate=mcfg.get('img_mask_rate', 0.3),
    )
