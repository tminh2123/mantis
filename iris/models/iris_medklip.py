"""
IRIS with MedKLIP Backbone

This version of IRIS uses MedKLIP pretrained encoders:
  - Visual: ResNet50 pretrained on MIMIC-CXR with medical knowledge
  - Text: BioBERT/PubMedBERT with entity translation

The architecture adapts IRIS's fusion and decoder to work with MedKLIP's
different channel dimensions.

Ablation options:
  - visual_encoder: "cnn" (IRIS original) | "medklip" (MedKLIP ResNet50)
  - text_encoder: "clip" (IRIS original) | "medklip" (PubMedBERT + entity translation)
  - fusion_type: "bcam" (original) | "bifmd" (BiFMD fusion)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from .medklip_encoder import MedKLIPVisualEncoder, MedKLIPTextEncoder, translate_entity
from .bifmd import BiFMD
from .iris import (
    IRIS,
    MLPGate, 
    Sinusoidal2DPositionEmbedding, 
    PixLevelModule,
    ChannelGate,
    GateModule,
    ConvBNReLU,
    conv_stack,
    UpBlockAttention,
    DownBlock,
    BCAModule,
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


# =============================================================================
# IRIS Ablation Model - Flexible encoder selection
# =============================================================================

class IRISAblation(nn.Module):
    """
    IRIS with flexible encoder selection for ablation studies.
    
    Supports mixing different visual and text encoders:
      - visual_encoder: "cnn" (IRIS original) | "medklip" (MedKLIP ResNet50)
      - text_encoder: "clip" (IRIS original) | "medklip" (PubMedBERT)
      - fusion_type: "bcam" (IRIS original) | "bifmd" (BiFMD fusion)
    
    Ablation configurations:
      1. visual=cnn, text=clip       -> IRIS baseline
      2. visual=cnn, text=medklip    -> IRIS + MedKLIP text
      3. visual=medklip, text=clip   -> MedKLIP visual + CLIP text
      4. visual=medklip, text=medklip -> Full MedKLIP (current IRISMedKLIP)
      5. Any combination + fusion=bifmd -> Use BiFMD fusion instead of BCAM
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        img_size: int = 224,
        # Encoder selection
        visual_encoder_type: str = "cnn",  # "cnn" | "medklip"
        text_encoder_type: str = "clip",   # "clip" | "medklip"
        # Fusion selection
        fusion_type: str = "bcam",         # "bcam" | "bifmd"
        # CLIP settings (for text_encoder_type="clip")
        clip_model: str = "openai/clip-vit-base-patch32",
        # MedKLIP settings
        medklip_visual_path: Optional[str] = None,
        medklip_text_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        # Common settings
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
        
        self.visual_encoder_type = visual_encoder_type
        self.text_encoder_type = text_encoder_type
        self.fusion_type = fusion_type
        self.deep_supervision = deep_supervision
        self.text_mask_rate = text_mask_rate
        self.img_mask_rate = img_mask_rate
        self.use_entity_translation = use_entity_translation and text_encoder_type == "medklip"
        self.base_channel = base_channel
        
        # Build visual encoder
        if visual_encoder_type == "medklip":
            self.visual_encoder = MedKLIPVisualEncoder(
                pretrained_path=medklip_visual_path,
                d_model=256,
                freeze=freeze_visual
            )
            self._use_resnet_visual = True
            # Channel adapters for ResNet50
            self.adapt_stem = nn.Conv2d(64, base_channel, 1)
            self.adapt_l1 = nn.Conv2d(256, base_channel * 2, 1)
            self.adapt_l2 = nn.Conv2d(512, base_channel * 4, 1)
            self.adapt_l3 = nn.Conv2d(1024, base_channel * 8, 1)
        else:
            # Use IRIS CNN encoder (DownBlocks)
            self._use_resnet_visual = False
            self.down0 = DownBlock(3, base_channel)
            self.down1 = DownBlock(base_channel, base_channel * 2)
            self.down2 = DownBlock(base_channel * 2, base_channel * 4)
            self.down3 = DownBlock(base_channel * 4, base_channel * 8)
            self.pool = nn.MaxPool2d(2)
        
        # Build text encoder
        if text_encoder_type == "medklip":
            self.text_encoder = MedKLIPTextEncoder(
                model_name=medklip_text_model,
                max_length=max_token_len,
                freeze=freeze_text
            )
            self.text_proj = nn.Linear(self.text_encoder.hidden_size, 512)
            self._text_dim = 512
        else:
            # Use CLIP text encoder
            from transformers import CLIPModel, CLIPTokenizer
            clip = CLIPModel.from_pretrained(clip_model)
            self.lang_enc = clip.text_model
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
            self._text_dim = self.lang_enc.config.hidden_size  # 512
            if freeze_text:
                for p in self.lang_enc.parameters():
                    p.requires_grad = False
        
        self.max_token_len = max_token_len
        
        img_scales = [img_size, img_size//2, img_size//4, img_size//8, img_size//16]
        
        # Cross-modal fusion (BCAM or BiFMD at 3 scales)
        if fusion_type == "bifmd":
            self.fuse2 = BiFMD(channels=base_channel * 2, text_dim=self._text_dim, num_heads=num_heads, expansion_ratio=expansion)
            self.fuse3 = BiFMD(channels=base_channel * 4, text_dim=self._text_dim, num_heads=num_heads, expansion_ratio=expansion)
            self.fuse4 = BiFMD(channels=base_channel * 8, text_dim=self._text_dim, num_heads=num_heads, expansion_ratio=expansion)
        else:
            # Default BCAM
            self.fuse2 = BCAModuleMedKLIP(img_scales[2], base_channel * 2, self._text_dim, num_heads, dropout, expansion)
            self.fuse3 = BCAModuleMedKLIP(img_scales[3], base_channel * 4, self._text_dim, num_heads, dropout, expansion)
            self.fuse4 = BCAModuleMedKLIP(img_scales[4], base_channel * 8, self._text_dim, num_heads, dropout, expansion)
        
        # Decoder
        self.up3 = UpBlockMedKLIP(base_channel * 4, base_channel * 8, base_channel * 4)
        self.up2 = UpBlockMedKLIP(base_channel * 2, base_channel * 4, base_channel * 2)
        self.up1 = UpBlockMedKLIP(base_channel, base_channel * 2, base_channel)
        
        # Final upsample
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
        self.text_proj_cl = nn.Linear(self._text_dim, 256)
    
    def _encode_visual_cnn(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Encode with IRIS CNN encoder."""
        v0 = self.down0(x)           # [B, base, H/2, W/2]
        v1 = self.down1(self.pool(v0))  # [B, 2*base, H/4, W/4]
        v2 = self.down2(self.pool(v1))  # [B, 4*base, H/8, W/8]
        v3 = self.down3(self.pool(v2))  # [B, 8*base, H/16, W/16]
        return v0, v1, v2, v3
    
    def _encode_visual_medklip(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Encode with MedKLIP ResNet50."""
        features = self.visual_encoder(x)
        v0 = self.adapt_stem(features['stem'])
        v1 = self.adapt_l1(features['layer1'])
        v2 = self.adapt_l2(features['layer2'])
        v3 = self.adapt_l3(features['layer3'])
        return v0, v1, v2, v3
    
    def _encode_text_clip(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Encode with CLIP text encoder."""
        tokens = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_len,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            out = self.lang_enc(**tokens)
        
        return out.last_hidden_state  # [B, N, 512]
    
    def _encode_text_medklip(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Encode with MedKLIP text encoder (PubMedBERT)."""
        if self.use_entity_translation:
            texts = [translate_entity(t) if len(t.split()) <= 3 else t for t in texts]
        
        token_emb, _ = self.text_encoder(texts, device)
        token_emb = self.text_proj(token_emb)  # [B, N, 512]
        return token_emb
    
    def _apply_text_mask(self, token_emb: torch.Tensor) -> torch.Tensor:
        """Apply random masking to text tokens during training."""
        if not self.training or self.text_mask_rate <= 0:
            return token_emb
        
        B, N, D = token_emb.shape
        mask = torch.rand(B, N, device=token_emb.device) > self.text_mask_rate
        mask[:, 0] = True
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
        """Forward pass."""
        device = images.device
        
        # Encode text
        l = None
        if texts is not None:
            if self.text_encoder_type == "medklip":
                l = self._encode_text_medklip(texts, device)
            else:
                l = self._encode_text_clip(texts, device)
            l = self._apply_text_mask(l)
        
        # Encode visual
        if self._use_resnet_visual:
            v0, v1, v2, v3 = self._encode_visual_medklip(images)
        else:
            v0, v1, v2, v3 = self._encode_visual_cnn(images)
        
        v3 = self._apply_img_mask(v3)
        
        # Cross-modal fusion + decoder
        if l is not None:
            f3 = self.fuse4(v3, l)
            up3 = self.up3(v2, f3)
            f2 = self.fuse3(up3, l)
            up2 = self.up2(v1, f2)
            f1 = self.fuse2(up2, l)
            up1 = self.up1(v0, f1)
        else:
            up3 = self.up3(v2, v3)
            up2 = self.up2(v1, up3)
            up1 = self.up1(v0, up2)
            f1, f2, f3 = up2, up3, v3
        
        up0 = self.final_up(up1)
        
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
        device = images.device
        
        # Encode image
        if self._use_resnet_visual:
            _, _, _, v3 = self._encode_visual_medklip(images)
        else:
            _, _, _, v3 = self._encode_visual_cnn(images)
        
        v_global = F.adaptive_avg_pool2d(v3, 1).flatten(1)
        
        # Encode text
        if self.text_encoder_type == "medklip":
            token_emb = self._encode_text_medklip(texts, device)
            sent_emb = token_emb[:, 0, :]  # CLS token
        else:
            token_emb = self._encode_text_clip(texts, device)
            # For CLIP, use mean pooling or last token
            sent_emb = token_emb.mean(dim=1)
        
        v = F.normalize(self.img_proj(v_global), dim=-1)
        l = F.normalize(self.text_proj_cl(sent_emb), dim=-1)
        l_raw = F.normalize(sent_emb.detach().float(), dim=-1)
        
        return v, l, l_raw


def build_iris_ablation(config) -> IRISAblation:
    """
    Build IRIS ablation model from config.
    
    Config options:
      model:
        visual_encoder: "cnn" | "medklip"
        text_encoder: "clip" | "medklip"
        fusion_type: "bcam" | "bifmd"
        
    Example configs for ablation study:
      1. Baseline:         visual_encoder: cnn, text_encoder: clip, fusion_type: bcam
      2. Text only:        visual_encoder: cnn, text_encoder: medklip
      3. Visual only:      visual_encoder: medklip, text_encoder: clip
      4. Both:             visual_encoder: medklip, text_encoder: medklip
      5. BiFMD fusion:     visual_encoder: cnn, text_encoder: clip, fusion_type: bifmd
    """
    mcfg = config.get('model', config)
    
    return IRISAblation(
        num_classes=mcfg.get('num_classes', 1),
        img_size=config.get('data', {}).get('image_size', 224),
        visual_encoder_type=mcfg.get('visual_encoder', 'cnn'),
        text_encoder_type=mcfg.get('text_encoder', 'clip'),
        fusion_type=mcfg.get('fusion_type', 'bcam'),
        clip_model=mcfg.get('clip_model', 'openai/clip-vit-base-patch32'),
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


# =============================================================================
# IRIS Ablation Native - No channel adapters, uses native ResNet50 channels
# =============================================================================

class IRISAblationNative(nn.Module):
    """
    IRIS Ablation with native ResNet50 channels (no channel adapters).
    
    ResNet50 channels: stem=64, layer1=256, layer2=512, layer3=1024
    
    This version uses ResNet's native channel dimensions directly,
    avoiding the information bottleneck from channel adapters.
    
    Only supports MedKLIP visual encoder (ResNet50).
    """
    
    # ResNet50 native channel dimensions
    RESNET_CHANNELS = {
        'stem': 64,
        'layer1': 256,
        'layer2': 512,
        'layer3': 1024,
    }
    
    def __init__(
        self,
        num_classes: int = 1,
        img_size: int = 224,
        # Text encoder selection
        text_encoder_type: str = "clip",   # "clip" | "medklip"
        # Fusion selection
        fusion_type: str = "bcam",         # "bcam" | "bifmd"
        # CLIP settings
        clip_model: str = "openai/clip-vit-base-patch32",
        # MedKLIP settings
        medklip_visual_path: Optional[str] = None,
        medklip_text_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        # Common settings
        max_token_len: int = 77,
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
        
        self.text_encoder_type = text_encoder_type
        self.fusion_type = fusion_type
        self.deep_supervision = deep_supervision
        self.text_mask_rate = text_mask_rate
        self.img_mask_rate = img_mask_rate
        self.use_entity_translation = use_entity_translation and text_encoder_type == "medklip"
        
        # Native ResNet channels
        ch = self.RESNET_CHANNELS
        
        # MedKLIP Visual Encoder (ResNet50) - always used
        self.visual_encoder = MedKLIPVisualEncoder(
            pretrained_path=medklip_visual_path,
            d_model=256,
            freeze=freeze_visual
        )
        
        # Build text encoder
        if text_encoder_type == "medklip":
            self.text_encoder = MedKLIPTextEncoder(
                model_name=medklip_text_model,
                max_length=max_token_len,
                freeze=freeze_text
            )
            self.text_proj = nn.Linear(self.text_encoder.hidden_size, 512)
            self._text_dim = 512
        else:
            # Use CLIP text encoder
            from transformers import CLIPModel, CLIPTokenizer
            clip = CLIPModel.from_pretrained(clip_model)
            self.lang_enc = clip.text_model
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
            self._text_dim = self.lang_enc.config.hidden_size  # 512
            if freeze_text:
                for p in self.lang_enc.parameters():
                    p.requires_grad = False
        
        self.max_token_len = max_token_len
        
        img_scales = [img_size, img_size//2, img_size//4, img_size//8, img_size//16]
        
        # Cross-modal fusion with native ResNet channels
        # layer1=256, layer2=512, layer3=1024
        if fusion_type == "bifmd":
            self.fuse2 = BiFMD(channels=ch['layer1'], text_dim=self._text_dim, num_heads=num_heads, expansion_ratio=expansion)
            self.fuse3 = BiFMD(channels=ch['layer2'], text_dim=self._text_dim, num_heads=num_heads, expansion_ratio=expansion)
            self.fuse4 = BiFMD(channels=ch['layer3'], text_dim=self._text_dim, num_heads=num_heads, expansion_ratio=expansion)
        else:
            # Default BCAM
            self.fuse2 = BCAModuleMedKLIP(img_scales[2], ch['layer1'], self._text_dim, num_heads, dropout, expansion)
            self.fuse3 = BCAModuleMedKLIP(img_scales[3], ch['layer2'], self._text_dim, num_heads, dropout, expansion)
            self.fuse4 = BCAModuleMedKLIP(img_scales[4], ch['layer3'], self._text_dim, num_heads, dropout, expansion)
        
        # Decoder with native ResNet channels
        # up3: skip=layer2(512), up=layer3(1024) -> out=512
        # up2: skip=layer1(256), up=512 -> out=256
        # up1: skip=stem(64), up=256 -> out=64
        self.up3 = UpBlockMedKLIP(ch['layer2'], ch['layer3'], ch['layer2'])  # 512, 1024 -> 512
        self.up2 = UpBlockMedKLIP(ch['layer1'], ch['layer2'], ch['layer1'])  # 256, 512 -> 256
        self.up1 = UpBlockMedKLIP(ch['stem'], ch['layer1'], ch['stem'])      # 64, 256 -> 64
        
        # Final upsample
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(ch['stem'], ch['stem'], kernel_size=2, stride=2),
            ConvBNReLU(ch['stem'], ch['stem']),
        )
        
        # Output heads
        self.out0 = nn.Conv2d(ch['stem'], num_classes, 1)
        
        if self.deep_supervision:
            self.out1 = nn.Conv2d(ch['stem'], num_classes, 1)      # after up1: 64
            self.out2 = nn.Conv2d(ch['layer1'], num_classes, 1)    # after fuse2: 256
            self.out3 = nn.Conv2d(ch['layer2'], num_classes, 1)    # after fuse3: 512
        
        # Projections for contrastive learning
        self.img_proj = nn.Linear(ch['layer3'], 256)
        self.text_proj_cl = nn.Linear(self._text_dim, 256)
    
    def _encode_visual(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Encode with MedKLIP ResNet50 - no adapters, native channels."""
        features = self.visual_encoder(x)
        v0 = features['stem']    # [B, 64, H/2, W/2]
        v1 = features['layer1']  # [B, 256, H/4, W/4]
        v2 = features['layer2']  # [B, 512, H/8, W/8]
        v3 = features['layer3']  # [B, 1024, H/16, W/16]
        return v0, v1, v2, v3
    
    def _encode_text_clip(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Encode with CLIP text encoder."""
        tokens = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_len,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            out = self.lang_enc(**tokens)
        
        return out.last_hidden_state  # [B, N, 512]
    
    def _encode_text_medklip(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Encode with MedKLIP text encoder (PubMedBERT)."""
        if self.use_entity_translation:
            texts = [translate_entity(t) if len(t.split()) <= 3 else t for t in texts]
        
        token_emb, _ = self.text_encoder(texts, device)
        token_emb = self.text_proj(token_emb)  # [B, N, 512]
        return token_emb
    
    def _apply_text_mask(self, token_emb: torch.Tensor) -> torch.Tensor:
        """Apply random masking to text tokens during training."""
        if not self.training or self.text_mask_rate <= 0:
            return token_emb
        
        B, N, D = token_emb.shape
        mask = torch.rand(B, N, device=token_emb.device) > self.text_mask_rate
        mask[:, 0] = True
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
        """Forward pass."""
        device = images.device
        
        # Encode text
        l = None
        if texts is not None:
            if self.text_encoder_type == "medklip":
                l = self._encode_text_medklip(texts, device)
            else:
                l = self._encode_text_clip(texts, device)
            l = self._apply_text_mask(l)
        
        # Encode visual (native ResNet channels)
        v0, v1, v2, v3 = self._encode_visual(images)
        
        v3 = self._apply_img_mask(v3)
        
        # Cross-modal fusion + decoder
        if l is not None:
            f3 = self.fuse4(v3, l)       # [B, 1024, 14, 14]
            up3 = self.up3(v2, f3)       # [B, 512, 28, 28]
            f2 = self.fuse3(up3, l)      # [B, 512, 28, 28]
            up2 = self.up2(v1, f2)       # [B, 256, 56, 56]
            f1 = self.fuse2(up2, l)      # [B, 256, 56, 56]
            up1 = self.up1(v0, f1)       # [B, 64, 112, 112]
        else:
            up3 = self.up3(v2, v3)
            up2 = self.up2(v1, up3)
            up1 = self.up1(v0, up2)
            f1, f2, f3 = up2, up3, v3
        
        up0 = self.final_up(up1)         # [B, 64, 224, 224]
        
        pred = self.out0(up0)
        outputs = {'pred': pred}

        if self.deep_supervision and self.training:
            outputs['aux_preds'] = [
                self.out1(up1),   # 64ch
                self.out2(f1),    # 256ch
                self.out3(f2),    # 512ch
            ]
        
        return outputs

    def get_contrastive_embeddings(
        self, 
        images: torch.Tensor, 
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For CaCL pretraining."""
        device = images.device
        
        # Encode image
        _, _, _, v3 = self._encode_visual(images)
        v_global = F.adaptive_avg_pool2d(v3, 1).flatten(1)
        
        # Encode text
        if self.text_encoder_type == "medklip":
            token_emb = self._encode_text_medklip(texts, device)
            sent_emb = token_emb[:, 0, :]  # CLS token
        else:
            token_emb = self._encode_text_clip(texts, device)
            sent_emb = token_emb.mean(dim=1)
        
        v = F.normalize(self.img_proj(v_global), dim=-1)
        l = F.normalize(self.text_proj_cl(sent_emb), dim=-1)
        l_raw = F.normalize(sent_emb.detach().float(), dim=-1)
        
        return v, l, l_raw


def build_iris_ablation_native(config) -> IRISAblationNative:
    """
    Build IRIS ablation native model from config.
    
    Uses native ResNet50 channels without adapters.
    Only supports MedKLIP visual encoder.
    
    Config options:
      model:
        backbone: "ablation_native"
        text_encoder: "clip" | "medklip"
        fusion_type: "bcam" | "bifmd"
    """
    mcfg = config.get('model', config)
    
    return IRISAblationNative(
        num_classes=mcfg.get('num_classes', 1),
        img_size=config.get('data', {}).get('image_size', 224),
        text_encoder_type=mcfg.get('text_encoder', 'clip'),
        fusion_type=mcfg.get('fusion_type', 'bcam'),
        clip_model=mcfg.get('clip_model', 'openai/clip-vit-base-patch32'),
        medklip_visual_path=mcfg.get('medklip_visual_path', None),
        medklip_text_model=mcfg.get('medklip_text_model',
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
        max_token_len=mcfg.get('max_token_len', 77),
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


# =============================================================================
# IRISAblationWithAdapter - Frozen MedKLIP encoder with trainable adapters
# =============================================================================

class IRISAblationWithAdapter(nn.Module):
    """
    IRIS with frozen MedKLIP visual encoder and trainable adapters.
    
    The visual encoder is completely frozen. Lightweight adapters are added
    after each encoder stage to adapt features for the segmentation task.
    
    This approach:
    - Preserves pretrained MedKLIP representations
    - Adds task-specific adaptation with minimal parameters
    - Reduces overfitting compared to full fine-tuning
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        text_encoder_type: "clip" or "medklip"
        fusion_type: "bcam" or "bifmd"
        adapter_type: "bottleneck", "spatial", or "multiscale"
        adapter_reduction: Channel reduction ratio for adapters
        adapter_scale: Initial scale factor for adapter outputs
        ... (other args same as IRISAblation)
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        img_size: int = 224,
        text_encoder_type: str = "clip",
        fusion_type: str = "bcam",
        adapter_type: str = "bottleneck",
        adapter_reduction: int = 4,
        adapter_scale: float = 0.1,
        clip_model: str = "openai/clip-vit-base-patch32",
        medklip_visual_path: Optional[str] = None,
        medklip_text_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        max_token_len: int = 77,
        base_channel: int = 64,
        num_heads: int = 8,
        dropout: float = 0.5,
        expansion: int = 4,
        deep_supervision: bool = True,
        freeze_text: bool = True,
        use_entity_translation: bool = True,
        text_mask_rate: float = 0.3,
        img_mask_rate: float = 0.3,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.deep_supervision = deep_supervision
        self.text_mask_rate = text_mask_rate
        self.img_mask_rate = img_mask_rate
        self.use_entity_translation = use_entity_translation and text_encoder_type == 'medklip'
        
        # Import adapter factory
        from .adapters import get_adapter
        
        # =====================================================================
        # Visual Encoder: MedKLIP ResNet50 (ALWAYS FROZEN)
        # =====================================================================
        self.visual_encoder = MedKLIPVisualEncoder(
            pretrained_path=medklip_visual_path,
            freeze=True,  # Always frozen when using adapters
        )
        
        # ResNet50 channel dimensions
        ch = {'stem': 64, 'layer1': 256, 'layer2': 512, 'layer3': 1024}
        
        # =====================================================================
        # Visual Adapters (trainable)
        # =====================================================================
        self.adapt_stem = get_adapter(adapter_type, ch['stem'], adapter_reduction, adapter_scale)
        self.adapt_l1 = get_adapter(adapter_type, ch['layer1'], adapter_reduction, adapter_scale)
        self.adapt_l2 = get_adapter(adapter_type, ch['layer2'], adapter_reduction, adapter_scale)
        self.adapt_l3 = get_adapter(adapter_type, ch['layer3'], adapter_reduction, adapter_scale)
        
        # Channel adapters to match decoder scale
        self.ch_adapt_stem = nn.Conv2d(ch['stem'], base_channel, kernel_size=1)
        self.ch_adapt_l1 = nn.Conv2d(ch['layer1'], base_channel * 2, kernel_size=1)
        self.ch_adapt_l2 = nn.Conv2d(ch['layer2'], base_channel * 4, kernel_size=1)
        self.ch_adapt_l3 = nn.Conv2d(ch['layer3'], base_channel * 8, kernel_size=1)
        
        # =====================================================================
        # Text Encoder
        # =====================================================================
        if text_encoder_type == 'medklip':
            from transformers import AutoTokenizer, AutoModel
            self.text_encoder = MedKLIPTextEncoder(
                model_name=medklip_text_model,
                max_length=max_token_len,
                freeze=freeze_text,
            )
            self.text_proj = nn.Linear(self.text_encoder.hidden_size, 512)
            self._text_dim = 512
        else:
            from transformers import CLIPTokenizer, CLIPTextModel
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
            self.text_encoder = CLIPTextModel.from_pretrained(clip_model)
            if freeze_text:
                for p in self.text_encoder.parameters():
                    p.requires_grad = False
            self.text_proj = None
            self._text_dim = 512
        
        self._text_encoder_type = text_encoder_type
        
        # =====================================================================
        # Fusion Modules
        # =====================================================================
        img_scales = [img_size, img_size//2, img_size//4, img_size//8, img_size//16]
        
        if fusion_type == "bifmd":
            self.fuse2 = BiFMD(channels=base_channel * 2, text_dim=self._text_dim, num_heads=num_heads, expansion_ratio=expansion)
            self.fuse3 = BiFMD(channels=base_channel * 4, text_dim=self._text_dim, num_heads=num_heads, expansion_ratio=expansion)
            self.fuse4 = BiFMD(channels=base_channel * 8, text_dim=self._text_dim, num_heads=num_heads, expansion_ratio=expansion)
        else:
            self.fuse2 = BCAModuleMedKLIP(img_scales[2], base_channel * 2, self._text_dim, num_heads, dropout, expansion)
            self.fuse3 = BCAModuleMedKLIP(img_scales[3], base_channel * 4, self._text_dim, num_heads, dropout, expansion)
            self.fuse4 = BCAModuleMedKLIP(img_scales[4], base_channel * 8, self._text_dim, num_heads, dropout, expansion)
        
        self._fusion_type = fusion_type
        
        # =====================================================================
        # Decoder
        # =====================================================================
        self.up4 = UpBlockMedKLIP(base_channel * 4, base_channel * 8, base_channel * 4)
        self.up3 = UpBlockMedKLIP(base_channel * 2, base_channel * 4, base_channel * 2)
        self.up2 = UpBlockMedKLIP(base_channel, base_channel * 2, base_channel)
        
        # Final prediction
        self.final = nn.Conv2d(base_channel, num_classes, kernel_size=1)
        
        # Deep supervision heads
        if deep_supervision:
            self.aux_head4 = nn.Conv2d(base_channel * 8, num_classes, kernel_size=1)
            self.aux_head3 = nn.Conv2d(base_channel * 4, num_classes, kernel_size=1)
            self.aux_head2 = nn.Conv2d(base_channel * 2, num_classes, kernel_size=1)
            self.aux_head1 = nn.Conv2d(base_channel, num_classes, kernel_size=1)
        
        # =====================================================================
        # Contrastive Learning projections
        # =====================================================================
        self.text_proj_gate = nn.Linear(self._text_dim, base_channel * 8)
        self.img_proj = nn.Linear(base_channel * 8, 256)
        self.text_proj_cl = nn.Linear(self._text_dim, 256)
    
    def _encode_visual(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Encode image through frozen MedKLIP + adapters."""
        features = self.visual_encoder(x)
        
        # Apply adapters then channel projection
        f0 = self.ch_adapt_stem(self.adapt_stem(features['stem']))
        f1 = self.ch_adapt_l1(self.adapt_l1(features['layer1']))
        f2 = self.ch_adapt_l2(self.adapt_l2(features['layer2']))
        f3 = self.ch_adapt_l3(self.adapt_l3(features['layer3']))
        
        return f0, f1, f2, f3
    
    def _encode_text(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text using selected encoder."""
        device = next(self.parameters()).device
        
        if self._text_encoder_type == 'medklip':
            if self.use_entity_translation:
                texts = [translate_entity(t) for t in texts]
            text_output = self.text_encoder(texts)
            token_emb = self.text_proj(text_output['last_hidden_state'])
            sentence_emb = self.text_proj(text_output['pooler_output'])
        else:
            tokens = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors='pt'
            ).to(device)
            
            outputs = self.text_encoder(**tokens)
            token_emb = outputs.last_hidden_state
            sentence_emb = token_emb[torch.arange(len(texts)), tokens.attention_mask.sum(1) - 1]
        
        return token_emb, sentence_emb
    
    def forward(self, images: torch.Tensor, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Encode
        f0, f1, f2, f3 = self._encode_visual(images)
        token_emb, sentence_emb = self._encode_text(texts)
        
        # Apply text masking during training
        if self.training and self.text_mask_rate > 0:
            mask = torch.rand(token_emb.size(0), token_emb.size(1), 1, device=token_emb.device)
            token_emb = token_emb * (mask > self.text_mask_rate).float()
        
        # Fusion
        fused2 = self.fuse2(f1, token_emb)
        fused3 = self.fuse3(f2, token_emb)
        fused4 = self.fuse4(f3, token_emb)
        
        # Decoder
        d3 = self.up4(fused3, fused4)
        d2 = self.up3(fused2, d3)
        d1 = self.up2(f0, d2)
        
        # Upsample to original size
        pred = F.interpolate(self.final(d1), size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
        
        outputs = {'pred': pred}
        
        if self.deep_supervision:
            aux4 = F.interpolate(self.aux_head4(fused4), size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
            aux3 = F.interpolate(self.aux_head3(d3), size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
            aux2 = F.interpolate(self.aux_head2(d2), size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
            aux1 = F.interpolate(self.aux_head1(d1), size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
            outputs['aux_preds'] = [aux1, aux2, aux3, aux4]
        
        return outputs
    
    def get_contrastive_embeddings(
        self,
        images: torch.Tensor,
        texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get embeddings for contrastive learning."""
        _, _, _, f3 = self._encode_visual(images)
        token_emb, sentence_emb = self._encode_text(texts)
        
        # Global average pool visual features
        v = F.adaptive_avg_pool2d(f3, 1).flatten(1)
        v = F.normalize(self.img_proj(v), dim=-1)
        
        # Text embeddings
        l = F.normalize(self.text_proj_cl(sentence_emb), dim=-1)
        l_raw = F.normalize(sentence_emb, dim=-1)
        
        return v, l, l_raw


def build_iris_ablation_with_adapter(config) -> IRISAblationWithAdapter:
    """
    Build IRIS ablation model with adapters.
    
    Config options:
      model:
        backbone: "ablation_adapter"
        adapter_type: "bottleneck" | "spatial" | "multiscale"
        adapter_reduction: 4 (default)
        adapter_scale: 0.1 (default)
        text_encoder: "clip" | "medklip"
        fusion_type: "bcam" | "bifmd"
    """
    mcfg = config.get('model', config)
    
    return IRISAblationWithAdapter(
        num_classes=mcfg.get('num_classes', 1),
        img_size=config.get('data', {}).get('image_size', 224),
        text_encoder_type=mcfg.get('text_encoder', 'clip'),
        fusion_type=mcfg.get('fusion_type', 'bcam'),
        adapter_type=mcfg.get('adapter_type', 'bottleneck'),
        adapter_reduction=mcfg.get('adapter_reduction', 4),
        adapter_scale=mcfg.get('adapter_scale', 0.1),
        clip_model=mcfg.get('clip_model', 'openai/clip-vit-base-patch32'),
        medklip_visual_path=mcfg.get('medklip_visual_path', None),
        medklip_text_model=mcfg.get('medklip_text_model',
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
        max_token_len=mcfg.get('max_token_len', 77),
        base_channel=mcfg.get('base_channel', 64),
        num_heads=mcfg.get('num_heads', 8),
        dropout=mcfg.get('dropout', 0.5),
        expansion=mcfg.get('expansion', 4),
        deep_supervision=mcfg.get('deep_supervision', True),
        freeze_text=mcfg.get('freeze_medklip_text', True),
        use_entity_translation=mcfg.get('use_entity_translation', True),
        text_mask_rate=mcfg.get('text_mask_rate', 0.3),
        img_mask_rate=mcfg.get('img_mask_rate', 0.3),
    )
