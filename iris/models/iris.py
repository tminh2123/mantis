"""
IRIS: Medical Image Segmentation
Baseline version - MLP only (no KAN), ported from C2Seg

Architecture:
  - Visual Encoder: 5 DownBlocks (Conv only)
  - BCAM: Bidirectional Cross Attention with Sinusoidal2D + MLPGate
  - Decoder: UpBlockAttention with PixLevelModule
  - Deep Supervision: 5 output heads at different scales
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# MLP Layer for gating (Transformer FFN style with expansion)
# ---------------------------------------------------------------------------
class MLPGate(nn.Module):
    """
    MLP-based gating layer with expansion ratio.
    
    Pattern: C → expansion*C → C (Inverted Bottleneck / Transformer FFN)
    This allows the network to learn richer intermediate representations.
    """
    def __init__(self, in_dim: int, out_dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = in_dim * expansion
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),  # GELU is common in Transformers
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B, N, C = x.shape
            x = x.reshape(B * N, C)
            x = self.mlp(x)
            x = x.reshape(B, N, -1)
            return x
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Sinusoidal 2D Position Embedding
# ---------------------------------------------------------------------------
class Sinusoidal2DPositionEmbedding(nn.Module):
    """2D sinusoidal position embedding for image features."""
    
    def __init__(self, num_channels: int, height: int, width: int):
        super().__init__()
        assert num_channels % 4 == 0, "Embedding dim must be divisible by 4"
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

        pe[0::4, :, :] = torch.sin(y_pos[None, :, :] * div_term[:, None, None])
        pe[1::4, :, :] = torch.cos(y_pos[None, :, :] * div_term[:, None, None])
        pe[2::4, :, :] = torch.sin(x_pos[None, :, :] * div_term[:, None, None])
        pe[3::4, :, :] = torch.cos(x_pos[None, :, :] * div_term[:, None, None])

        return pe.unsqueeze(0)  # [1, C, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_encoding


# ---------------------------------------------------------------------------
# Pixel Level Attention Module
# ---------------------------------------------------------------------------
class PixLevelModule(nn.Module):
    """Pixel-Level Attention Module for skip connections."""
    
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
        avg_feat = torch.mean(self.conv_avg(x), dim=1, keepdim=True)
        max_feat = torch.max(self.conv_max(x), dim=1, keepdim=True).values

        combined = torch.cat([avg_feat, max_feat, avg_feat + max_feat], dim=1)
        out = combined.permute(0, 2, 3, 1)
        out = self.bottleneck(out)
        out = out.permute(0, 3, 1, 2)

        return x * out


# ---------------------------------------------------------------------------
# MLP-based Channel Gate (replacement for KCGate)
# ---------------------------------------------------------------------------
class ChannelGate(nn.Module):
    """
    MLP-based Channel Gate with expansion.
    g = tanh(MLP(ReLU(MLP(x))))
    Output: x * g (element-wise gating)
    
    Uses Transformer FFN style: C → 4C → C for richer representations.
    """
    def __init__(self, in_channels: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.gate1 = MLPGate(in_channels, in_channels, expansion, dropout)
        self.gate2 = MLPGate(in_channels, in_channels, expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] or [B, HW, C]
        Returns:
            gated output [B, N, C]
        """
        g = self.gate1(x)
        g = F.relu(g)
        g = self.gate2(g)
        g = torch.tanh(g)
        return x * g


# ---------------------------------------------------------------------------
# Gate Module: MLP-based Attention Gating
# ---------------------------------------------------------------------------
class GateModule(nn.Module):
    """
    MLP-based Attention Gating module (replaces K-Gate).
    
    Architecture:
        F_l → MLP(expand) → ReLU → MLP(expand) → Tanh → ⊙ F_l → a_l
        F_v → MLP(expand) → ReLU → MLP(expand) → Tanh → ⊙ F_v → a_v
        Concat(a_l, a_v) → Conv → F_out
    """
    def __init__(self, in_channels: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.gate_l = ChannelGate(in_channels, expansion, dropout)
        self.gate_v = ChannelGate(in_channels, expansion, dropout)
        
        # Concat (2*C) → Conv → C
        self.fuse_conv = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, kernel_size=1),
            nn.InstanceNorm1d(in_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, f_l: torch.Tensor, f_v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_l: language features [B, HW, C]
            f_v: vision features [B, HW, C]
        Returns:
            fused features [B, C, HW]
        """
        a_l = self.gate_l(f_l)  # [B, HW, C]
        a_v = self.gate_v(f_v)  # [B, HW, C]
        
        # Concat and fuse
        concat = torch.cat([a_l, a_v], dim=-1)  # [B, HW, 2C]
        concat = concat.permute(0, 2, 1)         # [B, 2C, HW]
        out = self.fuse_conv(concat)             # [B, C, HW]
        
        return out


# ---------------------------------------------------------------------------
# BCAM: Bidirectional Cross Attention Module
# ---------------------------------------------------------------------------
class BCAModule(nn.Module):
    """
    Bidirectional Cross Attention (BCA) module.
    Uses MLP Gate with expansion for fusion.
    """
    def __init__(
        self, 
        img_size: int, 
        in_channels: int, 
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
        self.l_key = self._make_conv(512, in_channels)  # CLIP text dim = 512
        self.l_value = self._make_conv(512, in_channels)

        # MLP Gate module with expansion
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
            text_feat: [B, N, 512]
        Returns:
            Fused features [B, C, H, W]
        """
        b, c, h, w = img_feat.shape
        hw = h * w
        n_t = text_feat.size(1)

        # Flatten image features and add position encoding
        img_flat = img_feat.flatten(2)  # [B, C, HW]
        v = self.pos_embed(img_feat).flatten(2)  # [B, C, HW]
        l = text_feat.transpose(1, 2)  # [B, 512, N]

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


# ---------------------------------------------------------------------------
# Basic Conv blocks
# ---------------------------------------------------------------------------
class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


def conv_stack(in_channels: int, out_channels: int, num_layers: int) -> nn.Sequential:
    """Stack of Conv-BN-ReLU layers"""
    layers = [ConvBNReLU(in_channels, out_channels)]
    layers += [ConvBNReLU(out_channels, out_channels) for _ in range(num_layers - 1)]
    return nn.Sequential(*layers)


class DownBlock(nn.Module):
    """Down-sampling block with stacked conv layers (Full CNN)"""
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2):
        super().__init__()
        self.convs = conv_stack(in_channels, out_channels, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class DownBlockMix(nn.Module):
    """
    Down-sampling block with Conv + MLP (Mixed style).
    
    Architecture: Conv → Conv → MLP (channel mixing)
    Similar to ConvNeXt / MetaFormer pattern.
    """
    def __init__(self, in_channels: int, out_channels: int, expansion: int = 4):
        super().__init__()
        # Spatial mixing (Conv)
        self.conv1 = ConvBNReLU(in_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)
        
        # Channel mixing (MLP with expansion)
        self.norm = nn.BatchNorm2d(out_channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * expansion, 1),  # Expand
            nn.GELU(),
            nn.Conv2d(out_channels * expansion, out_channels, 1),  # Project
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        # MLP with residual
        x = x + self.mlp(self.norm(x))
        return x


class UpBlockAttention(nn.Module):
    """Up-sampling block with PixLevelModule attention on skip connection"""
    def __init__(self, skip_channels: int, up_channels: int, out_channels: int, num_layers: int = 2):
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


# ---------------------------------------------------------------------------
# Language Encoder (frozen CLIP)
# ---------------------------------------------------------------------------
class LanguageEncoder(nn.Module):
    """Frozen CLIP text encoder."""
    
    def __init__(self, clip_model: str = "openai/clip-vit-base-patch32", max_length: int = 18):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
        self.encoder = CLIPTextModel.from_pretrained(clip_model)
        self.max_length = max_length
        self.hidden_size = 512  # CLIP text dim
        # Freeze
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, texts: List[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        ).to(device)
        out = self.encoder(**tokens)
        token_emb = out.last_hidden_state  # [B, N, 512]
        sent_emb = out.pooler_output       # [B, 512]
        return token_emb, sent_emb


class BERTLanguageEncoder(nn.Module):
    """BERT-based text encoder (ClinicalBERT, PubMedBERT, etc.)."""
    
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", max_length: int = 18):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.hidden_size = self.encoder.config.hidden_size  # 768 for BERT
        # Freeze
        for p in self.encoder.parameters():
            p.requires_grad_(False)
    
    def forward(self, texts: List[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt"
        ).to(device)
        out = self.encoder(**tokens)
        token_emb = out.last_hidden_state  # [B, N, 768]
        sent_emb = out.pooler_output       # [B, 768]
        return token_emb, sent_emb


# ---------------------------------------------------------------------------
# IRIS Model
# ---------------------------------------------------------------------------
class IRIS(nn.Module):
    """
    IRIS: Medical Image Segmentation
    
    Baseline version with MLP-only gating (no KAN).
    Architecture follows C2Seg with:
      - Conv Encoder (5 DownBlocks) - can be CNN or Mix
      - BCAM Fusion (4 scales)
      - Conv Decoder with PixLevelModule
      - Deep Supervision (5 output heads)
    
    MLP uses Transformer FFN style: C → expansion*C → C
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 224,
        clip_model: str = "openai/clip-vit-base-patch32",
        text_encoder_type: str = "clip",  # "clip" or "bert"
        text_model: str = None,  # Override text model (e.g., ClinicalBERT)
        max_token_len: int = 18,
        base_channel: int = 64,
        num_heads: int = 8,
        dropout: float = 0.5,
        expansion: int = 4,  # MLP expansion ratio
        encoder_type: str = "cnn",  # "cnn" or "mix"
        deep_supervision: bool = True,
        text_mask_rate: float = 0.3,
        img_mask_rate: float = 0.3,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.text_mask_rate = text_mask_rate
        self.img_mask_rate = img_mask_rate
        self.base_ch = base_channel
        self.encoder_type = encoder_type
        self.text_encoder_type = text_encoder_type
        
        img_scales = [img_size, img_size//2, img_size//4, img_size//8, img_size//16]

        # Select encoder block type
        if encoder_type == "mix":
            DownBlockClass = lambda in_ch, out_ch: DownBlockMix(in_ch, out_ch, expansion)
        else:  # "cnn" (default)
            DownBlockClass = DownBlock

        # Vision encoder
        self.pool = nn.MaxPool2d(2, 2)
        self.down0 = DownBlockClass(in_channels, base_channel)
        self.down1 = DownBlockClass(base_channel, base_channel * 2)
        self.down2 = DownBlockClass(base_channel * 2, base_channel * 4)
        self.down3 = DownBlockClass(base_channel * 4, base_channel * 8)
        self.down4 = DownBlockClass(base_channel * 8, base_channel * 8)

        # Vision decoder with PixLevelModule
        self.up3 = UpBlockAttention(base_channel * 8, base_channel * 8, base_channel * 4)
        self.up2 = UpBlockAttention(base_channel * 4, base_channel * 4, base_channel * 2)
        self.up1 = UpBlockAttention(base_channel * 2, base_channel * 2, base_channel)
        self.up0 = UpBlockAttention(base_channel, base_channel, base_channel)

        # Language encoder
        if text_encoder_type == "bert":
            bert_model = text_model or "emilyalsentzer/Bio_ClinicalBERT"
            self.lang_enc = BERTLanguageEncoder(bert_model, max_length=max_token_len)
            text_dim = self.lang_enc.hidden_size  # 768
            # Project BERT dim to match CLIP dim for fusion
            self.text_proj_fusion = nn.Linear(768, 512)
        else:
            self.lang_enc = LanguageEncoder(clip_model, max_length=max_token_len)
            text_dim = 512
            self.text_proj_fusion = None

        # Cross-modal fusion (BCAM at 4 scales) with expansion
        # BCAM expects 512-dim text features
        self.fuse1 = BCAModule(img_scales[1], base_channel, num_heads, dropout, expansion)
        self.fuse2 = BCAModule(img_scales[2], base_channel * 2, num_heads, dropout, expansion)
        self.fuse3 = BCAModule(img_scales[3], base_channel * 4, num_heads, dropout, expansion)
        self.fuse4 = BCAModule(img_scales[4], base_channel * 8, num_heads, dropout, expansion)

        # Output heads
        self.out0 = nn.Conv2d(base_channel, num_classes, 1)
        
        if self.deep_supervision:
            self.out1 = nn.Conv2d(base_channel, num_classes, 1)
            self.out2 = nn.Conv2d(base_channel * 2, num_classes, 1)
            self.out3 = nn.Conv2d(base_channel * 4, num_classes, 1)
            self.out4 = nn.Conv2d(base_channel * 8, num_classes, 1)

        # Projections for CaCL pretraining
        self.img_proj = nn.Linear(base_channel * 8, 256)
        self.text_proj = nn.Linear(512, 256)

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
        # Encode text
        l = None
        if texts is not None:
            token_emb, _ = self.lang_enc(texts, images.device)
            # Project BERT features to 512 dim if needed
            if self.text_proj_fusion is not None:
                token_emb = self.text_proj_fusion(token_emb)
            l = self._apply_text_mask(token_emb)

        # Encode image
        v0 = self.down0(images)              # [B, C, 224, 224]
        v1 = self.down1(self.pool(v0))       # [B, 2C, 112, 112]
        v2 = self.down2(self.pool(v1))       # [B, 4C, 56, 56]
        v3 = self.down3(self.pool(v2))       # [B, 8C, 28, 28]
        v4 = self.down4(self.pool(v3))       # [B, 8C, 14, 14]
        
        # Apply image masking
        v4 = self._apply_img_mask(v4)

        # Cross-modal fusion + decoder
        if l is not None:
            f4 = self.fuse4(v4, l)
            
            up3 = self.up3(v3, f4)
            f3 = self.fuse3(up3, l)
            
            up2 = self.up2(v2, f3)
            f2 = self.fuse2(up2, l)
            
            up1 = self.up1(v1, f2)
            f1 = self.fuse1(up1, l)
            
            up0 = self.up0(v0, f1)
        else:
            # No text fusion
            up3 = self.up3(v3, v4)
            up2 = self.up2(v2, up3)
            up1 = self.up1(v1, up2)
            up0 = self.up0(v0, up1)
            f1, f2, f3, f4 = up1, up2, up3, v4

        # Output
        pred = self.out0(up0)
        outputs = {'pred': pred}

        if self.deep_supervision and self.training:
            outputs['aux_preds'] = [
                self.out1(f1),
                self.out2(f2),
                self.out3(f3),
                self.out4(f4),
            ]
        
        return outputs

    def get_contrastive_embeddings(
        self, 
        images: torch.Tensor, 
        texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """For CaCL pretraining."""
        # Forward through encoder only
        v0 = self.down0(images)
        v1 = self.down1(self.pool(v0))
        v2 = self.down2(self.pool(v1))
        v3 = self.down3(self.pool(v2))
        v4 = self.down4(self.pool(v3))

        # Global image vector
        v_global = F.adaptive_avg_pool2d(v4, 1).flatten(1)

        with torch.no_grad():
            _, sent_emb = self.lang_enc(texts, images.device)

        v = F.normalize(self.img_proj(v_global), dim=-1)
        l = F.normalize(self.text_proj(sent_emb), dim=-1)
        l_raw = F.normalize(sent_emb.detach().float(), dim=-1)
        
        return v, l, l_raw


def build_iris(config) -> IRIS:
    """Build IRIS model from config."""
    return IRIS(
        num_classes=config.model.num_classes,
        img_size=config.data.image_size,
        clip_model=config.model.clip_model,
        max_token_len=config.model.max_token_len,
        base_channel=config.model.base_channel,
        num_heads=config.model.get('num_heads', 8),
        dropout=config.model.dropout,
        deep_supervision=config.model.deep_supervision,
        text_mask_rate=config.model.text_mask_rate,
        img_mask_rate=config.model.img_mask_rate,
    )
