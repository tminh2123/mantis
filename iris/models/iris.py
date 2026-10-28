"""
IRIS: Unified Medical Image Segmentation Model

All encoder/decoder options in one class:
  - Visual: "cnn" (5-level Conv) | "resnet" (MedKLIP ResNet50)
  - Text: "clip" | "pubmedbert" | "clinicalbert"
  - Fusion: "bcam" (default) | "bifmd"
  - Adapters: None | "bottleneck" | "spatial" | "multiscale"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import (
    CLIPTextModel, CLIPTokenizer,
    AutoModel, AutoTokenizer,
)
from typing import List, Tuple, Optional, Dict


# =============================================================================
# Building Blocks
# =============================================================================

class MLPGate(nn.Module):
    """MLP gating with expansion (Transformer FFN style)."""
    def __init__(self, in_dim: int, out_dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = in_dim * expansion
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B, N, C = x.shape
            return self.mlp(x.reshape(B * N, C)).reshape(B, N, -1)
        return self.mlp(x)


class Sinusoidal2DPositionEmbedding(nn.Module):
    """2D sinusoidal position embedding."""
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        assert channels % 4 == 0
        self.register_buffer("pe", self._build(channels, height, width))

    def _build(self, C, H, W) -> torch.Tensor:
        pe = torch.zeros(C, H, W)
        y = torch.arange(H, dtype=torch.float32).unsqueeze(1).repeat(1, W)
        x = torch.arange(W, dtype=torch.float32).unsqueeze(0).repeat(H, 1)
        div = torch.exp(torch.arange(0, C // 2, 2).float() * -(math.log(10000.0) / (C // 2)))
        pe[0::4] = torch.sin(y[None] * div[:, None, None])
        pe[1::4] = torch.cos(y[None] * div[:, None, None])
        pe[2::4] = torch.sin(x[None] * div[:, None, None])
        pe[3::4] = torch.cos(x[None] * div[:, None, None])
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class PixLevelModule(nn.Module):
    """Pixel-level attention for skip connections."""
    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        self.conv_avg = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(True)
        )
        self.conv_max = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(True)
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * reduction), nn.ReLU(True), nn.Linear(3 * reduction, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(self.conv_avg(x), dim=1, keepdim=True)
        mx = torch.max(self.conv_max(x), dim=1, keepdim=True).values
        combined = torch.cat([avg, mx, avg + mx], dim=1).permute(0, 2, 3, 1)
        out = self.bottleneck(combined).permute(0, 3, 1, 2)
        return x * out


class ChannelGate(nn.Module):
    """Channel gating with tanh activation."""
    def __init__(self, channels: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.g1 = MLPGate(channels, channels, expansion, dropout)
        self.g2 = MLPGate(channels, channels, expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(self.g2(F.relu(self.g1(x))))


class GateModule(nn.Module):
    """Dual-path gating for vision-language fusion."""
    def __init__(self, channels: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.gate_l = ChannelGate(channels, expansion, dropout)
        self.gate_v = ChannelGate(channels, expansion, dropout)
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1),
            nn.InstanceNorm1d(channels), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, f_l: torch.Tensor, f_v: torch.Tensor) -> torch.Tensor:
        a_l, a_v = self.gate_l(f_l), self.gate_v(f_v)
        return self.fuse(torch.cat([a_l, a_v], dim=-1).permute(0, 2, 1))


# =============================================================================
# Fusion Modules
# =============================================================================

class BCAModule(nn.Module):
    """Bidirectional Cross Attention Module."""
    def __init__(self, img_size: int, channels: int, text_dim: int = 512,
                 num_heads: int = 8, dropout: float = 0.0, expansion: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.pos = Sinusoidal2DPositionEmbedding(channels, img_size, img_size)
        
        self.v_key = nn.Sequential(nn.Conv1d(channels, channels, 1), nn.InstanceNorm1d(channels))
        self.v_val = nn.Sequential(nn.Conv1d(channels, channels, 1), nn.InstanceNorm1d(channels))
        self.l_key = nn.Sequential(nn.Conv1d(text_dim, channels, 1), nn.InstanceNorm1d(channels))
        self.l_val = nn.Sequential(nn.Conv1d(text_dim, channels, 1), nn.InstanceNorm1d(channels))
        
        self.gate = GateModule(channels, expansion, dropout)
        self.proj = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.InstanceNorm1d(channels),
            nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, img: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        HW, N = H * W, text.size(1)
        
        img_flat = img.flatten(2)
        v = self.pos(img).flatten(2)
        l = text.transpose(1, 2)
        
        v_k, v_v = self.v_key(v), self.v_val(v)
        l_k, l_v = self.l_key(l), self.l_val(l)
        
        hd = C // self.num_heads
        def reshape_v(t): return t.view(B, self.num_heads, hd, HW).permute(0, 1, 3, 2)
        def reshape_l(t): return t.view(B, self.num_heads, hd, N)
        
        v_k, v_v = reshape_v(v_k), reshape_v(v_v)
        l_k = reshape_l(l_k)
        l_v = reshape_l(l_v).permute(0, 1, 3, 2)
        
        sim = (v_k @ l_k) * (C ** -0.5)
        attn_v2l = F.softmax(sim, dim=-1)
        attn_l2v = F.softmax(sim.transpose(-1, -2), dim=-1)
        
        f_v = (attn_v2l @ l_v).permute(0, 2, 1, 3).reshape(B, HW, C)
        f_l = torch.einsum("bhij,bhjk->bhjk", attn_l2v, v_v).permute(0, 2, 1, 3).reshape(B, HW, C)
        
        fused = self.gate(f_l, f_v)
        return self.proj(img_flat + fused).view(B, C, H, W)


class BiFMD(nn.Module):
    """Bidirectional FiLM-Modulated Decoder."""
    def __init__(self, channels: int, text_dim: int, num_heads: int = 4, expansion_ratio: int = 4):
        super().__init__()
        self.channels, self.text_dim = channels, text_dim
        self.num_heads = num_heads
        hd = channels // num_heads
        
        self.t2i = nn.Linear(text_dim, 2 * channels)
        self.i2t = nn.Linear(channels, 2 * text_dim)
        nn.init.zeros_(self.t2i.weight); nn.init.zeros_(self.t2i.bias)
        nn.init.zeros_(self.i2t.weight); nn.init.zeros_(self.i2t.bias)
        
        self.img_q = nn.Conv2d(channels, channels, 1)
        self.img_v = nn.Conv2d(channels, channels, 1)
        self.txt_k = nn.Linear(text_dim, channels)
        self.txt_v = nn.Linear(text_dim, channels)
        
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.l_proj = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Conv2d(2 * channels, channels, 3, padding=1)
        self.fuse = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        
        hidden = channels * expansion_ratio
        self.refine = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)

    def forward(self, img: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        N = text.size(1)
        P = H * W
        h, d = self.num_heads, C // self.num_heads
        
        pooled_t = text.mean(1)
        pooled_i = img.mean(dim=(2, 3))
        
        dg, beta = self.t2i(pooled_t).chunk(2, dim=-1)
        img_mod = img * (1 + dg[..., None, None]) + beta[..., None, None]
        
        dg2, beta2 = self.i2t(pooled_i).chunk(2, dim=-1)
        text_mod = text * (1 + dg2[:, None]) + beta2[:, None]
        
        q = self.img_q(img_mod).flatten(2).transpose(1, 2).reshape(B, P, h, d).transpose(1, 2)
        v_img = self.img_v(img_mod).flatten(2).transpose(1, 2).reshape(B, P, h, d).transpose(1, 2)
        k_txt = self.txt_k(text_mod).reshape(B, N, h, d).transpose(1, 2)
        v_txt = self.txt_v(text_mod).reshape(B, N, h, d).transpose(1, 2)
        
        scores = (q @ k_txt.transpose(-2, -1)) / math.sqrt(d)
        alpha_t = F.softmax(scores, dim=-1)
        alpha_s = F.softmax(scores, dim=-2).sum(dim=-1) / N
        
        f_v = (alpha_t @ v_txt).transpose(1, 2).reshape(B, P, C).transpose(1, 2).reshape(B, C, H, W)
        f_l = (v_img * alpha_s.unsqueeze(-1) * P).transpose(1, 2).reshape(B, P, C).transpose(1, 2).reshape(B, C, H, W)
        
        fv, fl = self.v_proj(f_v), self.l_proj(f_l)
        g = torch.sigmoid(self.gate(torch.cat([fv, fl], dim=1)))
        mix = self.fuse(g * fv + (1 - g) * fl)
        fused = self.norm(mix + img_mod)
        return fused + self.refine(fused)


# =============================================================================
# Encoder/Decoder Blocks
# =============================================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True)
        )
    def forward(self, x): return self.op(x)


class DownBlock(nn.Module):
    """CNN encoder block."""
    def __init__(self, in_ch: int, out_ch: int, num_layers: int = 2):
        super().__init__()
        layers = [ConvBNReLU(in_ch, out_ch)]
        layers += [ConvBNReLU(out_ch, out_ch) for _ in range(num_layers - 1)]
        self.convs = nn.Sequential(*layers)
    def forward(self, x): return self.convs(x)


class UpBlock(nn.Module):
    """Decoder block with PixLevelModule."""
    def __init__(self, skip_ch: int, up_ch: int, out_ch: int, num_layers: int = 2):
        super().__init__()
        self.att = PixLevelModule(skip_ch)
        self.up = nn.ConvTranspose2d(up_ch, up_ch, 2, stride=2)
        layers = [ConvBNReLU(skip_ch + up_ch, out_ch)]
        layers += [ConvBNReLU(out_ch, out_ch) for _ in range(num_layers - 1)]
        self.convs = nn.Sequential(*layers)

    def forward(self, skip: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        skip_att = self.att(skip)
        up_x = self.up(x)
        if up_x.shape[2:] != skip_att.shape[2:]:
            up_x = F.interpolate(up_x, skip_att.shape[2:], mode='bilinear', align_corners=False)
        return self.convs(torch.cat([skip_att, up_x], dim=1))


# =============================================================================
# Adapters
# =============================================================================

class BottleneckAdapter(nn.Module):
    """Lightweight adapter for frozen encoders."""
    def __init__(self, channels: int, reduction: int = 4, scale: float = 0.1):
        super().__init__()
        mid = max(channels // reduction, 64)
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True),
            nn.Conv2d(mid, channels, 1, bias=False), nn.BatchNorm2d(channels),
        )
        self.scale = nn.Parameter(torch.ones(1) * scale)
    def forward(self, x): return x + self.scale * self.adapter(x)


class SpatialAdapter(nn.Module):
    """Adapter with spatial 3x3 convolution."""
    def __init__(self, channels: int, reduction: int = 4, scale: float = 0.1):
        super().__init__()
        mid = max(channels // reduction, 64)
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True),
            nn.Conv2d(mid, channels, 1, bias=False), nn.BatchNorm2d(channels),
        )
        self.scale = nn.Parameter(torch.ones(1) * scale)
    def forward(self, x): return x + self.scale * self.adapter(x)


class MultiScaleAdapter(nn.Module):
    """Multi-branch adapter."""
    def __init__(self, channels: int, reduction: int = 4, scale: float = 0.1):
        super().__init__()
        mid = max(channels // reduction, 64)
        self.local = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 2, channels, 1, bias=False), nn.BatchNorm2d(channels)
        )
        self.scale = nn.Parameter(torch.ones(1) * scale)

    def forward(self, x):
        return x + self.scale * self.fuse(torch.cat([self.local(x), self.spatial(x)], dim=1))


def get_adapter(name: str, channels: int, reduction: int = 4, scale: float = 0.1):
    adapters = {"bottleneck": BottleneckAdapter, "spatial": SpatialAdapter, "multiscale": MultiScaleAdapter}
    return adapters[name](channels, reduction, scale) if name else nn.Identity()


# =============================================================================
# Text Encoders
# =============================================================================

class CLIPTextEncoder(nn.Module):
    """CLIP text encoder (frozen)."""
    def __init__(self, model: str = "openai/clip-vit-base-patch32", max_len: int = 18):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model)
        self.encoder = CLIPTextModel.from_pretrained(model)
        self.max_len = max_len
        self.hidden_size = 512
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, texts: List[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(texts, padding="max_length", truncation=True,
                                max_length=self.max_len, return_tensors="pt").to(device)
        out = self.encoder(**tokens)
        return out.last_hidden_state, out.pooler_output


class BERTTextEncoder(nn.Module):
    """BERT-based text encoder (ClinicalBERT, PubMedBERT)."""
    def __init__(self, model: str = "emilyalsentzer/Bio_ClinicalBERT", max_len: int = 18):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.encoder = AutoModel.from_pretrained(model)
        self.max_len = max_len
        self.hidden_size = self.encoder.config.hidden_size  # 768
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, texts: List[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(texts, padding="max_length", truncation=True,
                                max_length=self.max_len, return_tensors="pt").to(device)
        out = self.encoder(**tokens)
        # Use last non-padding token as sentence embedding (like MedKLIP)
        mask_sum = tokens.attention_mask.sum(1) - 1
        sent_emb = out.last_hidden_state[torch.arange(len(texts)), mask_sum]
        return out.last_hidden_state, sent_emb


# =============================================================================
# Visual Encoders
# =============================================================================

class CNNEncoder(nn.Module):
    """5-level CNN encoder (original IRIS)."""
    def __init__(self, base_ch: int = 64):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.down0 = DownBlock(3, base_ch)
        self.down1 = DownBlock(base_ch, base_ch * 2)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)
        self.down4 = DownBlock(base_ch * 8, base_ch * 8)
        self.channels = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 8]

    def forward(self, x) -> List[torch.Tensor]:
        v0 = self.down0(x)
        v1 = self.down1(self.pool(v0))
        v2 = self.down2(self.pool(v1))
        v3 = self.down3(self.pool(v2))
        v4 = self.down4(self.pool(v3))
        return [v0, v1, v2, v3, v4]


class ResNetEncoder(nn.Module):
    """ResNet50 encoder (MedKLIP style)."""
    def __init__(self, pretrained_path: Optional[str] = None, freeze: bool = False):
        super().__init__()
        resnet = models.resnet50(pretrained=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        # channels: stem=64, l1=256, l2=512, l3=1024
        self.channels = [64, 256, 512, 1024]
        
        if pretrained_path:
            self._load_weights(pretrained_path)
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def _load_weights(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('model', ckpt)
        new_state = {}
        for k, v in state.items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('res_features.'):
                idx = int(k.split('.')[1])
                rest = '.'.join(k.split('.')[2:])
                mapping = {0: 'conv1', 1: 'bn1', 4: 'layer1', 5: 'layer2', 6: 'layer3'}
                if idx in mapping and rest:
                    new_state[f'{mapping[idx]}.{rest}'] = v
        self.load_state_dict(new_state, strict=False)
        print(f"Loaded ResNet weights from {path}")

    def forward(self, x) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        stem = x  # 64, H/2
        x = self.maxpool(x)
        l1 = self.layer1(x)   # 256, H/4
        l2 = self.layer2(l1)  # 512, H/8
        l3 = self.layer3(l2)  # 1024, H/16
        return [stem, l1, l2, l3]


# =============================================================================
# IRIS: Unified Model
# =============================================================================

class IRIS(nn.Module):
    """
    IRIS: Unified Medical Image Segmentation.
    
    Args:
        visual_encoder: "cnn" (5-level) | "resnet" (MedKLIP ResNet50)
        text_encoder: "clip" | "pubmedbert" | "clinicalbert"
        fusion_type: "bcam" | "bifmd"
        adapter: None | "bottleneck" | "spatial" | "multiscale"
        num_classes: Output classes
        img_size: Input image size
        base_channel: Base channel width
        max_token_len: Max text token length
        num_heads: Attention heads
        dropout: Dropout rate
        expansion: MLP expansion ratio
        deep_supervision: Enable auxiliary outputs
        freeze_visual: Freeze visual encoder
        visual_pretrained: Path to pretrained visual weights
        text_mask_rate: Text masking during training
        img_mask_rate: Image masking during training
    """
    
    # Text encoder configs
    TEXT_ENCODERS = {
        "clip": ("openai/clip-vit-base-patch32", CLIPTextEncoder, 512),
        "pubmedbert": ("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", BERTTextEncoder, 768),
        "clinicalbert": ("emilyalsentzer/Bio_ClinicalBERT", BERTTextEncoder, 768),
    }

    def __init__(
        self,
        # Encoder selection
        visual_encoder: str = "cnn",
        text_encoder: str = "clip",
        fusion_type: str = "bcam",
        adapter: Optional[str] = None,
        # Architecture
        num_classes: int = 1,
        img_size: int = 224,
        base_channel: int = 64,
        max_token_len: int = 18,
        num_heads: int = 8,
        dropout: float = 0.5,
        expansion: int = 4,
        deep_supervision: bool = True,
        # Training
        freeze_visual: bool = False,
        visual_pretrained: Optional[str] = None,
        text_mask_rate: float = 0.3,
        img_mask_rate: float = 0.3,
        adapter_reduction: int = 4,
        adapter_scale: float = 0.1,
    ):
        super().__init__()
        
        self.visual_encoder_type = visual_encoder
        self.text_encoder_type = text_encoder
        self.fusion_type = fusion_type
        self.deep_supervision = deep_supervision
        self.text_mask_rate = text_mask_rate
        self.img_mask_rate = img_mask_rate
        self.base_ch = base_channel
        self.img_size = img_size
        
        # =====================================================================
        # Visual Encoder
        # =====================================================================
        if visual_encoder == "resnet":
            self.v_enc = ResNetEncoder(visual_pretrained, freeze_visual)
            v_ch = self.v_enc.channels  # [64, 256, 512, 1024]
            self.num_levels = 4
            # Channel adapters to base_channel scale
            self.ch_adapt = nn.ModuleList([
                nn.Conv2d(v_ch[0], base_channel, 1),
                nn.Conv2d(v_ch[1], base_channel * 2, 1),
                nn.Conv2d(v_ch[2], base_channel * 4, 1),
                nn.Conv2d(v_ch[3], base_channel * 8, 1),
            ])
            # Visual adapters (if using)
            if adapter:
                self.v_adapters = nn.ModuleList([
                    get_adapter(adapter, v_ch[i], adapter_reduction, adapter_scale)
                    for i in range(4)
                ])
            else:
                self.v_adapters = None
        else:  # cnn
            self.v_enc = CNNEncoder(base_channel)
            self.num_levels = 5
            self.ch_adapt = None
            self.v_adapters = None
            if freeze_visual:
                for p in self.v_enc.parameters():
                    p.requires_grad_(False)

        # =====================================================================
        # Text Encoder
        # =====================================================================
        model_name, encoder_cls, text_dim = self.TEXT_ENCODERS[text_encoder]
        self.t_enc = encoder_cls(model_name, max_token_len)
        
        # Project text to 512 for fusion (BCAM expects 512)
        if text_dim != 512:
            self.text_proj = nn.Linear(text_dim, 512)
        else:
            self.text_proj = None
        self._text_dim = 512  # Fusion always uses 512

        # =====================================================================
        # Fusion Modules
        # =====================================================================
        scales = [img_size // (2 ** i) for i in range(5)]  # [224, 112, 56, 28, 14]
        
        FusionClass = BiFMD if fusion_type == "bifmd" else BCAModule
        
        if self.num_levels == 5:
            # CNN: fusion at levels 1-4 (skip level 0 = full res)
            self.fuse1 = FusionClass(scales[1], base_channel, self._text_dim, num_heads, dropout if fusion_type == "bcam" else expansion)
            self.fuse2 = FusionClass(scales[2], base_channel * 2, self._text_dim, num_heads, dropout if fusion_type == "bcam" else expansion)
            self.fuse3 = FusionClass(scales[3], base_channel * 4, self._text_dim, num_heads, dropout if fusion_type == "bcam" else expansion)
            self.fuse4 = FusionClass(scales[4], base_channel * 8, self._text_dim, num_heads, dropout if fusion_type == "bcam" else expansion)
        else:
            # ResNet: fusion at levels 1-3 (4 total levels)
            self.fuse1 = FusionClass(scales[2], base_channel * 2, self._text_dim, num_heads, dropout if fusion_type == "bcam" else expansion)
            self.fuse2 = FusionClass(scales[3], base_channel * 4, self._text_dim, num_heads, dropout if fusion_type == "bcam" else expansion)
            self.fuse3 = FusionClass(scales[4], base_channel * 8, self._text_dim, num_heads, dropout if fusion_type == "bcam" else expansion)

        # =====================================================================
        # Decoder
        # =====================================================================
        if self.num_levels == 5:
            self.up3 = UpBlock(base_channel * 8, base_channel * 8, base_channel * 4)
            self.up2 = UpBlock(base_channel * 4, base_channel * 4, base_channel * 2)
            self.up1 = UpBlock(base_channel * 2, base_channel * 2, base_channel)
            self.up0 = UpBlock(base_channel, base_channel, base_channel)
        else:
            self.up2 = UpBlock(base_channel * 4, base_channel * 8, base_channel * 4)
            self.up1 = UpBlock(base_channel * 2, base_channel * 4, base_channel * 2)
            self.up0 = UpBlock(base_channel, base_channel * 2, base_channel)
            self.final_up = nn.Sequential(
                nn.ConvTranspose2d(base_channel, base_channel, 2, stride=2),
                ConvBNReLU(base_channel, base_channel),
            )

        # =====================================================================
        # Output Heads
        # =====================================================================
        self.out0 = nn.Conv2d(base_channel, num_classes, 1)
        
        if deep_supervision:
            if self.num_levels == 5:
                self.out1 = nn.Conv2d(base_channel, num_classes, 1)
                self.out2 = nn.Conv2d(base_channel * 2, num_classes, 1)
                self.out3 = nn.Conv2d(base_channel * 4, num_classes, 1)
                self.out4 = nn.Conv2d(base_channel * 8, num_classes, 1)
            else:
                self.out1 = nn.Conv2d(base_channel, num_classes, 1)
                self.out2 = nn.Conv2d(base_channel * 2, num_classes, 1)
                self.out3 = nn.Conv2d(base_channel * 4, num_classes, 1)

        # Contrastive projections
        self.img_proj = nn.Linear(base_channel * 8, 256)
        self.text_proj_cl = nn.Linear(512, 256)

    def _encode_visual(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Encode image to multi-scale features."""
        feats = self.v_enc(x)
        
        if self.v_adapters:
            feats = [adapter(f) for adapter, f in zip(self.v_adapters, feats)]
        
        if self.ch_adapt:
            feats = [adapt(f) for adapt, f in zip(self.ch_adapt, feats)]
        
        return feats

    def _encode_text(self, texts: List[str], device) -> torch.Tensor:
        """Encode text to token embeddings."""
        token_emb, _ = self.t_enc(texts, device)
        if self.text_proj:
            token_emb = self.text_proj(token_emb)
        return token_emb

    def _apply_text_mask(self, emb: torch.Tensor) -> torch.Tensor:
        if not self.training or self.text_mask_rate <= 0:
            return emb
        B, N, D = emb.shape
        mask = torch.rand(B, N, device=emb.device) > self.text_mask_rate
        mask[:, 0] = True  # Keep first token
        return emb * mask.unsqueeze(-1).float()

    def _apply_img_mask(self, feat: torch.Tensor) -> torch.Tensor:
        if not self.training or self.img_mask_rate <= 0:
            return feat
        B, C, H, W = feat.shape
        ps = max(1, H // 7)
        pH, pW = H // ps, W // ps
        mask = torch.rand(B, 1, pH, pW, device=feat.device) > self.img_mask_rate
        mask = F.interpolate(mask.float(), (H, W), mode='nearest')
        return feat * mask

    def forward(self, images: torch.Tensor, texts: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        device = images.device
        
        # Encode
        v = self._encode_visual(images)
        l = self._encode_text(texts, device) if texts else None
        if l is not None:
            l = self._apply_text_mask(l)
        
        # Apply image mask to deepest level
        v[-1] = self._apply_img_mask(v[-1])
        
        # Fusion + Decode
        if self.num_levels == 5:
            # CNN: 5 levels [v0, v1, v2, v3, v4]
            if l is not None:
                f4 = self.fuse4(v[4], l)
                up3 = self.up3(v[3], f4)
                f3 = self.fuse3(up3, l)
                up2 = self.up2(v[2], f3)
                f2 = self.fuse2(up2, l)
                up1 = self.up1(v[1], f2)
                f1 = self.fuse1(up1, l)
                up0 = self.up0(v[0], f1)
            else:
                up3 = self.up3(v[3], v[4])
                up2 = self.up2(v[2], up3)
                up1 = self.up1(v[1], up2)
                up0 = self.up0(v[0], up1)
                f1, f2, f3, f4 = up1, up2, up3, v[4]
            
            pred = self.out0(up0)
            outputs = {'pred': pred}
            
            if self.deep_supervision and self.training:
                outputs['aux_preds'] = [self.out1(f1), self.out2(f2), self.out3(f3), self.out4(f4)]
        else:
            # ResNet: 4 levels [stem, l1, l2, l3]
            if l is not None:
                f3 = self.fuse3(v[3], l)
                up2 = self.up2(v[2], f3)
                f2 = self.fuse2(up2, l)
                up1 = self.up1(v[1], f2)
                f1 = self.fuse1(up1, l)
                up0 = self.up0(v[0], f1)
            else:
                up2 = self.up2(v[2], v[3])
                up1 = self.up1(v[1], up2)
                up0 = self.up0(v[0], up1)
                f1, f2, f3 = up1, up2, v[3]
            
            final = self.final_up(up0)
            pred = self.out0(final)
            outputs = {'pred': pred}
            
            if self.deep_supervision and self.training:
                outputs['aux_preds'] = [self.out1(up0), self.out2(f1), self.out3(f2)]
        
        return outputs

    def get_contrastive_embeddings(self, images: torch.Tensor, texts: List[str]):
        """For CaCL pretraining."""
        v = self._encode_visual(images)
        v_global = F.adaptive_avg_pool2d(v[-1], 1).flatten(1)
        
        with torch.no_grad():
            _, sent_emb = self.t_enc(texts, images.device)
            if self.text_proj:
                sent_emb = self.text_proj(sent_emb)
        
        v_out = F.normalize(self.img_proj(v_global), dim=-1)
        l_out = F.normalize(self.text_proj_cl(sent_emb), dim=-1)
        l_raw = F.normalize(sent_emb.detach().float(), dim=-1)
        
        return v_out, l_out, l_raw


# =============================================================================
# Build Function
# =============================================================================

def build_iris(config) -> IRIS:
    """Build IRIS model from config."""
    mcfg = config.get('model', config)
    dcfg = config.get('data', {})
    
    return IRIS(
        visual_encoder=mcfg.get('visual_encoder', 'cnn'),
        text_encoder=mcfg.get('text_encoder', 'clip'),
        fusion_type=mcfg.get('fusion_type', 'bcam'),
        adapter=mcfg.get('adapter', None),
        num_classes=mcfg.get('num_classes', 1),
        img_size=dcfg.get('image_size', 224),
        base_channel=mcfg.get('base_channel', 64),
        max_token_len=mcfg.get('max_token_len', 18),
        num_heads=mcfg.get('num_heads', 8),
        dropout=mcfg.get('dropout', 0.5),
        expansion=mcfg.get('expansion', 4),
        deep_supervision=mcfg.get('deep_supervision', True),
        freeze_visual=mcfg.get('freeze_visual', False),
        visual_pretrained=mcfg.get('visual_pretrained', None),
        text_mask_rate=mcfg.get('text_mask_rate', 0.3),
        img_mask_rate=mcfg.get('img_mask_rate', 0.3),
        adapter_reduction=mcfg.get('adapter_reduction', 4),
        adapter_scale=mcfg.get('adapter_scale', 0.1),
    )
