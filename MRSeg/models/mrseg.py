"""
MRSeg — top-level model assembly.
"""
from typing import Dict, List

import torch
import torch.nn as nn

from .adapters import MedImageAdapter, Router, PairAdapter
from .encoders import ConvNeXtWithLoRA, PubMedBERTWithLoRA
from .region_bridge import RegionBridge
from .decoder import MRSegDecoder


class MRSegModel(nn.Module):
    """
    Medical Region-guided Segmentation model.

    Forward:
        x     : (B, 1, H, W) grayscale CT slice  [or (B,3,H,W) — auto-converted]
        texts : list[str] of length B
    Returns:
        {'pred': (B, 1, H, W)}  raw logits
    """

    def __init__(
        self,
        convnext_path: str = "pretrained_models/convnext-tiny-224",
        bert_path: str = "pretrained_models/pubmedbert",
        max_token_len: int = 24,
        lora_rank: int = 8,
        lora_scale: float = 16.0,
        lora_dropout: float = 0.05,
        decoder_dim: int = 192,
        num_regions_f3: int = 12,
        num_regions_f4: int = 6,
        router_topk: int = 2,
        router_num_adapters: int = 4,
        pair_adapter_rank: int = 8,
        pair_adapter_eta: float = 16.0,
        img_adapter_channels: int = 64,
    ):
        super().__init__()

        self.med_adapter = MedImageAdapter(img_adapter_channels)

        self.visual_enc = ConvNeXtWithLoRA(
            convnext_path,
            lora_rank=lora_rank,
            lora_scale=lora_scale,
            lora_dropout=lora_dropout,
        )

        self.text_enc = PubMedBERTWithLoRA(
            bert_path,
            max_length=max_token_len,
            lora_rank=lora_rank,
            lora_scale=lora_scale,
            lora_dropout=lora_dropout,
        )

        self.router = Router(
            visual_dim=768,
            text_dim=768,
            num_adapters=router_num_adapters,
            topk=router_topk,
        )

        self.pair_adapter = PairAdapter(
            f3_channels=384,
            f4_channels=768,
            text_dim=768,
            num_adapters=router_num_adapters,
            rank=pair_adapter_rank,
            eta=pair_adapter_eta,
        )

        self.region_bridge_f3 = RegionBridge(
            visual_channels=384,
            num_regions=num_regions_f3,
            D=decoder_dim,
        )
        self.region_bridge_f4 = RegionBridge(
            visual_channels=768,
            num_regions=num_regions_f4,
            D=decoder_dim,
        )

        self.decoder = MRSegDecoder(decoder_dim)

    def forward(self, x: torch.Tensor, texts: List[str]) -> Dict[str, torch.Tensor]:
        # Accept 3-channel input (convert to grayscale)
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # Grayscale → I3 + stem skips
        i3, skip1, skip2 = self.med_adapter(x)

        # Visual features
        f1, f2, f3, f4 = self.visual_enc(i3)

        # Text features
        z_text = self.text_enc(texts, x.device)

        # Routing
        alpha = self.router(f4, z_text)

        # Adapt visual and text features
        f3p, f4p, t_prime = self.pair_adapter(f3, f4, z_text, alpha)

        # Region bridge refinement
        f3pp = self.region_bridge_f3(f3p, t_prime)
        f4pp = self.region_bridge_f4(f4p, t_prime)

        # Decode
        pred = self.decoder(f1, f2, f3pp, f4pp, skip1, skip2)

        return {'pred': pred}


def build_mrseg(cfg: dict) -> MRSegModel:
    """Instantiate MRSegModel from a config dict (expects a 'model' sub-dict)."""
    mcfg = cfg.get('model', {})
    return MRSegModel(
        convnext_path=mcfg.get('convnext_path', 'pretrained_models/convnext-tiny-224'),
        bert_path=mcfg.get('bert_path', 'pretrained_models/pubmedbert'),
        max_token_len=mcfg.get('max_token_len', 24),
        lora_rank=mcfg.get('lora_rank', 8),
        lora_scale=mcfg.get('lora_scale', 16.0),
        lora_dropout=mcfg.get('lora_dropout', 0.05),
        decoder_dim=mcfg.get('decoder_dim', 192),
        num_regions_f3=mcfg.get('num_regions_f3', 12),
        num_regions_f4=mcfg.get('num_regions_f4', 6),
        router_topk=mcfg.get('router_topk', 2),
        router_num_adapters=mcfg.get('router_num_adapters', 4),
        pair_adapter_rank=mcfg.get('pair_adapter_rank', 8),
        pair_adapter_eta=mcfg.get('pair_adapter_eta', 16.0),
        img_adapter_channels=mcfg.get('img_adapter_channels', 64),
    )
