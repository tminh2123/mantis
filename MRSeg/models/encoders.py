"""
Visual and text encoders for MRSeg.

ConvNeXtWithLoRA  — ConvNeXt-Tiny with LoRALinearWithDW on stages[2,3]
PubMedBERTWithLoRA — BiomedBERT with LoRALinear on Q,V of last 4 layers
"""
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import ConvNextModel, AutoModel, AutoTokenizer

from .lora import LoRALinear, LoRALinearWithDW, apply_lora_to_linear


class ConvNeXtWithLoRA(nn.Module):
    """
    ConvNeXt-Tiny encoder with LoRA on the last two stages.
    Returns four feature maps: f1 (H/4), f2 (H/8), f3 (H/16), f4 (H/32).
    """

    def __init__(
        self,
        pretrained_path: str,
        lora_rank: int = 8,
        lora_scale: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.model = ConvNextModel.from_pretrained(pretrained_path)

        # Freeze all backbone params first
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Apply LoRA to pwconv1 and pwconv2 of stages[2] and stages[3]
        lora_kwargs = dict(rank=lora_rank, scale=lora_scale, dropout=lora_dropout)
        for stage_idx in (2, 3):
            for layer in self.model.encoder.stages[stage_idx].layers:
                apply_lora_to_linear(layer, 'pwconv1', LoRALinearWithDW, **lora_kwargs)
                apply_lora_to_linear(layer, 'pwconv2', LoRALinearWithDW, **lora_kwargs)
                # Unfreeze LayerNorm in LoRA'd layers
                layer.layernorm.weight.requires_grad_(True)
                layer.layernorm.bias.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        x: (B, 3, H, W)
        Returns: f1 (B,96,H/4,W/4), f2 (B,192,H/8,W/8),
                 f3 (B,384,H/16,W/16), f4 (B,768,H/32,W/32)
        """
        outputs = self.model(x, output_hidden_states=True)
        hs = outputs.hidden_states
        # hidden_states indices for ConvNeXt-Tiny (5 entries):
        #   [0] after embeddings (stem)   96 × H/4
        #   [1] after stage 0             96 × H/4   <- f1
        #   [2] after stage 1            192 × H/8   <- f2
        #   [3] after stage 2            384 × H/16  <- f3
        #   [4] after stage 3            768 × H/32  <- f4
        return hs[1], hs[2], hs[3], hs[4]


class PubMedBERTWithLoRA(nn.Module):
    """
    BiomedBERT encoder with LoRA on Q and V of the last 4 transformer layers.
    Returns last_hidden_state: (B, max_length, 768).
    """

    def __init__(
        self,
        pretrained_path: str,
        max_length: int = 24,
        lora_rank: int = 8,
        lora_scale: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.model = AutoModel.from_pretrained(pretrained_path)

        # Freeze all params first
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Apply LoRA to Q and V of last 4 layers
        lora_kwargs = dict(rank=lora_rank, scale=lora_scale, dropout=lora_dropout)
        for layer in self.model.encoder.layer[-4:]:
            apply_lora_to_linear(layer.attention.self, 'query', LoRALinear, **lora_kwargs)
            apply_lora_to_linear(layer.attention.self, 'value', LoRALinear, **lora_kwargs)
            # Unfreeze LayerNorms in those layers
            for ln in (layer.attention.output.LayerNorm, layer.output.LayerNorm):
                ln.weight.requires_grad_(True)
                ln.bias.requires_grad_(True)

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        texts: list of B strings
        Returns: (B, max_length, 768)
        """
        tokens = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        return self.model(**tokens).last_hidden_state
