"""
LoRA modules for MRSeg.

LoRALinear        — standard LoRA for BERT Q/V (input shape B×L×C)
LoRALinearWithDW  — LoRA + DW3×3 for ConvNeXt pwconv (input shape B×H×W×C, channel-last)
apply_lora_to_linear — helper to wrap an existing nn.Linear in-place
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Standard LoRA wrapper for nn.Linear. Input: (B, L, C)."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 scale: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.rank = rank
        self.scale = scale
        self.original: nn.Linear  # set externally

        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.drop = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + (self.scale / self.rank) * self.B(self.drop(self.A(x)))


class LoRALinearWithDW(nn.Module):
    """LoRA + DW-Conv3×3 wrapper for ConvNeXt pwconv. Input: (B, H, W, C) channel-last."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 scale: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.rank = rank
        self.scale = scale
        self.original: nn.Linear  # set externally

        self.A = nn.Linear(in_features, rank, bias=False)
        self.dw = nn.Conv2d(rank, rank, 3, padding=1, groups=rank)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.drop = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.dw.weight, a=5 ** 0.5)
        nn.init.zeros_(self.dw.bias)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        B, H, W, C = x.shape

        a = self.drop(self.A(x))                     # (B, H, W, rank)
        a = a.permute(0, 3, 1, 2)                    # (B, rank, H, W)
        a = self.dw(a)                               # (B, rank, H, W)
        a = a.permute(0, 2, 3, 1)                    # (B, H, W, rank)
        b = self.B(a)                                # (B, H, W, out_features)

        return self.original(x) + (self.scale / self.rank) * b


def apply_lora_to_linear(
    module: nn.Module,
    attr_name: str,
    lora_cls,
    rank: int = 8,
    scale: float = 16.0,
    dropout: float = 0.05,
):
    """Replace module.attr_name (nn.Linear) with a LoRA wrapper in-place."""
    original = getattr(module, attr_name)
    assert isinstance(original, nn.Linear), \
        f"{attr_name} is {type(original)}, expected nn.Linear"

    wrapper = lora_cls(
        original.in_features,
        original.out_features,
        rank=rank,
        scale=scale,
        dropout=dropout,
    )
    wrapper.original = original

    # Keep original frozen
    for p in original.parameters():
        p.requires_grad_(False)

    setattr(module, attr_name, wrapper)
