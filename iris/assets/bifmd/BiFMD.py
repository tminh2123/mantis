"""BiFMD: Bidirectional FiLM-Modulated Decoder.

Research prototype replacing BCAM + K-Gate in C2Seg; no KAN.
Inputs: image (B,C,H,W), text (B,N,D), optional text_mask (B,N) True=valid.
Output: fused spatial map (B,C,H,W).

This is a research design, not an official C2Seg implementation.
"""
from __future__ import annotations
import math
import torch
from torch import nn


class BiFMD(nn.Module):
    """Bidirectional FiLM -> two-path multihead cross-modal attention -> Conv2d fusion."""

    def __init__(self, channels: int, text_dim: int, num_heads: int = 4,
                 normalize_spatial: bool = True, use_text_film: bool = True,
                 use_image_film: bool = True, expansion_ratio: int = 4):
        super().__init__()
        if channels <= 0 or text_dim <= 0 or num_heads <= 0:
            raise ValueError('dimensions and num_heads must be positive')
        if channels % num_heads:
            raise ValueError('channels must be divisible by num_heads')
        if expansion_ratio < 1:
            raise ValueError("expansion_ratio must be >= 1")
        self.expansion_ratio = expansion_ratio
        self.channels, self.text_dim = channels, text_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.normalize_spatial = normalize_spatial
        self.use_text_film = use_text_film
        self.use_image_film = use_image_film

        # Token sequences use Linear; spatial maps use Conv2d.
        self.text_to_image = nn.Linear(text_dim, 2 * channels)
        self.image_to_text = nn.Linear(channels, 2 * text_dim)
        # Identity FiLM at initialization.
        for module in (self.text_to_image, self.image_to_text):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

        self.image_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.image_v = nn.Conv2d(channels, channels, kernel_size=1)
        self.text_k = nn.Linear(text_dim, channels)
        self.text_v = nn.Linear(text_dim, channels)

        self.vision_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.language_projection = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        # Conv-FFN: expand channels, spatial mixing via depthwise 3x3, project back.
        hidden = channels * expansion_ratio
        self.refine = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        # Start residual Conv-FFN as identity; trainable projection learns updates.
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, image: torch.Tensor, text: torch.Tensor,
                text_mask: torch.Tensor | None = None) -> torch.Tensor:
        if image.ndim != 4 or text.ndim != 3:
            raise ValueError('Expected image (B,C,H,W), text (B,N,D)')
        batch, channels, height, width = image.shape
        bt, n_tokens, dim = text.shape
        if batch != bt or channels != self.channels or dim != self.text_dim:
            raise ValueError('Input batch/channels/text_dim mismatch')
        if n_tokens < 1:
            raise ValueError('At least one text token is required')
        if text_mask is None:
            text_mask = torch.ones((batch, n_tokens), device=text.device, dtype=torch.bool)
        else:
            if text_mask.shape != (batch, n_tokens):
                raise ValueError('text_mask shape must be (B,N)')
            text_mask = text_mask.to(device=text.device, dtype=torch.bool)
        if not bool(text_mask.any(dim=1).all()):
            raise ValueError('Each example needs >= 1 valid text token')

        positions = height * width
        h, d = self.num_heads, self.head_dim
        m = text_mask.unsqueeze(-1)  # B,N,1
        pooled_text = (text * m).sum(1) / m.sum(1).clamp(min=1)
        pooled_image = image.mean(dim=(2, 3))

        image_mod = image
        if self.use_text_film:
            dg, beta = self.text_to_image(pooled_text).chunk(2, dim=-1)
            image_mod = image * (1 + dg[..., None, None]) + beta[..., None, None]
        text_mod = text
        if self.use_image_film:
            dg, beta = self.image_to_text(pooled_image).chunk(2, dim=-1)
            text_mod = text * (1 + dg[:, None, :]) + beta[:, None, :]
        # Ignore padded positions even after affine shifts.
        text_mod = text_mod * m

        def image_heads(x):
            return x.flatten(2).transpose(1, 2).reshape(batch, positions, h, d).transpose(1, 2)

        q = image_heads(self.image_q(image_mod))          # B,h,P,d
        v_image = image_heads(self.image_v(image_mod))   # B,h,P,d
        k_text = self.text_k(text_mod).reshape(batch, n_tokens, h, d).transpose(1, 2)
        v_text = self.text_v(text_mod).reshape(batch, n_tokens, h, d).transpose(1, 2)

        scores = (q @ k_text.transpose(-2, -1)) / math.sqrt(d)  # B,h,P,N
        # Vision-dominant: softmax over VALID text tokens.
        masked_scores = scores.masked_fill(~text_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        alpha_text = masked_scores.softmax(dim=-1)
        f_vision = alpha_text @ v_text  # B,h,P,d

        # Language-dominant: each valid text token normalizes over P pixels;
        # invalid text tokens are removed from the average.
        alpha_spatial = scores.softmax(dim=-2) * text_mask[:, None, None, :]
        spatial = alpha_spatial.sum(dim=-1) / text_mask.sum(dim=1)[:, None, None].clamp(min=1)
        if self.normalize_spatial:
            # Intentional variant vs original BCAM: uniform spatial map has scale 1.
            spatial = spatial * positions
        f_language = v_image * spatial.unsqueeze(-1)

        def to_map(x):
            return x.transpose(1, 2).contiguous().reshape(batch, channels, height, width)

        fv = self.vision_projection(to_map(f_vision))
        fl = self.language_projection(to_map(f_language))
        gate = torch.sigmoid(self.gate(torch.cat([fv, fl], dim=1)))
        mix = self.fuse(gate * fv + (1 - gate) * fl)
        fused = self.norm(mix + image_mod)
        return fused + self.refine(fused)


def _self_test():
    torch.manual_seed(7)
    model = BiFMD(channels=32, text_dim=48, num_heads=4, expansion_ratio=4)
    assert model.refine[1].groups == 128
    image = torch.randn(2, 32, 8, 10, requires_grad=True)
    text = torch.randn(2, 6, 48, requires_grad=True)
    valid = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    output = model(image, text, valid)
    assert output.shape == image.shape, output.shape
    assert torch.isfinite(output).all()
    loss = output.square().mean()
    loss.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert text.grad is not None and torch.isfinite(text.grad).all()
    assert model.text_to_image.weight.grad is not None
    # Padded tokens should not affect output.
    model.eval()
    with torch.no_grad():
        a = model(image.detach(), text.detach(), valid)
        t2 = text.detach().clone()
        t2[~valid] = 1e4
        b = model(image.detach(), t2, valid)
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)
    print('BiFMD tests passed:', tuple(output.shape), '| loss:', round(loss.item(), 6))


if __name__ == '__main__':
    _self_test()
