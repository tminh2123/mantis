# -*- coding: utf-8 -*-
"""
CLIP model implementation with ResNet and Transformer backbones.
Includes pretrained model downloading, initialization, and weight conversion.
"""

import os
import hashlib
import urllib
import warnings
from collections import OrderedDict
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# =============================
# Pretrained Models
# =============================
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}

_PT_NAME = {
    "RN50": "RN50.pt",
    "RN101": "RN101.pt",
    "RN50x4": "RN50x4.pt",
    "RN50x16": "RN50x16.pt",
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
    "ViT-L/14": "ViT-L-14.pt",
}


def available_models():
    """Return list of available pretrained CLIP models."""
    return list(_MODELS.keys())


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")) -> str:
    """Download pretrained model weights with checksum verification."""
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    target_path = os.path.join(root, filename)

    if os.path.isfile(target_path):
        if hashlib.sha256(open(target_path, "rb").read()).hexdigest() == expected_sha256:
            return target_path
        warnings.warn(f"{target_path} exists, but checksum mismatch. Re-downloading...")

    with urllib.request.urlopen(url) as src, open(target_path, "wb") as dst:
        with tqdm(total=int(src.info().get("Content-Length")), ncols=80, unit="iB", unit_scale=True) as pbar:
            while True:
                buf = src.read(8192)
                if not buf:
                    break
                dst.write(buf)
                pbar.update(len(buf))

    if hashlib.sha256(open(target_path, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Downloaded model checksum mismatch.")

    return target_path


# =============================
# Core Modules
# =============================
class LayerNorm(nn.LayerNorm):
    """LayerNorm supporting mixed precision (fp16)."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        out = super().forward(x.float())
        return out.to(orig_type)


class QuickGELU(nn.Module):
    """Fast approximation of GELU."""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Bottleneck(nn.Module):
    """ResNet bottleneck block with optional downsampling."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = None
        if stride > 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class AttentionPool2d(nn.Module):
    """QKV attention pooling for 2D feature maps."""
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.q_proj, self.k_proj, self.v_proj = [nn.Linear(embed_dim, embed_dim) for _ in range(3)]
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), -1).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)     # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)

        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1], num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]


class ModifiedResNet(nn.Module):
    """ResNet backbone with modified stem, anti-aliasing, and attention pooling."""
    def __init__(self, layers, output_dim, heads, input_res=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_res = input_res
        self.relu = nn.ReLU(inplace=True)

        # Stem
        self.conv1 = nn.Conv2d(3, width // 2, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)

        # Residual layers
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_res // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        layers += [Bottleneck(self._inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            return self.avgpool(x)

        x = stem(x.type(self.conv1.weight.dtype))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.attnpool(x)


# =============================
# Utilities
# =============================
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


# =============================
# Transformer & CLIP
# =============================
class ResidualAttentionBlock(nn.Module):
    """Basic transformer block with residual attention and MLP."""
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super(ResidualAttentionBlock, self).__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask_: torch.Tensor):
        attn_mask_ = attn_mask_.repeat_interleave(self.n_head, dim=0)
        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x, attn_mask


class Transformer(nn.Module):
    """Stacked transformer encoder."""
    def __init__(self, width: int, layers: int, heads: int):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))


class VisualTransformer(nn.Module):
    """Vision Transformer backbone for CLIP."""
    def __init__(self, input_res: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_res // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # freeze patch projection conv
        for p in self.conv1.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor):
        # shape: [B, 3, H, W] -> [B, grid**2+1, width]
        x = self.conv1(x).reshape(x.size(0), x.size(1), -1).permute(0, 2, 1)
        class_tok = self.class_embedding + torch.zeros(x.size(0), 1, x.size(-1), dtype=x.dtype, device=x.device)
        x = torch.cat([class_tok, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # Transformer: NLD -> LND -> NLD
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        return x


class CLIP(nn.Module):
    """
    Contrastive Language-Image Pretraining (CLIP).
    Supports ResNet-based or Transformer-based visual backbones,
    with a Transformer encoder for text.
    """
    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size

        # ---------------- Vision Backbone ----------------
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_res=image_resolution,
                width=vision_width,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_res=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )

        # ---------------- Text Transformer ----------------
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
        )
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        # ---------------- Scaling Parameter ----------------
        self.logit_scale = nn.Parameter(torch.ones([]))

        # ---------------- Initialization ----------------
        self.initialize_parameters()

        # Optionally freeze embeddings
        self.token_embedding.requires_grad = False

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # Initialize ResNet attention pooling if available
        if isinstance(self.visual, ModifiedResNet) and self.visual.attnpool is not None:
            std = self.visual.attnpool.c_proj.in_features ** -0.5
            for proj in [
                self.visual.attnpool.q_proj,
                self.visual.attnpool.k_proj,
                self.visual.attnpool.v_proj,
                self.visual.attnpool.c_proj,
            ]:
                nn.init.normal_(proj.weight, std=std)

            for res_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4,
            ]:
                for name, param in res_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        # Transformer blocks
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # Text projection
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @staticmethod
    def get_config(pretrained_clip_name: str = "ViT-B/32") -> str:
        if pretrained_clip_name not in _PT_NAME:
            raise ValueError(f"Model {pretrained_clip_name} not supported. Available = {available_models()}")

        model_path = os.path.join(local_model_dir, _PT_NAME[pretrained_clip_name])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Expected local model at {model_path}, but not found.")
        return model_path

    @staticmethod
    def build_attention_mask(context_length: int) -> torch.Tensor:
        """Build a causal attention mask (upper triangular filled with -inf)."""
        mask = torch.full((context_length, context_length), float("-inf"))
        return mask.triu_(1)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image: torch.Tensor, return_hidden: bool = False):
        features = self.visual(image.type(self.dtype))
        features = self.visual.ln_post(features) @ self.visual.proj
        cls_feature = features[:, 0, :]  # CLS token

        return (cls_feature, features) if return_hidden else cls_feature

    def encode_text(
        self,
        text: torch.Tensor,
        mask: torch.Tensor = None,
        text_vec: torch.Tensor = None,
        return_hidden: bool = False,
    ):
        x = self.token_embedding(text).type(self.dtype)
        if text_vec is not None:
            x = torch.cat([x, text_vec], dim=1)

        x = x + self.positional_embedding[: x.size(1), :]

        # attention mask
        attn_mask = self.build_attention_mask(x.size(1)).to(x.device)
        if mask is not None:
            expanded_mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)
            attn_mask = torch.where(expanded_mask > 0, attn_mask, torch.full_like(attn_mask, float("-inf")))

        x = x.permute(1, 0, 2)  # [B, L, D] -> [L, B, D]
        y, _ = self.transformer(x, attn_mask)
        y = y.permute(1, 0, 2)  # back to [B, L, D]

        hidden = self.ln_final(y).type(self.dtype) @ self.text_projection
        eot_indices = text.argmax(dim=-1)
        out = hidden[torch.arange(hidden.shape[0]), eot_indices]

        return (out, hidden) if return_hidden else out

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


# =============================
# Model Loader
# =============================
def load_clip_model(config) -> CLIP:
    """Load CLIP model given a config with backbone and freezing options."""
    backbone = config.clip_backbone
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])

    if not os.path.exists(model_path):
        model_path = CLIP.get_config(backbone)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # infer model hyperparameters
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len({k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")})

    # build CLIP
    clip_model = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                      context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

    clip_model.load_state_dict(state_dict, strict=False)
    clip_model.float()

    # if torch.cuda.is_available():
    #     convert_weights(clip_model)

    if config.frozen_clip:
        for p in clip_model.parameters():
            p.requires_grad = False

    return clip_model

