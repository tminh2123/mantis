#!/usr/bin/env python3
"""
Visualize per-token attention maps.

Shows which spatial regions each text token attends to.
Example: "left lung infection" → 3 heatmaps for "left", "lung", "infection"

Usage:
    python scripts/visualize_attention.py \
        --config configs/qata.yaml \
        --checkpoint output/checkpoints/qata_fusion_h8_e4/best.pth \
        --sample_idx 0 \
        --output_dir output/analysis/attention_maps

Output:
    - {sample_id}_attention.png: Grid of per-token attention maps
    - {sample_id}_overlay.png: Attention overlaid on original image
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.iris import IRIS
from datasets.dataset import IRISDataset
from torch.utils.data import DataLoader


class AttentionExtractor:
    """Hook to extract attention weights from BiFMD layers."""
    
    def __init__(self):
        self.attentions = {}
        self.hooks = []
    
    def hook_fn(self, name):
        def fn(module, input, output):
            # Store the attention weights
            # We need to modify BiFMD to return attention, or compute it here
            self.attentions[name] = {
                'input': [x.detach() if torch.is_tensor(x) else x for x in input],
                'output': output.detach() if torch.is_tensor(output) else output
            }
        return fn
    
    def register(self, model):
        """Register hooks on fusion layers."""
        for name, module in model.named_modules():
            if 'fuse' in name and hasattr(module, 'img_q'):
                # This is a BiFMD layer
                hook = module.register_forward_hook(self.hook_fn(name))
                self.hooks.append(hook)
    
    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def compute_attention_maps(model, img, text, tokenizer, device):
    """
    Compute per-token attention maps.
    
    Returns:
        tokens: List of token strings
        attn_maps: Dict[layer_name -> Tensor[N_tokens, H, W]]
    """
    model.eval()
    
    # Get text tokens for display
    tokens_enc = tokenizer(
        [text], 
        padding="max_length", 
        truncation=True, 
        max_length=18, 
        return_tensors="pt"
    )
    token_ids = tokens_enc['input_ids'][0]
    attention_mask = tokens_enc['attention_mask'][0]
    
    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    # Filter to actual tokens (not padding)
    valid_len = attention_mask.sum().item()
    tokens = tokens[:valid_len]
    
    # Forward pass with attention extraction
    img = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get text embeddings
        text_emb, _ = model.t_enc([text], device)  # [1, N, D]
        if model.text_proj is not None:
            text_emb = model.text_proj(text_emb)
        
        # Get visual features
        v_feats = model.v_enc(img)
        
        # Apply adapters if ResNet
        if model.v_adapters is not None:
            v_feats = [adapter(f) for adapter, f in zip(model.v_adapters, v_feats)]
        
        # Apply channel adaptation if ResNet
        if model.ch_adapt is not None:
            v_feats = [adapt(f) for adapt, f in zip(model.ch_adapt, v_feats)]
        
        # Compute attention at each fusion layer
        attn_maps = {}
        
        if model.num_levels == 5:
            fuse_layers = [
                ('fuse1', model.fuse1, v_feats[0]),
                ('fuse2', model.fuse2, v_feats[1]),
                ('fuse3', model.fuse3, v_feats[2]),
                ('fuse4', model.fuse4, v_feats[3]),
            ]
        else:
            fuse_layers = [
                ('fuse1', model.fuse1, v_feats[1]),
                ('fuse2', model.fuse2, v_feats[2]),
                ('fuse3', model.fuse3, v_feats[3]),
            ]
        
        for name, fuse, v_feat in fuse_layers:
            if hasattr(fuse, 'img_q'):  # BiFMD
                attn = compute_bifmd_attention(fuse, v_feat, text_emb, device)
                attn_maps[name] = attn[:valid_len]  # Only valid tokens
    
    return tokens, attn_maps


def compute_bifmd_attention(fuse, img, text, device):
    """
    Extract attention weights from BiFMD layer.
    
    Returns:
        attn: Tensor[N_tokens, H, W] - attention from each token to spatial locations
    """
    B, C, H, W = img.shape
    N = text.size(1)
    P = H * W
    h = fuse.num_heads
    d = C // h
    
    # Apply LayerNorm
    img_ln = fuse.img_norm(img.flatten(2).transpose(1, 2))
    img_ln = img_ln.transpose(1, 2).reshape(B, C, H, W)
    text_ln = fuse.txt_norm(text)
    
    # FiLM modulation
    pooled_t = text_ln.mean(1)
    pooled_i = img_ln.mean(dim=(2, 3))
    
    dg, beta = fuse.t2i(pooled_t).chunk(2, dim=-1)
    img_mod = img_ln * (1 + dg[..., None, None]) + beta[..., None, None]
    
    dg2, beta2 = fuse.i2t(pooled_i).chunk(2, dim=-1)
    text_mod = text_ln * (1 + dg2[:, None]) + beta2[:, None]
    
    # Compute Q, K
    q = fuse.img_q(img_mod).flatten(2).transpose(1, 2)  # [B, P, C]
    q = q.reshape(B, P, h, d).transpose(1, 2)  # [B, h, P, d]
    
    k_txt = fuse.txt_k(text_mod).reshape(B, N, h, d).transpose(1, 2)  # [B, h, N, d]
    
    # Attention scores
    scores = (q @ k_txt.transpose(-2, -1)) / fuse.temperature.clamp(min=0.1)
    
    # Softmax over text tokens (which text token does each pixel attend to)
    # But we want: which pixels does each token attend to
    # So transpose and softmax over pixels
    attn_t2v = F.softmax(scores.transpose(-1, -2), dim=-1)  # [B, h, N, P]
    
    # Average over heads
    attn_t2v = attn_t2v.mean(dim=1)  # [B, N, P]
    
    # Reshape to spatial
    attn_maps = attn_t2v[0].reshape(N, H, W)  # [N, H, W]
    
    return attn_maps


def visualize_token_attention(
    image: np.ndarray,
    tokens: list,
    attn_maps: dict,
    output_path: str,
    sample_id: str,
    gt_mask: np.ndarray = None
):
    """
    Create visualization of per-token attention.
    
    Args:
        image: Original image [H, W, 3]
        tokens: List of token strings
        attn_maps: Dict[layer_name -> Tensor[N, H, W]]
        output_path: Output directory
        sample_id: Sample identifier
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Filter special tokens
    skip_tokens = {'[CLS]', '[SEP]', '[PAD]', '<|startoftext|>', '<|endoftext|>'}
    valid_indices = [i for i, t in enumerate(tokens) if t not in skip_tokens]
    valid_tokens = [tokens[i] for i in valid_indices]
    
    if len(valid_tokens) == 0:
        print(f"No valid tokens for {sample_id}")
        return
    
    # Use the deepest layer (highest resolution attention patterns)
    layer_names = sorted(attn_maps.keys())
    deepest_layer = layer_names[-1]
    attn = attn_maps[deepest_layer]
    
    # Upsample attention to image size
    H_img = image.shape[0]
    attn_up = F.interpolate(
        attn[valid_indices].unsqueeze(0),
        size=(H_img, H_img),
        mode='bilinear',
        align_corners=False
    )[0].cpu().numpy()
    
    n_tokens = len(valid_tokens)
    
    # =========================================================================
    # Plot 1: Grid of attention maps per token
    # =========================================================================
    n_cols = min(4, n_tokens)
    n_rows = (n_tokens + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]
    
    for idx, (token, attn_map) in enumerate(zip(valid_tokens, attn_up)):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        # Normalize attention for visualization
        attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
        ax.imshow(image)
        ax.imshow(attn_norm, cmap='jet', alpha=0.5)
        ax.set_title(f'"{token}"', fontsize=14)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_tokens, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Per-Token Attention Maps ({deepest_layer})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{sample_id}_attention.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Plot 1.5: Summary with GT mask
    # =========================================================================
    if gt_mask is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Input Image', fontsize=12)
        axes[0].axis('off')
        
        # GT mask overlay
        axes[1].imshow(image)
        axes[1].imshow(gt_mask, cmap='Reds', alpha=0.5)
        axes[1].set_title('Ground Truth Mask', fontsize=12)
        axes[1].axis('off')
        
        # Combined attention (average of all tokens)
        combined_attn = np.mean(attn_up, axis=0)
        combined_attn = (combined_attn - combined_attn.min()) / (combined_attn.max() - combined_attn.min() + 1e-8)
        axes[2].imshow(image)
        axes[2].imshow(combined_attn, cmap='jet', alpha=0.5)
        axes[2].set_title('Combined Attention', fontsize=12)
        axes[2].axis('off')
        
        plt.suptitle(f'Sample: {sample_id}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{sample_id}_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # =========================================================================
    # Plot 2: Attention across all layers
    # =========================================================================
    n_layers = len(layer_names)
    fig, axes = plt.subplots(n_layers, n_tokens, figsize=(3 * n_tokens, 3 * n_layers))
    
    if n_layers == 1:
        axes = axes[np.newaxis, :]
    if n_tokens == 1:
        axes = axes[:, np.newaxis]
    
    for layer_idx, layer_name in enumerate(layer_names):
        attn = attn_maps[layer_name]
        attn_up = F.interpolate(
            attn[valid_indices].unsqueeze(0),
            size=(H_img, H_img),
            mode='bilinear',
            align_corners=False
        )[0].cpu().numpy()
        
        for token_idx, (token, attn_map) in enumerate(zip(valid_tokens, attn_up)):
            ax = axes[layer_idx, token_idx]
            attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            
            ax.imshow(image)
            ax.imshow(attn_norm, cmap='jet', alpha=0.5)
            
            if layer_idx == 0:
                ax.set_title(f'"{token}"', fontsize=10)
            if token_idx == 0:
                ax.set_ylabel(layer_name, fontsize=10)
            ax.axis('off')
    
    plt.suptitle(f'Attention Across Layers', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{sample_id}_layers.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {sample_id}_attention.png, {sample_id}_layers.png")


def compute_all_dice_scores(model, dataset, device):
    """Compute Dice scores for all samples in dataset."""
    from tqdm import tqdm
    
    model.eval()
    scores = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Computing Dice scores"):
            sample = dataset[idx]
            img = sample['image'].unsqueeze(0).to(device)
            mask = sample['mask'].unsqueeze(0).to(device)
            text = sample['text']
            
            pred = model(img, [text])
            pred_bin = (torch.sigmoid(pred) > 0.5).float()
            
            # Dice score
            intersection = (pred_bin * mask).sum()
            union = pred_bin.sum() + mask.sum()
            dice = (2 * intersection / (union + 1e-8)).item()
            
            # Extract image_id from path
            img_path = sample['path']
            image_id = os.path.basename(img_path)
            
            scores.append({
                'idx': idx,
                'image_id': image_id,
                'text': text,
                'dice': dice
            })
    
    return scores


def select_diverse_samples(scores, num_per_group=3):
    """
    Select samples with diverse Dice scores.
    
    Returns samples from:
    - Worst (< 50%)
    - Below average (50-70%)
    - Average (70-85%)
    - Good (85-95%)
    - Best (> 95%)
    """
    groups = {
        'worst': [],      # < 50%
        'poor': [],       # 50-70%
        'average': [],    # 70-85%
        'good': [],       # 85-95%
        'best': []        # > 95%
    }
    
    for s in scores:
        dice = s['dice'] * 100
        if dice < 50:
            groups['worst'].append(s)
        elif dice < 70:
            groups['poor'].append(s)
        elif dice < 85:
            groups['average'].append(s)
        elif dice < 95:
            groups['good'].append(s)
        else:
            groups['best'].append(s)
    
    selected = []
    for group_name, samples in groups.items():
        # Sort by dice within group
        samples_sorted = sorted(samples, key=lambda x: x['dice'])
        # Take first, middle, last
        n = len(samples_sorted)
        if n == 0:
            continue
        elif n <= num_per_group:
            selected.extend([(group_name, s) for s in samples_sorted])
        else:
            indices = [0, n // 2, n - 1][:num_per_group]
            selected.extend([(group_name, samples_sorted[i]) for i in indices])
    
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sample_idx', type=int, default=None, help='Specific sample index')
    parser.add_argument('--sample_id', type=str, default=None, help='Specific sample ID')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples (if using sample_idx)')
    parser.add_argument('--diverse', action='store_true', help='Select diverse samples by Dice score')
    parser.add_argument('--num_per_group', type=int, default=2, help='Samples per Dice group (with --diverse)')
    parser.add_argument('--output_dir', type=str, default='output/analysis/attention_maps')
    # Model args
    parser.add_argument('--visual', type=str, default='resnet')
    parser.add_argument('--text', type=str, default='clinicalbert')
    parser.add_argument('--fusion', type=str, default='bifmd')
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--expansion', type=int, default=4)
    parser.add_argument('--memory_size', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Create model
    model = IRIS(
        visual_encoder=args.visual,
        text_encoder=args.text,
        fusion=args.fusion,
        num_classes=cfg.get('model', {}).get('num_classes', 1),
        img_size=cfg['data']['image_size'],
        base_channel=cfg.get('model', {}).get('base_channel', 64),
        num_heads=args.num_heads,
        expansion=args.expansion,
        memory_size=args.memory_size,
    ).to(device)
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Load dataset (test split)
    dataset = IRISDataset(
        root_dir=cfg['data']['test_dir'],
        image_size=cfg['data']['image_size'],
        augment=False,
    )
    
    # Get tokenizer for token display
    tokenizer = model.t_enc.tokenizer
    
    # Visualize samples
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which samples to visualize
    if args.diverse:
        print("\n=== Computing Dice scores for diverse sample selection ===")
        scores = compute_all_dice_scores(model, dataset, device)
        selected = select_diverse_samples(scores, args.num_per_group)
        
        print(f"\nSelected {len(selected)} samples across Dice groups:")
        for group, s in selected:
            print(f"  [{group:>8}] {s['image_id']}: Dice={s['dice']*100:.1f}% - '{s['text'][:50]}...'")
        
        samples_to_viz = [(s['idx'], group, s['dice']) for group, s in selected]
    elif args.sample_id:
        # Find sample by ID
        found_idx = None
        for i in range(len(dataset)):
            sample_path = dataset.samples[i][0]
            sample_name = os.path.basename(str(sample_path))
            if sample_name == args.sample_id or sample_name == args.sample_id + '.png':
                found_idx = i
                break
        if found_idx is None:
            print(f"Sample ID {args.sample_id} not found")
            return
        samples_to_viz = [(found_idx, None, None)]
    elif args.sample_idx is not None:
        samples_to_viz = [(i, None, None) for i in range(args.sample_idx, min(args.sample_idx + args.num_samples, len(dataset)))]
    else:
        # Default: first 5 samples
        samples_to_viz = [(i, None, None) for i in range(min(5, len(dataset)))]
    
    # Visualize each sample
    print(f"\n=== Generating attention visualizations ===")
    for idx, group, dice in samples_to_viz:
        sample = dataset[idx]
        img = sample['image']
        text = sample['text']
        mask = sample['mask']
        sample_id = os.path.basename(sample['path']).replace('.png', '')
        
        # Add group prefix to output filename if diverse mode
        if group:
            output_prefix = f"{group}_{dice*100:.0f}pct_{sample_id}"
        else:
            output_prefix = sample_id
        
        if dice is not None:
            print(f"\n[{idx}] {output_prefix}: Dice={dice*100:.1f}%")
        else:
            print(f"\n[{idx}] {output_prefix}")
        print(f"  Text: '{text}'")
        
        # Compute attention
        tokens, attn_maps = compute_attention_maps(model, img, text, tokenizer, device)
        print(f"  Tokens: {tokens}")
        
        # Denormalize image for visualization
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img_vis = img.permute(1, 2, 0) * std + mean
        img_vis = (img_vis.numpy() * 255).clip(0, 255).astype(np.uint8)
        
        # Also save GT mask overlay
        mask_np = mask.squeeze().numpy()
        
        # Visualize attention
        visualize_token_attention(img_vis, tokens, attn_maps, args.output_dir, output_prefix, mask_np)
    
    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
