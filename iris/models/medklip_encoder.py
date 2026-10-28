"""
MedKLIP Encoder Integration for IRIS

This module provides MedKLIP pretrained encoders (visual + text) for IRIS.
MedKLIP uses:
  - Visual: ResNet50 (pretrained on MIMIC-CXR with medical knowledge)
  - Text: BioBERT/ClinicalBERT with entity translation

Reference:
  Wu et al., "MedKLIP: Medical Knowledge Enhanced Language-Image Pre-Training"
  ICCV 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel, AutoTokenizer
from einops import rearrange
from typing import List, Tuple, Optional


class MedKLIPVisualEncoder(nn.Module):
    """
    MedKLIP Visual Encoder based on ResNet50.
    
    Outputs multi-scale features for segmentation decoder.
    When loading MedKLIP pretrained weights, it initializes from their checkpoint.
    """
    
    def __init__(
        self, 
        pretrained_path: Optional[str] = None,
        d_model: int = 256,
        freeze: bool = False
    ):
        super().__init__()
        
        # Build ResNet50 backbone (same as MedKLIP)
        resnet = models.resnet50(pretrained=False)
        
        # Extract feature layers (matching MedKLIP structure)
        # res_features = layers up to layer3 (conv1 -> layer3)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256 channels, H/4
        self.layer2 = resnet.layer2  # 512 channels, H/8
        self.layer3 = resnet.layer3  # 1024 channels, H/16
        self.layer4 = resnet.layer4  # 2048 channels, H/32
        
        # Channel dimensions for each stage
        self.channels = {
            'stem': 64,      # After conv1
            'layer1': 256,   # H/4
            'layer2': 512,   # H/8
            'layer3': 1024,  # H/16
            'layer4': 2048,  # H/32 (if used)
        }
        
        # MedKLIP projection layers (for compatibility)
        self.d_model = d_model
        num_ftrs = 1024 // 2  # MedKLIP uses 512
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, d_model)
        
        # Load pretrained weights if provided
        if pretrained_path:
            self._load_medklip_weights(pretrained_path)
            print(f"Loaded MedKLIP visual encoder from {pretrained_path}")
        
        if freeze:
            self._freeze_encoder()
            print("MedKLIP visual encoder is frozen")
    
    def _load_medklip_weights(self, path: str):
        """Load weights from MedKLIP checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        
        # Map MedKLIP keys to our structure
        # MedKLIP uses: res_features.0 -> conv1, res_features.1 -> bn1, etc.
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Visual encoder weights
            if key.startswith('res_features.'):
                # res_features is Sequential: [conv1, bn1, relu, maxpool, layer1, layer2, layer3]
                idx = int(key.split('.')[1])
                rest = '.'.join(key.split('.')[2:])
                
                if idx == 0:
                    new_key = f'conv1.{rest}' if rest else 'conv1'
                elif idx == 1:
                    new_key = f'bn1.{rest}' if rest else 'bn1'
                elif idx == 2:
                    continue  # relu, skip
                elif idx == 3:
                    continue  # maxpool, skip
                elif idx == 4:
                    new_key = f'layer1.{rest}'
                elif idx == 5:
                    new_key = f'layer2.{rest}'
                elif idx == 6:
                    new_key = f'layer3.{rest}'
                else:
                    continue
                    
                new_state_dict[new_key] = value
                
            elif key.startswith('res_l1.') or key.startswith('res_l2.'):
                new_state_dict[key] = value
        
        # Load with strict=False to allow missing layer4
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Missing keys (expected): {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    
    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass returning multi-scale features.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Dictionary with features at different scales:
            - 'stem': [B, 64, H/2, W/2]
            - 'layer1': [B, 256, H/4, W/4]
            - 'layer2': [B, 512, H/8, W/8]
            - 'layer3': [B, 1024, H/16, W/16]
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        stem = x  # [B, 64, H/2, W/2]
        
        x = self.maxpool(x)
        
        # Stages
        x1 = self.layer1(x)   # [B, 256, H/4, W/4]
        x2 = self.layer2(x1)  # [B, 512, H/8, W/8]
        x3 = self.layer3(x2)  # [B, 1024, H/16, W/16]
        
        return {
            'stem': stem,
            'layer1': x1,
            'layer2': x2,
            'layer3': x3,
        }
    
    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get patch embeddings (for cross-attention with text).
        Matches MedKLIP's image_encoder output.
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Patch embeddings [B, num_patches, d_model]
        """
        features = self.forward(x)
        x3 = features['layer3']  # [B, 1024, H/16, W/16]
        
        batch_size = x3.shape[0]
        # Rearrange to patches
        x3 = rearrange(x3, 'b d h w -> b (h w) d')  # [B, HW, 1024]
        
        # Project to d_model (MedKLIP style)
        h = rearrange(x3, 'b n d -> (b n) d')
        h = self.res_l1(h[:, :512])  # MedKLIP uses half features
        h = F.relu(h)
        h = self.res_l2(h)
        
        out = rearrange(h, '(b n) d -> b n d', b=batch_size)
        return out


class MedKLIPTextEncoder(nn.Module):
    """
    MedKLIP Text Encoder based on BioBERT/ClinicalBERT.
    
    Uses entity translation: converts disease names to detailed descriptions
    before encoding.
    
    Can load pretrained weights from MedKLIP checkpoint (jointly trained with visual).
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        max_length: int = 77,
        freeze: bool = True,
        pretrained_path: str = None,  # Path to MedKLIP checkpoint
    ):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.hidden_size = self.encoder.config.hidden_size  # 768 for BERT
        
        # Load MedKLIP pretrained text encoder weights if provided
        if pretrained_path:
            self._load_medklip_text_weights(pretrained_path)
            print(f"Loaded MedKLIP text encoder weights from {pretrained_path}")
        
        if freeze:
            self._freeze_encoder()
            print(f"MedKLIP text encoder ({model_name}) is frozen")
    
    def _load_medklip_text_weights(self, path: str):
        """Load text encoder weights from MedKLIP checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        
        # MedKLIP text encoder keys: text_encoder.* or bert.*
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # Check for text encoder weights
            if key.startswith('text_encoder.'):
                # Remove 'text_encoder.' prefix
                new_key = key.replace('text_encoder.', '')
                new_state_dict[new_key] = value
            elif key.startswith('bert.'):
                # Alternative naming: bert.*
                new_key = key.replace('bert.', '')
                new_state_dict[new_key] = value
        
        if new_state_dict:
            missing, unexpected = self.encoder.load_state_dict(new_state_dict, strict=False)
            print(f"  Text encoder - Loaded {len(new_state_dict)} keys")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print("  Warning: No text encoder weights found in checkpoint")
    
    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(
        self, 
        texts: List[str], 
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode texts using BERT.
        
        Args:
            texts: List of text strings
            device: Target device
            
        Returns:
            Tuple of:
            - token_embeddings: [B, seq_len, 768]
            - sentence_embeddings: [B, 768] (CLS token)
        """
        tokens = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad() if not any(p.requires_grad for p in self.encoder.parameters()) else torch.enable_grad():
            outputs = self.encoder(**tokens)
        
        token_emb = outputs.last_hidden_state  # [B, seq_len, 768]
        sent_emb = outputs.last_hidden_state[:, 0, :]  # CLS token [B, 768]
        
        return token_emb, sent_emb


# Entity descriptions from MedKLIP paper (knowledge base)
MEDKLIP_ENTITY_DESCRIPTIONS = {
    'normal': 'It means the absence of diseases and infirmity, indicating the structure is normal.',
    'clear': 'The lungs are clear and normal. No evidence for other diseases on lung.',
    'opacity': 'It is defined as an area of hazy opacification due to air displacement by fluid, airway collapse, fibrosis, or a neoplastic process.',
    'pneumothorax': 'A pneumothorax is an abnormal collection of air in the pleural space between the lung and the chest wall.',
    'edema': 'Pulmonary edema, also known as pulmonary congestion, is excessive liquid accumulation in the tissue and air spaces of the lungs.',
    'atelectasis': 'It is the collapse or closure of a lung resulting in reduced or absent gas exchange.',
    'consolidation': 'It is a region of normally compressible lung tissue that has filled with liquid instead of air.',
    'pneumonia': 'Pneumonia is an inflammatory condition of the lung primarily small air sacs known as alveoli. Pneumonia may present with opacities.',
    'cardiomegaly': 'Cardiomegaly is a medical condition in which the heart is enlarged.',
    'effusion': 'A pleural effusion is accumulation of excessive fluid in the pleural space.',
    'covid-19': 'It is a contagious disease caused by a virus. Ground-glass opacities, consolidation, thickening, pleural effusions commonly appear in infection.',
    'infection': 'An infection caused by bacteria, virus or other pathogens, often showing consolidation or opacity patterns.',
    'lung': 'The lungs are the primary organs of the respiratory system.',
    'heart': 'The heart is a muscular organ that pumps blood through the circulatory system.',
}


def translate_entity(entity: str) -> str:
    """
    Translate a medical entity to its description (Entity Translation from MedKLIP).
    
    Args:
        entity: Medical entity name (e.g., 'pneumonia')
        
    Returns:
        Detailed description of the entity
    """
    entity_lower = entity.lower().strip()
    
    if entity_lower in MEDKLIP_ENTITY_DESCRIPTIONS:
        return MEDKLIP_ENTITY_DESCRIPTIONS[entity_lower]
    
    # Default: return the entity name itself
    return f"Medical finding related to {entity}."


class MedKLIPEncoders(nn.Module):
    """
    Combined MedKLIP Visual + Text Encoders for IRIS integration.
    
    This is a drop-in replacement for IRIS's default encoders.
    """
    
    def __init__(
        self,
        visual_pretrained_path: Optional[str] = None,
        text_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        d_model: int = 256,
        max_token_len: int = 77,
        freeze_visual: bool = False,
        freeze_text: bool = True,
        use_entity_translation: bool = True
    ):
        super().__init__()
        
        self.visual_encoder = MedKLIPVisualEncoder(
            pretrained_path=visual_pretrained_path,
            d_model=d_model,
            freeze=freeze_visual
        )
        
        self.text_encoder = MedKLIPTextEncoder(
            model_name=text_model,
            max_length=max_token_len,
            freeze=freeze_text
        )
        
        self.use_entity_translation = use_entity_translation
        self.d_model = d_model
        self.text_hidden_size = self.text_encoder.hidden_size
        
        # Projection to match IRIS dimensions (768 -> 512 for compatibility with BCAM)
        self.text_proj = nn.Linear(self.text_hidden_size, 512)
    
    def encode_image(self, x: torch.Tensor) -> dict:
        """Encode image to multi-scale features."""
        return self.visual_encoder(x)
    
    def encode_text(
        self, 
        texts: List[str], 
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode texts with optional entity translation.
        
        Returns:
            - token_emb: [B, seq_len, 512]
            - sent_emb: [B, 512]
        """
        if self.use_entity_translation:
            texts = [translate_entity(t) if len(t.split()) <= 3 else t for t in texts]
        
        token_emb, sent_emb = self.text_encoder(texts, device)
        
        # Project to 512 dim
        token_emb = self.text_proj(token_emb)
        sent_emb = self.text_proj(sent_emb)
        
        return token_emb, sent_emb
    
    @property
    def visual_channels(self) -> dict:
        """Return channel dimensions for each visual encoder stage."""
        return self.visual_encoder.channels
