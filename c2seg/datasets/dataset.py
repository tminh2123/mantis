"""
Dataset for C2Seg segmentation task.
Synced with c2seg-origin logic.

Supports both folder structures:
    1. QaTa-COV19 style:
        root/
          images/   *.png | *.jpg
          masks/    mask_*.png
          texts.json

    2. MosMedData+/LViT style:
        root/
          img/       *.png | *.tif
          labelcol/  *.png
          *_text.xlsx
"""

import json
import os
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class C2SegDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        image_size: int = 224,
        augment: bool = False,
    ):
        self.root = Path(root_dir)
        self.size = image_size
        self.augment = augment

        # Find image and mask directories
        self.img_dir = self._find_dir(['images', 'img'])
        self.mask_dir = self._find_dir(['masks', 'labelcol'])
        
        # Load text mapping (supports both json and xlsx)
        self.texts = self._load_texts()

        # Collect samples
        self.samples = []
        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
                continue
            mask_path = self._find_mask(img_path.stem)
            if mask_path is None:
                continue
            text = self._get_text(img_path.name, mask_path.name)
            self.samples.append((img_path, mask_path, text))

    def _find_dir(self, candidates: list) -> Path:
        """Find first existing directory from candidates."""
        for name in candidates:
            path = self.root / name
            if path.exists():
                return path
        raise FileNotFoundError(f"None of {candidates} found in {self.root}")

    def _find_mask(self, stem: str) -> Path:
        """Find mask file with various naming conventions."""
        # Try: stem.png, mask_stem.png, stem.tif, etc.
        for prefix in ['', 'mask_']:
            for ext in ['.png', '.tif', '.tiff']:
                path = self.mask_dir / f"{prefix}{stem}{ext}"
                if path.exists():
                    return path
        return None

    def _load_texts(self) -> dict:
        """Load text annotations from json or xlsx."""
        # Try JSON first
        json_file = self.root / "texts.json"
        if json_file.exists():
            with open(json_file) as f:
                return json.load(f)
        
        # Try Excel (LViT format)
        xlsx_files = list(self.root.glob("*_text.xlsx")) + list(self.root.glob("*_text.xls"))
        if xlsx_files:
            import pandas as pd
            df = pd.read_excel(xlsx_files[0])
            texts = {}
            for _, row in df.iterrows():
                # Column names: "Image" and "Description" (LViT format)
                img_name = str(row.iloc[0])  # First column is image name
                desc = str(row.iloc[1]) if len(row) > 1 else ""  # Second column is description
                texts[img_name] = desc.strip()
            return texts
        
        # No text file found - return empty dict (will use empty strings)
        print(f"Warning: No text annotations found in {self.root}")
        return {}

    def _get_text(self, img_name: str, mask_name: str) -> str:
        """Get text for image, trying multiple key formats."""
        # Try different key formats
        for key in [img_name, mask_name, f"mask_{img_name}", 
                    img_name.replace('.png', ''), mask_name.replace('.png', '')]:
            if key in self.texts:
                return self.texts[key]
        return ""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, text = self.samples[idx]

        # Load image with cv2 (matches c2seg-origin)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.size, self.size))
        
        # Binarize mask (matches c2seg-origin)
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # [C, H, W], 0-1
        mask = torch.from_numpy(mask).long()  # [H, W], 0 or 1

        # Augmentation (matches c2seg-origin)
        if self.augment:
            if random.random() > 0.5:
                image, mask = self._random_rot_flip(image, mask)
            if random.random() > 0.5:
                image, mask = self._random_rotate(image, mask)

        return image, mask, text

    def _random_rot_flip(self, image: torch.Tensor, mask: torch.Tensor):
        """Random rotation by 0/90/180/270 + random flip (matches c2seg-origin)"""
        # Random 90-degree rotation
        k = random.randint(0, 3)
        image = torch.rot90(image, k, dims=[1, 2])  # [C, H, W]
        mask = torch.rot90(mask, k, dims=[0, 1])    # [H, W]

        # Random flip (horizontal or vertical)
        if random.random() > 0.5:
            axis = random.choice([1, 2])  # 1=vertical, 2=horizontal for image
            image = torch.flip(image, dims=[axis])
            mask = torch.flip(mask, dims=[axis - 1])  # 0=vertical, 1=horizontal for mask
        
        return image, mask

    def _random_rotate(self, image: torch.Tensor, mask: torch.Tensor):
        """Random small-angle rotation [-30, 30] (matches c2seg-origin)"""
        import torchvision.transforms.functional as TF
        
        angle = random.uniform(-30, 30)
        
        # Rotate image
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        # Rotate mask (need to add channel dim for TF.rotate)
        mask = mask.unsqueeze(0).float()  # [1, H, W]
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        mask = mask.squeeze(0).long()  # [H, W]
        
        return image, mask
