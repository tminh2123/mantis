"""
Dataset for IRIS segmentation task.
Ported from C2Seg dataset.

Supports folder structures:
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
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class IRISDataset(Dataset):
    """Medical image segmentation dataset with text prompts."""
    
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
        
        # Load text mapping
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

    def _find_dir(self, candidates: List[str]) -> Path:
        """Find first existing directory from candidates."""
        for name in candidates:
            path = self.root / name
            if path.exists():
                return path
        raise FileNotFoundError(f"None of {candidates} found in {self.root}")

    def _find_mask(self, stem: str) -> Optional[Path]:
        """Find mask file with various naming conventions."""
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
                img_name = str(row.iloc[0])
                desc = str(row.iloc[1]) if len(row) > 1 else ""
                texts[img_name] = desc.strip()
            return texts
        
        print(f"Warning: No text annotations found in {self.root}")
        return {}

    def _get_text(self, img_name: str, mask_name: str) -> str:
        """Get text for image, trying multiple key formats."""
        for key in [img_name, mask_name, f"mask_{img_name}", 
                    img_name.replace('.png', ''), mask_name.replace('.png', '')]:
            if key in self.texts:
                return self.texts[key]
        return ""

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path, mask_path, text = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.size, self.size))
        
        # Binarize mask
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                image, mask = self._random_rot_flip(image, mask)
            if random.random() > 0.5:
                image, mask = self._random_rotate(image, mask)

        return image, mask, text

    def _random_rot_flip(
        self, 
        image: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random rotation by 0/90/180/270 + random flip."""
        k = random.randint(0, 3)
        image = torch.rot90(image, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[0, 1])

        if random.random() > 0.5:
            axis = random.choice([1, 2])
            image = torch.flip(image, dims=[axis])
            mask = torch.flip(mask, dims=[axis - 1])
        
        return image, mask

    def _random_rotate(
        self, 
        image: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random small-angle rotation [-30, 30]."""
        import torchvision.transforms.functional as TF
        
        angle = random.uniform(-30, 30)
        
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        mask = mask.unsqueeze(0).float()
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        mask = mask.squeeze(0).long()
        
        return image, mask


def get_dataloader(
    root_dir: str,
    batch_size: int,
    image_size: int = 224,
    augment: bool = False,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create dataloader for a split."""
    
    dataset = IRISDataset(
        root_dir=root_dir,
        image_size=image_size,
        augment=augment,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,  # Drop last for training
    )
