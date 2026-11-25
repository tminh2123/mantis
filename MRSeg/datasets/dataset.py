"""
MRSeg dataset — grayscale variant of iris IRISDataset.

Supports:
  1. QaTa-COV19 (c2seg format): images/, masks/, texts.json
  2. MosMedData+ (LViT format): img/, labelcol/, *_text.xlsx
"""
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MRSegDataset(Dataset):
    """Grayscale medical image segmentation dataset with text prompts."""

    def __init__(self, root_dir: str, image_size: int = 224, augment: bool = False):
        self.root = Path(root_dir)
        self.size = image_size
        self.augment = augment

        self.img_dir = self._find_dir(['images', 'img'])
        self.mask_dir = self._find_dir(['masks', 'labelcol'])
        self.texts = self._load_texts()

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
        for name in candidates:
            path = self.root / name
            if path.exists():
                return path
        raise FileNotFoundError(f"None of {candidates} found in {self.root}")

    def _find_mask(self, stem: str) -> Optional[Path]:
        for prefix in ['', 'mask_']:
            for ext in ['.png', '.tif', '.tiff']:
                path = self.mask_dir / f"{prefix}{stem}{ext}"
                if path.exists():
                    return path
        return None

    def _load_texts(self) -> dict:
        json_file = self.root / "texts.json"
        if json_file.exists():
            with open(json_file) as f:
                return json.load(f)

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
        for key in [img_name, mask_name, f"mask_{img_name}",
                    img_name.replace('.png', ''), mask_name.replace('.png', '')]:
            if key in self.texts:
                return self.texts[key]
        return ""

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path, text = self.samples[idx]

        # Load as grayscale — output: (1, H, W)
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.size, self.size))

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0   # (1, H, W) in [0,1]
        mask = torch.from_numpy(mask).float().unsqueeze(0)              # (1, H, W)

        if self.augment:
            if random.random() > 0.5:
                image, mask = self._random_crop(image, mask)
            if random.random() > 0.5:
                image, mask, text = self._random_rot_flip(image, mask, text)
            if random.random() > 0.5:
                image, mask = self._random_rotate(image, mask)
            if random.random() > 0.5:
                image = self._random_jitter(image)

        # Per-image channel-wise normalization (z-score)
        mean = image.mean()
        std  = image.std() + 1e-6
        image = (image - mean) / std
        image = image.clamp(-3.0, 3.0)

        return {'image': image, 'mask': mask, 'text': text, 'path': str(img_path)}

    def _swap_left_right(self, text: str) -> str:
        text = text.replace('left', '__LEFT__').replace('right', '__RIGHT__')
        return text.replace('__LEFT__', 'right').replace('__RIGHT__', 'left')

    def _random_crop(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        import torchvision.transforms.functional as TF
        _, H, W = image.shape
        scale    = random.uniform(0.75, 1.0)
        crop_h   = int(H * scale)
        crop_w   = int(W * scale)
        top      = random.randint(0, H - crop_h)
        left     = random.randint(0, W - crop_w)
        image = TF.crop(image, top, left, crop_h, crop_w)
        mask  = TF.crop(mask,  top, left, crop_h, crop_w)
        image = TF.resize(image, [H, W], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        mask  = TF.resize(mask,  [H, W], interpolation=TF.InterpolationMode.NEAREST)
        return image, mask

    def _random_jitter(self, image: torch.Tensor) -> torch.Tensor:
        # Brightness: add uniform offset
        brightness = random.uniform(-0.15, 0.15)
        image = image + brightness
        # Contrast: scale around mean
        contrast = random.uniform(0.75, 1.25)
        mean = image.mean()
        image = (image - mean) * contrast + mean
        return image.clamp(0.0, 1.0)

    def _random_rot_flip(
        self, image: torch.Tensor, mask: torch.Tensor, text: str
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        k = random.randint(0, 3)
        image = torch.rot90(image, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[1, 2])
        if random.random() > 0.5:
            axis = random.choice([1, 2])
            image = torch.flip(image, dims=[axis])
            mask = torch.flip(mask, dims=[axis])
            if axis == 2:
                text = self._swap_left_right(text)
        return image, mask, text

    def _random_rotate(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        import torchvision.transforms.functional as TF
        angle = random.uniform(-30, 30)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


def get_dataloader(
    root_dir: str,
    batch_size: int,
    image_size: int = 224,
    augment: bool = False,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = MRSegDataset(root_dir=root_dir, image_size=image_size, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )
