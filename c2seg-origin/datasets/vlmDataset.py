# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class VlmDataset(Dataset):
    def __init__(self, data_file, augment=False):
        self.data = torch.load(data_file)
        self.keys = list(self.data.keys())
        self.augment = augment

    def __len__(self):
        return len(self.keys)

    def random_rot_flip(self, image, mask):
        """Random rotation by 0/90/180/270 + random flip"""
        k = random.randint(0, 3)
        image = torch.rot90(image, k, dims=[1, 2])   # [C,H,W]
        mask = torch.rot90(mask, k, dims=[0, 1])     # [H,W]

        if random.random() > 0.5:
            axis = random.choice([1, 2])  # horizontal or vertical
            image = torch.flip(image, dims=[axis])
            mask = torch.flip(mask, dims=[axis-1])
        return image, mask

    def random_rotate(self, image, mask):
        """Random small-angle rotation [-30,30]"""
        angle = random.uniform(-30, 30)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        if mask.shape[0] == 1:
            mask = mask.squeeze(0)
        return image, mask

    def __getitem__(self, idx):
        key = self.keys[idx]
        sample = self.data[key]

        image = sample["image"].clone()       # [C,H,W], float
        mask = sample["mask"].clone()         # [H,W], long
        text_token = sample["text_token"].clone()
        text_mask = sample["text_mask"].clone()
        roi_t = sample["roi_t"].clone()

        # ---- augmentation (only for training) ----
        if self.augment:
            if random.random() > 0.5:
                image, mask = self.random_rot_flip(image, mask)
            if random.random() > 0.5:
                image, mask = self.random_rotate(image, mask)

        return {"image": image,
                "mask": mask,
                "text_token": text_token,
                "text_mask": text_mask,
                "roi_t": roi_t,
                }, key

