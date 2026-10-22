# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from modules.module_clip import load_clip_model
from modules.pixlevel_attention import PixLevelModule
from nets.BCAM import BCAModule


# -----------------------------
# Basic building blocks
# -----------------------------
class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


def conv_stack(in_channels, out_channels, num_layers):
    """Stack of Conv-BN-ReLU layers"""
    layers = [ConvBNReLU(in_channels, out_channels)]
    layers += [ConvBNReLU(out_channels, out_channels) for _ in range(num_layers - 1)]
    return nn.Sequential(*layers)


class DownBlock(nn.Module):
    """Down-sampling block with stacked conv layers"""
    def __init__(self, in_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = conv_stack(in_channels, out_channels, num_layers)

    def forward(self, x):
        return self.convs(x)


class UpBlockAttention(nn.Module):
    """Up-sampling block with skip-connection"""
    def __init__(self, skip_channels, up_channels, out_channels, num_layers=2):
        super().__init__()
        self.att = PixLevelModule(skip_channels)
        self.up = nn.ConvTranspose2d(up_channels, up_channels, kernel_size=2, stride=2)
        self.convs = conv_stack(skip_channels + up_channels, out_channels, num_layers)

    def forward(self, skip_x, x):
        skip_att = self.att(skip_x)
        up_x = self.up(x)
        out = torch.cat([skip_att, up_x], dim=1)
        return self.convs(out)


# -----------------------------
# Main Network
# -----------------------------
class MBINet(nn.Module):
    """Multiscale Bidirectional Interaction Network"""
    def __init__(self, config, n_channels=3, n_classes=1, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        base_ch = config.base_channel
        img_scales = [224, 112, 56, 28, 14]

        # Vision encoder
        self.pool = nn.MaxPool2d(2, 2)
        self.down0 = DownBlock(n_channels, base_ch)
        self.down1 = DownBlock(base_ch, base_ch * 2)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)
        self.down4 = DownBlock(base_ch * 8, base_ch * 8)

        # Vision decoder
        self.up3 = UpBlockAttention(base_ch * 8, base_ch * 8, base_ch * 4)
        self.up2 = UpBlockAttention(base_ch * 4, base_ch * 4, base_ch * 2)
        self.up1 = UpBlockAttention(base_ch * 2, base_ch * 2, base_ch)
        self.up0 = UpBlockAttention(base_ch, base_ch, base_ch)

        # Language encoder (frozen CLIP backbone)
        self.clip = load_clip_model(config)

        # Cross-modal fusion
        self.fuse1 = BCAModule(img_size=img_scales[1], in_channels=base_ch,     num_heads=8)
        self.fuse2 = BCAModule(img_size=img_scales[2], in_channels=base_ch * 2, num_heads=8)
        self.fuse3 = BCAModule(img_size=img_scales[3], in_channels=base_ch * 4, num_heads=8)
        self.fuse4 = BCAModule(img_size=img_scales[4], in_channels=base_ch * 8, num_heads=8)

        # Output heads
        self.out0 = nn.Sequential(nn.Conv2d(base_ch, n_classes, 1), nn.Sigmoid())
        if self.deep_supervision:
            self.out1 = nn.Sequential(nn.Conv2d(base_ch,     n_classes, 1), nn.Sigmoid())
            self.out2 = nn.Sequential(nn.Conv2d(base_ch * 2, n_classes, 1), nn.Sigmoid())
            self.out3 = nn.Sequential(nn.Conv2d(base_ch * 4, n_classes, 1), nn.Sigmoid())
            self.out4 = nn.Sequential(nn.Conv2d(base_ch * 8, n_classes, 1), nn.Sigmoid())

    def forward(self, image, text_token, text_mask):
        # Encode text
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        _, l = self.clip.encode_text(text_token, mask=text_mask, return_hidden=True)

        # Encode image (vision stream)
        v0 = self.down0(image)                # [B, C,  224, 224]
        v1 = self.down1(self.pool(v0))        # [B, 2C, 112, 112]
        v2 = self.down2(self.pool(v1))        # [B, 4C,  56,  56]
        v3 = self.down3(self.pool(v2))        # [B, 8C,  28,  28]
        v4 = self.down4(self.pool(v3))        # [B, 8C,  14,  14]

        # Cross-modal fusion + decoder
        f4 = self.fuse4(v4, l)

        up3 = self.up3(v3, f4)
        f3 = self.fuse3(up3, l)

        up2 = self.up2(v2, f3)
        f2 = self.fuse2(up2, l)

        up1 = self.up1(v1, f2)
        f1 = self.fuse1(up1, l)

        up0 = self.up0(v0, f1)
        out0 = self.out0(up0)

        # Deep supervision (if enabled)
        if self.deep_supervision:
            return (out0,
                    self.out1(f1),
                    self.out2(f2),
                    self.out3(f3),
                    self.out4(f4))
        return out0

