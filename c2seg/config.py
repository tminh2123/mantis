"""
C2Seg Configuration
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    root: str = "../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg"
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4


@dataclass
class ModelConfig:
    # CLIP
    clip_model: str = "ViT-B/32"
    
    # KAN
    kan_grid_size: int = 3
    kan_spline_order: int = 2
    
    # Architecture
    encoder_channels: list = field(default_factory=lambda: [64, 128, 256, 512])
    decoder_channels: list = field(default_factory=lambda: [256, 128, 64, 32])


@dataclass
class TrainConfig:
    epochs: int = 300
    lr: float = 1e-4
    weight_decay: float = 1e-4
    
    # Scheduler
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 10
    
    # Loss weights
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    
    # Checkpointing
    checkpoint_dir: str = "output/checkpoints"
    save_every: int = 10
    
    # Logging
    log_every: int = 10
    eval_every: int = 1


@dataclass 
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    seed: int = 42
    device: str = "cuda"
    
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            seed=d.get("seed", 42),
            device=d.get("device", "cuda"),
        )
