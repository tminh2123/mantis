"""
IRIS Configuration
Ported from C2Seg config.
"""
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import yaml


@dataclass
class DataConfig:
    root: str = "../../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg"
    train_dir: str = "../../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg/train"
    val_dir: str = "../../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg/val"
    test_dir: str = "../../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg/test"
    image_size: int = 224
    num_workers: int = 4


@dataclass
class ModelConfig:
    # CLIP
    clip_model: str = "openai/clip-vit-base-patch32"
    max_token_len: int = 18
    frozen_clip: bool = True
    
    # Architecture
    base_channel: int = 64
    num_heads: int = 8
    dropout: float = 0.5
    expansion: int = 4  # MLP expansion ratio (Transformer FFN style)
    encoder_type: str = "cnn"  # "cnn" or "mix"
    deep_supervision: bool = True
    num_classes: int = 1
    
    # Masking (for training)
    text_mask_rate: float = 0.3
    img_mask_rate: float = 0.3


@dataclass
class PretrainConfig:
    enabled: bool = True  # Whether to do pretraining
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.07
    eps: float = 0.8
    checkpoint: str = "output/checkpoints/iris_pretrain.pth"


@dataclass
class TrainConfig:
    epochs: int = 500
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    bce_weight: float = 0.5
    grad_clip: float = 1.0
    cosine_lr: bool = True
    
    # Deep supervision weights
    loss_weights: List[float] = field(default_factory=lambda: [0.5, 0.2, 0.15, 0.1, 0.05])
    asymmetric_weights: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.5, 0.5),
        (0.4, 0.6),
        (0.3, 0.7),
        (0.2, 0.8),
        (0.1, 0.9),
    ])
    
    # Early stopping
    early_stopping: bool = True
    es_patience: int = 50
    save_after: int = 50


@dataclass
class EvalConfig:
    batch_size: int = 16
    threshold: float = 0.5
    checkpoint: str = "output/checkpoints/iris_best.pth"


@dataclass
class LoggingConfig:
    log_dir: str = "output/logs"
    checkpoint_dir: str = "output/checkpoints"
    save_every: int = 10


@dataclass
class Config:
    # Experiment
    exp_name: str = "iris_baseline"
    seed: int = 922
    device: str = "cuda"
    
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load config from yaml file and merge with defaults."""
    config = Config()
    
    if config_path:
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Update nested configs
        if 'project' in yaml_config:
            if 'name' in yaml_config['project']:
                config.exp_name = yaml_config['project'].get('exp_name', yaml_config['project']['name'])
            if 'seed' in yaml_config['project']:
                config.seed = yaml_config['project']['seed']
                
        if 'data' in yaml_config:
            for k, v in yaml_config['data'].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)
                    
        if 'model' in yaml_config:
            for k, v in yaml_config['model'].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
                    
        if 'pretrain' in yaml_config:
            for k, v in yaml_config['pretrain'].items():
                if hasattr(config.pretrain, k):
                    setattr(config.pretrain, k, v)
                    
        if 'train' in yaml_config:
            for k, v in yaml_config['train'].items():
                if hasattr(config.train, k):
                    setattr(config.train, k, v)
                    
        if 'eval' in yaml_config:
            for k, v in yaml_config['eval'].items():
                if hasattr(config.eval, k):
                    setattr(config.eval, k, v)
                    
        if 'logging' in yaml_config:
            for k, v in yaml_config['logging'].items():
                if hasattr(config.logging, k):
                    setattr(config.logging, k, v)
    
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='IRIS Training')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['pretrain', 'train', 'eval', 'full'])
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    return parser.parse_args()
