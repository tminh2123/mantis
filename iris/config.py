"""
IRIS Configuration
- Defaults in Python
- Dataset paths in YAML (configs/qata.yaml, configs/mosmed.yaml)
- Override with CLI args
"""
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import yaml


@dataclass
class DataConfig:
    root: str = ""
    train_dir: str = ""
    val_dir: str = ""
    test_dir: str = ""
    image_size: int = 224
    num_workers: int = 4


@dataclass
class ModelConfig:
    clip_model: str = "openai/clip-vit-base-patch32"
    max_token_len: int = 18
    frozen_clip: bool = True
    base_channel: int = 64
    num_heads: int = 8
    dropout: float = 0.5
    expansion: int = 4
    encoder_type: str = "cnn"  # "cnn" or "mix"
    deep_supervision: bool = True
    num_classes: int = 1
    text_mask_rate: float = 0.3
    img_mask_rate: float = 0.3


@dataclass
class PretrainConfig:
    enabled: bool = True
    epochs: int = 100
    batch_size: int = 64  # Reduced for mix encoder compatibility
    lr: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.07
    eps: float = 0.8
    checkpoint: str = ""


@dataclass
class TrainConfig:
    epochs: int = 300
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    bce_weight: float = 0.5
    grad_clip: float = 1.0
    cosine_lr: bool = True
    loss_weights: List[float] = field(default_factory=lambda: [0.5, 0.2, 0.15, 0.1, 0.05])
    asymmetric_weights: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9),
    ])
    early_stopping: bool = True
    es_patience: int = 10
    save_after: int = 50


@dataclass
class EvalConfig:
    batch_size: int = 16
    threshold: float = 0.5
    checkpoint: str = ""


@dataclass
class LoggingConfig:
    log_dir: str = ""
    checkpoint_dir: str = "output/checkpoints"
    save_every: int = 10


@dataclass
class Config:
    exp_name: str = ""
    dataset: str = ""  # qata or mosmed
    seed: int = 922
    device: str = "cuda"
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def parse_args():
    parser = argparse.ArgumentParser(description='IRIS Training')
    
    # Required
    parser.add_argument('--config', type=str, required=True, 
                        help='Dataset config (configs/qata.yaml or configs/mosmed.yaml)')
    parser.add_argument('--tag', type=str, required=True,
                        help='Experiment tag (e.g., baseline_cnn, attn_v2). '
                             'Output: output/{logs,checkpoints}/{dataset}_{tag}/')
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['pretrain', 'train', 'eval', 'full'])
    
    # Model overrides
    parser.add_argument('--encoder_type', type=str, choices=['cnn', 'mix'])
    parser.add_argument('--expansion', type=int)
    
    # Training overrides
    parser.add_argument('--no_pretrain', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--es_patience', type=int)
    
    # Paths
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--checkpoint', type=str,
                        help='Override checkpoint path for eval mode')
    
    # Other
    parser.add_argument('--resume', type=str)
    parser.add_argument('--seed', type=int)
    
    return parser.parse_args()


def load_config(args) -> Config:
    """Load config from YAML + apply CLI overrides."""
    config = Config()
    
    # Load dataset config from YAML
    with open(args.config, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    
    dataset_name = yaml_cfg.get('dataset', 'unknown')
    config.dataset = dataset_name
    config.data.root = yaml_cfg['data']['root']
    config.data.train_dir = yaml_cfg['data']['train_dir']
    config.data.val_dir = yaml_cfg['data']['val_dir']
    config.data.test_dir = yaml_cfg['data']['test_dir']
    
    # Apply CLI overrides
    if args.encoder_type:
        config.model.encoder_type = args.encoder_type
    if args.expansion:
        config.model.expansion = args.expansion
    if args.no_pretrain:
        config.pretrain.enabled = False
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.lr = args.lr
    if args.es_patience:
        config.train.es_patience = args.es_patience
    if args.seed:
        config.seed = args.seed
    
    # Experiment name: {dataset}_{tag}
    config.exp_name = f"{dataset_name}_{args.tag}"
    
    # Set output paths - each exp has its own folder
    output_dir = args.output_dir
    config.pretrain.checkpoint = f"{output_dir}/checkpoints/{config.exp_name}/pretrain.pth"
    config.eval.checkpoint = f"{output_dir}/checkpoints/{config.exp_name}/best.pth"
    config.logging.log_dir = f"{output_dir}/logs/{config.exp_name}"
    config.logging.checkpoint_dir = f"{output_dir}/checkpoints/{config.exp_name}"
    
    # Override checkpoint if specified (for eval mode)
    if args.checkpoint:
        config.eval.checkpoint = args.checkpoint
    
    return config
