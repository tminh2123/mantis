# Environment Context

## Local vs Server

- **Local machine** (macOS): Không có GPU và torch, chỉ dùng để edit code
- **Server**: JupyterHub với GPU A100 40GB, dùng để train/test model
- Khi cần test code, hãy nhờ user copy file lên server và chạy thử

## Server Info

- Server type: JupyterHub
- GPU: NVIDIA A100 40GB
- Server path: `/dscilab_hvch261_tuanminh/workspace/code/c2seg/`
- Dataset path: `../../dataset/QaTa-COV19/QaTa-COV19-v2/c2seg/`

## Project Structure

```
c2seg/
├── config.py           # Configuration
├── run.py              # Training/eval script
├── datasets/           # Dataset loading
├── models/             # Model architectures
│   ├── c2seg.py        # Main C2Seg model
│   ├── kan.py          # KAN layers
│   ├── losses.py       # Loss functions
│   └── unet.py         # U-Net backbone
├── utils/              # Utilities (metrics, etc.)
├── logs/               # Training logs
├── output/             # Outputs
│   └── checkpoints/    # Model checkpoints
└── docs/               # Documentation
```

## Workflow

1. Edit code trên local
2. User copy files lên server
3. User chạy test/train trên server
4. User report kết quả/lỗi về

## Files to Copy

Khi có thay đổi, thường cần copy các files sau:
- `config.py` - Configuration
- `run.py` - Training script
- `models/` - Model architecture
- `datasets/` - Dataset loading
- `utils/` - Utilities
