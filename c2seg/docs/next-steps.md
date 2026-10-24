# C2Seg - Next Steps

## Decisions

- **KAN**: Không sử dụng nữa do không cải thiện đáng kể so với MLP

## Proposed Improvements

### 1. Visual Encoder Backbone

Hiện tại backbone khá đơn giản. Cần thử:
- **MedCLIP** hoặc các pretrained model trên medical images
- Backbone có Vision Transformer (ViT) thay vì chỉ CNN
- Ablation study so sánh các backbone

### 2. Fusion Module

- Thử thêm **CBAM** (Convolutional Block Attention Module) vào fusion
- So sánh với BCAM hiện tại

## Experiment Plan

| Order | Experiment | Description |
|-------|------------|-------------|
| 1 | Baseline | Chạy với config hiện tại (MLP + DeepSup) |
| 2 | Backbone ablation | Thử MedCLIP / ViT backbone |
| 3 | Fusion ablation | Thử CBAM fusion module |
| 4 | Pretrain ablation | So sánh pretrained vs scratch |

## New Project

- Tên: **IRIS** (Image Segmentation)
- Folder: `iris/`
- Ý nghĩa: Mống mắt - kết hợp vision và medical imaging
- Status: **Implemented** - Baseline version with MLP-only (no KAN)
