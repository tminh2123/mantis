# MRSeg | MICCAI 2026 | TIA

**Multimodal Routing and Region Refinement for Language-Guided Medical Image Segmentation (MRSeg)** has been accepted to MICCAI 2026 (TIA). This is the official implementation of MRSeg.


## Environment Setup

Please create a Python 3.8 environment first, then install the required dependencies from `requirements.txt`.

```bash
conda create -n mrseg python=3.8 -y
conda activate mrseg
pip install -r requirements.txt
```

## Dataset Preparation

This project uses the following datasets:

### 1. QaTa-COV19
Original source: [QaTa-COV19 on Kaggle](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)

### 2. MosMedData+
Original source: [MosMedData+](http://medicalsegmentation.com/covid19/)

### 3. Text Annotations
The text annotations for **QaTa-COV19** and **MosMedData+** can be obtained from the resources provided by **LViT**:

https://github.com/HUANGLIZI/LViT

We thank **Li et al.** for making these resources publicly available.

## Pretrained Backbones

### Vision Backbone
- **ConvNeXt-Tiny**: [facebook/convnext-tiny-224](https://huggingface.co/facebook/convnext-tiny-224/tree/main)

### Language Backbone
- **BiomedBERT**: [microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)

## Configuration

Model and training settings can be modified in:

```bash
config/train.yaml
```

Please ensure the dataset paths and runtime options are correctly configured before training or evaluation.

## Training

Run:

```bash
python train.py --config ./config/train.yaml
```

## Evaluation

Pretrained checkpoints will be publicly shared later.

### MosMedData+

```bash
python evaluate.py --config ./config/train.yaml --ckpt ./pretrained_models/MRSeg_MosMedplus.ckpt
```

### QaTa-COV19

```bash
python evaluate.py --config ./config/train.yaml --ckpt ./pretrained_models/MRSeg_QaTa-Covid19.ckpt
```

## Results

| Dataset | Dice (%) | mIoU (%) |
|---|---:|---:|
| QaTa-COV19 | **90.90** | **83.32** |
| MosMedData+ | **81.53** | **68.82** |

MRSeg uses **7.11M trainable parameters** and **7.60 GFLOPs**.

## Acknowledgment

We thank the authors of [**LViT**](https://github.com/HUANGLIZI/LViT) for making their code, text annotations, and related resources publicly available.

## Citation

If you find this repository helpful, please consider citing our paper.

```bibtex
@inproceedings{rahman2026mrseg,
  title={Multimodal Routing and Region Refinement for Language-Guided Medical Image Segmentation},
  author={Rahman, Md Maklachur and Banna, Md Hasan Al and Anjum, Saraf and Arnob, Assame and Hammond, Tracy},
  booktitle={MICCAI 2026},
  year={2026}
}
```
