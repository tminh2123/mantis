# 🏥 C2Seg - Medical Image Segmentation

> **Paper:** "Towards Text-Mask Consistency in Medical Image Segmentation" (ICLR 2026)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📊 Summary

| Dataset | Paper Target | Our Best | Gap | Model |
|:--------|:------------:|:--------:|:---:|:------|
| **QaTa-COV19** | 85.25% Dice | **84.00%** | -1.25% | `qata_c2seg_mlp_deepsup` |
| **QaTa-COV19** | 74.13% mIoU | **75.23%** | **+1.10%** ✨ | `qata_c2seg_mlp_deepsup` |
| **MosMedData+** | 77.81% Dice | **75.12%** | -2.69% | `mosmed_c2seg_full` |

---

## 📈 Experiment Results

### QaTa-COV19 Dataset (Chest X-ray)

<table>
<thead>
<tr>
<th align="left">🏷️ Tag</th>
<th align="center">Description</th>
<th align="center">Scale</th>
<th align="center">KAN</th>
<th align="center">Dice ↑</th>
<th align="center">mIoU ↑</th>
<th align="center">HD95 ↓</th>
<th align="center">ASSD ↓</th>
</tr>
</thead>
<tbody>
<tr style="background: #f0f7ff;">
<td><strong>📄 Paper</strong></td>
<td>C2Seg official target</td>
<td align="center">1.0</td>
<td align="center">✅</td>
<td align="center"><strong>85.25%</strong></td>
<td align="center">74.13%</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
<tr>
<td><code>qata_custom_kan</code></td>
<td>Custom impl, B-spline KAN</td>
<td align="center">1.0</td>
<td align="center">✅</td>
<td align="center">80.37%</td>
<td align="center">70.83%</td>
<td align="center">21.30</td>
<td align="center">4.30</td>
</tr>
<tr>
<td><code>qata_custom_kan_lite</code></td>
<td>Custom impl, no base spline</td>
<td align="center">1.0</td>
<td align="center">✅</td>
<td align="center">79.38%</td>
<td align="center">69.51%</td>
<td align="center">24.45</td>
<td align="center">4.92</td>
</tr>
<tr>
<td><code>qata_ukan_0.5</code></td>
<td>U-KAN block, scale 0.5</td>
<td align="center">0.5</td>
<td align="center">✅</td>
<td align="center">79.46%</td>
<td align="center">69.71%</td>
<td align="center">21.69</td>
<td align="center">4.47</td>
</tr>
<tr>
<td><code>qata_custom_mlp</code></td>
<td>Custom impl, full MLP</td>
<td align="center">1.0</td>
<td align="center">❌</td>
<td align="center">69.94%</td>
<td align="center">57.70%</td>
<td align="center">46.36</td>
<td align="center">8.97</td>
</tr>
<tr>
<td><code>qata_c2seg_kan</code></td>
<td>C2Seg origin + KAN K-Gate</td>
<td align="center">1.0</td>
<td align="center">✅</td>
<td align="center">78.87%</td>
<td align="center">69.45%</td>
<td align="center">28.68</td>
<td align="center">6.15</td>
</tr>
<tr>
<td><code>qata_c2seg_kan_0.5</code></td>
<td>C2Seg origin + KAN, scale 0.5</td>
<td align="center">0.5</td>
<td align="center">✅</td>
<td align="center">78.43%</td>
<td align="center">68.73%</td>
<td align="center">29.54</td>
<td align="center">6.36</td>
</tr>
<tr>
<td><code>qata_c2seg_mlp</code></td>
<td>C2Seg origin + MLP K-Gate</td>
<td align="center">1.0</td>
<td align="center">❌</td>
<td align="center">83.81%</td>
<td align="center">75.02%</td>
<td align="center">28.68</td>
<td align="center">6.15</td>
</tr>
<tr style="background: #e6ffe6;">
<td>🏆 <strong><code>qata_c2seg_mlp_deepsup</code></strong></td>
<td>C2Seg + MLP + DeepSup</td>
<td align="center">1.0</td>
<td align="center">❌</td>
<td align="center"><strong>84.00%</strong></td>
<td align="center"><strong>75.23%</strong></td>
<td align="center"><strong>15.53</strong></td>
<td align="center"><strong>3.03</strong></td>
</tr>
</tbody>
</table>

### MosMedData+ Dataset (CT Scan)

<table>
<thead>
<tr>
<th align="left">🏷️ Tag</th>
<th align="center">Description</th>
<th align="center">Scale</th>
<th align="center">KAN</th>
<th align="center">Dice ↑</th>
<th align="center">mIoU ↑</th>
<th align="center">HD95 ↓</th>
<th align="center">ASSD ↓</th>
</tr>
</thead>
<tbody>
<tr style="background: #f0f7ff;">
<td><strong>📄 Paper</strong></td>
<td>C2Seg official target</td>
<td align="center">1.0</td>
<td align="center">✅</td>
<td align="center"><strong>~77.81%</strong></td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
<tr>
<td><code>mosmed_c2seg_mlp</code></td>
<td>C2Seg origin + MLP K-Gate</td>
<td align="center">1.0</td>
<td align="center">❌</td>
<td align="center">72.59%</td>
<td align="center">58.97%</td>
<td align="center">18.44</td>
<td align="center">3.43</td>
</tr>
<tr>
<td><code>mosmed_c2seg_mlp_deepsup</code></td>
<td>C2Seg + MLP + DeepSup</td>
<td align="center">1.0</td>
<td align="center">❌</td>
<td align="center">73.74%</td>
<td align="center">60.23%</td>
<td align="center">21.06</td>
<td align="center">4.20</td>
</tr>
<tr>
<td><code>mosmed_c2seg_kan_deepsup</code></td>
<td>C2Seg + KAN + DeepSup</td>
<td align="center">1.0</td>
<td align="center">✅</td>
<td align="center">74.04%</td>
<td align="center">60.76%</td>
<td align="center">18.20</td>
<td align="center">3.53</td>
</tr>
<tr>
<td><code>mosmed_c2seg_kan_deepsup_einsum</code></td>
<td>C2Seg + KAN + DeepSup + einsum</td>
<td align="center">1.0</td>
<td align="center">✅</td>
<td align="center">70.32%</td>
<td align="center">56.84%</td>
<td align="center">19.70</td>
<td align="center">4.94</td>
</tr>
<tr style="background: #e6ffe6;">
<td>🏆 <strong><code>mosmed_c2seg_full</code></strong></td>
<td>C2Seg full sync (latest)</td>
<td align="center">1.0</td>
<td align="center">✅</td>
<td align="center"><strong>75.12%</strong></td>
<td align="center">TBD</td>
<td align="center">TBD</td>
<td align="center">TBD</td>
</tr>
</tbody>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         C2Seg Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │  Input   │    │  Visual Encoder │    │      BCAM        │   │
│  │ Image    │───▶│   (5 DownBlocks)│───▶│  (Cross Attn +   │   │
│  │ 224×224  │    │   Conv-based    │    │   K-Gate Fusion) │   │
│  └──────────┘    └─────────────────┘    └────────┬─────────┘   │
│                                                   │              │
│  ┌──────────┐    ┌─────────────────┐              │              │
│  │  Text    │───▶│  CLIP Encoder   │──────────────┘              │
│  │ Prompt   │    │   (Frozen)      │                             │
│  └──────────┘    └─────────────────┘                             │
│                                                                  │
│                         ┌──────────────────┐                     │
│                         │     Decoder      │                     │
│                         │  (4 UpBlocks +   │                     │
│                         │  PixLevelModule) │                     │
│                         └────────┬─────────┘                     │
│                                  │                               │
│                    ┌─────────────┴─────────────┐                 │
│                    ▼                           ▼                 │
│              ┌──────────┐              ┌───────────────┐         │
│              │ Main Out │              │Deep Supervision│         │
│              │ 224×224  │              │(out1-4 scales)│         │
│              └──────────┘              └───────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

📐 See detailed diagrams: [`c2seg/docs/`](c2seg/docs/index.html)

---

## 🚀 Quick Start

### Training Commands

```bash
# ═══════════════════════════════════════════════════════════════
# QaTa-COV19 Dataset
# ═══════════════════════════════════════════════════════════════

# MLP + DeepSup (recommended)
python run.py --stage pretrain --config configs/qata.yaml --tag qata_c2seg_mlp_deepsup --no-kan
python run.py --stage train    --config configs/qata.yaml --tag qata_c2seg_mlp_deepsup --no-kan
python run.py --stage eval     --config configs/qata.yaml --tag qata_c2seg_mlp_deepsup --no-kan

# ═══════════════════════════════════════════════════════════════
# MosMedData+ Dataset
# ═══════════════════════════════════════════════════════════════

# KAN + DeepSup (recommended)
python run.py --stage pretrain --config configs/mosmed.yaml --tag mosmed_c2seg_full --accum-steps 2
python run.py --stage train    --config configs/mosmed.yaml --tag mosmed_c2seg_full --accum-steps 2
python run.py --stage eval     --config configs/mosmed.yaml --tag mosmed_c2seg_full
```

### Evaluation from Checkpoint

```bash
# Auto-detect tag from checkpoint path
python run.py --stage eval --config configs/qata.yaml \
    --checkpoint output/checkpoints/qata_c2seg_mlp_deepsup/c2seg_best.pth
```

---

## 📁 Project Structure

```
c2seg/
├── models/
│   ├── c2seg.py          # Main model architecture
│   ├── kan.py            # KANLinear implementation
│   ├── losses.py         # Loss functions (CaCL, DeepSup)
│   └── unet.py           # U-Net baseline
├── datasets/
│   └── dataset.py        # Data loading
├── utils/
│   └── seg_metrics.py    # Dice, mIoU, HD95, ASSD
├── configs/
│   ├── qata.yaml         # QaTa-COV19 config
│   └── mosmed.yaml       # MosMedData+ config
├── docs/                 # Architecture diagrams
│   ├── index.html
│   ├── c2seg-overall-architecture.html
│   ├── c2seg-component-details.html
│   └── c2seg-training-stages.html
└── run.py                # Training/eval script
```

---

## 📚 References

- **Paper:** "Towards Text-Mask Consistency in Medical Image Segmentation" (C2Seg)
- **Datasets:**
  - [QaTa-COV19](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset) - Chest X-ray COVID-19 segmentation
  - [MosMedData+](https://mosmed.ai/) - CT scan COVID-19 segmentation

---

<p align="center">
  <i>Last updated: October 2026</i>
</p>
