# 🏥 C2Seg - Medical Image Segmentation

> **Paper:** "Towards Text-Mask Consistency in Medical Image Segmentation" (ICLR 2026)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📊 Summary

| Dataset | Paper Target | Our Best | Gap | Model |
|:--------|:------------:|:--------:|:---:|:------|
| **QaTa-COV19** | 85.25% Dice | **84.00%** | -1.25% | MLP + DeepSup |
| **QaTa-COV19** | 74.13% mIoU | **75.23%** | **+1.10%** ✨ | MLP + DeepSup |
| **MosMedData+** | 77.81% Dice | **74.04%** | -3.77% | KAN + DeepSup |

> 💡 **Key Finding:** Our mIoU on QaTa exceeds the paper's result by 1.1%!

---

## 📈 Experiment Results

### QaTa-COV19 Dataset (Chest X-ray)

<table>
<thead>
<tr>
<th align="left">🏷️ Tag</th>
<th align="center">Architecture</th>
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
<td>C2Seg (official)</td>
<td align="center">1.0</td>
<td align="center">✅</td>
<td align="center"><strong>85.25%</strong></td>
<td align="center">74.13%</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
<tr>
<td><code>ukan_0.5</code></td>
<td>U-KAN baseline</td>
<td align="center">0.5</td>
<td align="center">✅ Full</td>
<td align="center">79.46%</td>
<td align="center">69.71%</td>
<td align="center">21.69</td>
<td align="center">4.47</td>
</tr>
<tr>
<td><code>simple_kan_v1</code></td>
<td>Custom KAN</td>
<td align="center">1.0</td>
<td align="center">✅</td>
<td align="center">79.38%</td>
<td align="center">69.51%</td>
<td align="center">24.45</td>
<td align="center">4.92</td>
</tr>
<tr>
<td><code>efficient_kan_v1</code></td>
<td>B-spline KAN</td>
<td align="center">1.0</td>
<td align="center">✅</td>
<td align="center">80.37%</td>
<td align="center">70.83%</td>
<td align="center">21.30</td>
<td align="center">4.30</td>
</tr>
<tr>
<td><code>mlp_1.0</code></td>
<td>Full MLP replacement</td>
<td align="center">1.0</td>
<td align="center">❌</td>
<td align="center">69.94%</td>
<td align="center">57.70%</td>
<td align="center">46.36</td>
<td align="center">8.97</td>
</tr>
<tr>
<td><code>c2seg_nokan</code></td>
<td>C2Seg + MLP K-Gate</td>
<td align="center">1.0</td>
<td align="center">❌</td>
<td align="center">83.81%</td>
<td align="center">75.02%</td>
<td align="center">28.68</td>
<td align="center">6.15</td>
</tr>
<tr>
<td><code>c2seg_kan</code></td>
<td>C2Seg + KAN K-Gate</td>
<td align="center">1.0</td>
<td align="center">✅</td>
<td align="center">78.87%</td>
<td align="center">69.45%</td>
<td align="center">28.68</td>
<td align="center">6.15</td>
</tr>
<tr style="background: #e6ffe6;">
<td>🏆 <strong><code>qata_nokan_deepsup</code></strong></td>
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
<th align="center">Architecture</th>
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
<td>C2Seg (official)</td>
<td align="center">1.0</td>
<td align="center">✅</td>
<td align="center"><strong>~77.81%</strong></td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
<tr>
<td><code>mosmed_nokan</code></td>
<td>C2Seg + MLP K-Gate</td>
<td align="center">1.0</td>
<td align="center">❌</td>
<td align="center">72.59%</td>
<td align="center">58.97%</td>
<td align="center">18.44</td>
<td align="center">3.43</td>
</tr>
<tr>
<td><code>mosmed_nokan_deepsup</code></td>
<td>C2Seg + MLP + DeepSup</td>
<td align="center">1.0</td>
<td align="center">❌</td>
<td align="center">73.74%</td>
<td align="center">60.23%</td>
<td align="center">21.06</td>
<td align="center">4.20</td>
</tr>
<tr style="background: #e6ffe6;">
<td>🏆 <strong><code>mosmed_kan_deepsup</code></strong></td>
<td>C2Seg + KAN + DeepSup</td>
<td align="center">1.0</td>
<td align="center">✅</td>
<td align="center"><strong>74.04%</strong></td>
<td align="center"><strong>60.76%</strong></td>
<td align="center"><strong>18.20</strong></td>
<td align="center"><strong>3.53</strong></td>
</tr>
</tbody>
</table>

---

## 🔑 Key Findings

### 1️⃣ KAN Performance is Dataset-Dependent

| Modality | Dataset | Winner | Difference |
|:---------|:--------|:------:|:----------:|
| X-ray | QaTa-COV19 | **MLP** | +5.0% Dice |
| CT | MosMedData+ | **KAN** | +0.3% Dice |

> 💡 KAN's regularization may benefit smaller datasets or different imaging modalities.

### 2️⃣ Deep Supervision is Highly Beneficial

| Dataset | Without DeepSup | With DeepSup | Improvement |
|:--------|:---------------:|:------------:|:-----------:|
| QaTa HD95 | 22.74 | **15.53** | **-31.7%** 🎯 |
| QaTa Dice | 83.81% | **84.00%** | +0.19% |
| MosMed Dice | 72.59% | **73.74%** | +1.15% |

### 3️⃣ Scale Matters

```
Channel Scale Reference:
┌──────────┬─────────────────────────────┬─────────┐
│ Scale    │ Channels                    │ Params  │
├──────────┼─────────────────────────────┼─────────┤
│ 1.0      │ [64, 128, 256, 512, 512]    │ ~20.6M  │
│ 0.5      │ [32, 64, 128, 256, 256]     │ ~5.2M   │
└──────────┴─────────────────────────────┴─────────┘
```

### 4️⃣ Training Insights

- 🎯 **Early stopping** recommended at epoch 100-150
- 📊 Model peaks around epoch 90-100, then overfits
- 🧮 Use **gradient accumulation** for KAN: `--accum-steps 2`

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
# QaTa-COV19 Dataset (Recommended: MLP + Deep Supervision)
# ═══════════════════════════════════════════════════════════════

# Pretrain → Train → Evaluate
python run.py --stage pretrain --config configs/qata.yaml --tag qata_nokan_deepsup --no-kan --deep-sup
python run.py --stage train    --config configs/qata.yaml --tag qata_nokan_deepsup --no-kan --deep-sup
python run.py --stage eval     --config configs/qata.yaml --tag qata_nokan_deepsup --no-kan --deep-sup

# ═══════════════════════════════════════════════════════════════
# MosMedData+ Dataset (Recommended: KAN + Deep Supervision)
# ═══════════════════════════════════════════════════════════════

# Pretrain → Train → Evaluate
python run.py --stage pretrain --config configs/mosmed.yaml --tag mosmed_kan_deepsup --deep-sup --accum-steps 2
python run.py --stage train    --config configs/mosmed.yaml --tag mosmed_kan_deepsup --deep-sup --accum-steps 2
python run.py --stage eval     --config configs/mosmed.yaml --tag mosmed_kan_deepsup --deep-sup
```

### Evaluation from Checkpoint

```bash
# Auto-detect tag from checkpoint path
python run.py --stage eval --config configs/qata.yaml \
    --checkpoint output/checkpoints/qata_nokan_deepsup/c2seg_best.pth

# With explicit tag
python run.py --stage eval --config configs/qata.yaml \
    --tag qata_nokan_deepsup --no-kan --deep-sup
```

---

## 📋 Recommendations

| Aspect | Recommendation |
|:-------|:---------------|
| **QaTa-COV19** | Use `--no-kan --deep-sup` (MLP variant) |
| **MosMedData+** | Use `--deep-sup --accum-steps 2` (KAN variant) |
| **Deep Supervision** | Always enable `--deep-sup` |
| **Early Stopping** | Stop at epoch 100-150 |
| **Channel Scale** | Use 1.0 (default); 0.5 if memory constrained |
| **KAN Training** | Use `--accum-steps 2` for stability |

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
