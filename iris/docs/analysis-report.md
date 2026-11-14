# IRIS Analysis Report - QaTa-COV19 Dataset

## Current Status

- Best model: MedKLIP + ClinicalBERT + BiFMD (h8_e4)
- Best Dice: 84.61%
- Target: Beat C2Seg paper (85.25%)

---

## 1. Text Ablation Study

Evaluating how much the model relies on text vs visual features.

### Results

| Text Condition | Dice | Std | Delta |
|----------------|------|-----|-------|
| Correct text | 84.61% | 16.68% | baseline |
| Random text | 69.46% | 24.61% | -15.15% |
| Fixed text ("pulmonary infection") | TBD | - | - |
| Empty text | TBD | - | - |

### Key Findings

1. **Text IS important**: Random text drops 15% Dice
   - Model genuinely uses text information, not just visual features
   
2. **Variance increases with wrong text**: Std 16.68% -> 24.61%
   - Some samples are robust to text, others are not

### Interpretation

- ~15% of performance comes from text guidance
- Improving text understanding has potential for gains
- Focus areas: location words (left/right, upper/lower)

---

## 2. Dice Score Distribution

### Statistics

| Metric | Value |
|--------|-------|
| Samples | 2113 |
| Mean | 84.61% |
| Std | 16.68% |
| Median | 90.24% |
| Min | 0.00% |
| Max | 98.81% |

### Percentiles

| Percentile | Dice |
|------------|------|
| 5th | 51.03% |
| 10th | 65.63% |
| 25th | 80.09% |
| 50th (median) | 90.24% |
| 75th | 95.86% |
| 90th | 97.17% |
| 95th | 97.51% |

### Distribution by Range

| Range | Count | Percentage |
|-------|-------|------------|
| Failed (0-10%) | 22 | 1.0% |
| Very Poor (10-30%) | 20 | 0.9% |
| Poor (30-50%) | 59 | 2.8% |
| Below Avg (50-70%) | 172 | 8.1% |
| Average (70-85%) | 461 | 21.8% |
| Good (85-95%) | 740 | 35.0% |
| Excellent (95-100%) | 639 | 30.2% |

### Key Observations

1. **Bimodal-ish distribution**: 
   - 65% samples are "Good" or "Excellent" (>85%)
   - But 5% samples are "Poor" or worse (<50%)
   
2. **Mean vs Median gap**: Mean 84.61% < Median 90.24%
   - Left-skewed by ~100 bad samples dragging mean down
   
3. **Worst samples pattern**: All 0% Dice cases are "Unilateral, one infected area"
   - Model fails on small, localized infections
   - These need precise location understanding

4. **Improvement potential**: 
   - Fixing ~100 worst samples (< 50%) could boost mean by 2-3%
   - These are likely text-understanding failures

---

## 3. Dataset Analysis

### Token Statistics (ClinicalBERT)

| Split | Samples | Unique Texts | Duplication |
|-------|---------|--------------|-------------|
| Train | 5716 | 273 | 95.2% |
| Test | 2113 | 197 | 90.7% |

### Token Length Distribution

- Min: 12 tokens
- Max: 21 tokens  
- Mean: 15.9 tokens
- Model max_token_len: 18 (effective: 16)

### Truncation Issue

| Split | Truncated | Percentage |
|-------|-----------|------------|
| Train | 1061 | 18.6% |
| Test | 521 | 24.7% |

Truncated texts are typically the most complex (multiple regions).

### Dataset Quality Issues

1. **Low diversity**: Only 273 unique texts for 5716 samples
2. **High duplication**: 95% samples share text with others
3. **Top 2 texts**: Cover 48% of all data
4. **Typos**: "thrr", "lwoer", "midldle" in original data
5. **Truncation**: 18-25% samples lose location info

---

## 3. Error Analysis - Where Model Goes Wrong

### By Text Category

Analyzed all 2113 test samples by text attributes:

| Category | Samples | Mean Dice | Worst (<50%) |
|----------|---------|-----------|--------------|
| Bilateral | 1675 | 88.2% | 2.0% |
| **Unilateral** | 438 | 70.9% | **15.5%** |
| Two areas | 1594 | 88.8% | 1.8% |
| **One area** | 417 | 70.6% | **15.8%** |
| Both sides | 1666 | 88.3% | 1.9% |
| **Left only** | 238 | 68.9% | **18.9%** |
| Right only | 208 | 72.9% | 11.5% |

Main problem: **"Unilateral, one infected area"** - model fails 15.9% of the time.

### Prediction Confusion (101 worst samples)

Compared what model predicts vs ground truth:

| Type | Correct | Wrong | Wrong Rate |
|------|---------|-------|------------|
| Left/Right | 58 | 15 | 20.5% |
| Upper/Middle/Lower | 53 | 17 | 24.3% |
| Number of regions | 85 | 16 | 15.8% |

Patterns found:
- **Left/Right confusion**: Model predicts left when GT is right (10/15 cases)
- **Vertical confusion**: Model often predicts middle when GT is upper
- **Over-segmentation**: Model predicts 2 regions when GT has 1 (12/16 cases)

### Dataset Quality Check

Checked if text labels match ground truth position:

| Dimension | Consistent | Inconsistent |
|-----------|------------|--------------|
| Left/Right | 97.5% | 2.5% (11 samples) |
| Upper/Middle/Lower | 42.4% | 57.6% (392 samples) |

- Left/right labels are reliable
- Vertical labels don't match well (probably different definition of "upper/middle/lower")

### Key Takeaways

1. Model works well on bilateral/two-area cases (89% dice)
2. Model struggles with single small lesions (71% dice)
3. Model has slight bias toward predicting left side
4. Model tends to over-segment (sees extra regions that don't exist)
5. ~80% of worst cases are correct on position - main issue is missing the lesion entirely

---

## 4. Attention Visualization

### Observations

- Combined attention focuses on image **corners/edges** instead of lung regions
- "left" and "right" tokens have **nearly identical** attention patterns
- Model doesn't differentiate laterality well

### Possible Causes

1. Position embedding bias at corners
2. Text encoder (ClinicalBERT) doesn't capture spatial meaning
3. Cross-attention dominated by visual features

---

## 5. Architecture Experiments

### Linear Memory Attention

| Config | Memory Size | Dice | Delta |
|--------|-------------|------|-------|
| Baseline | 0 | 84.61% | - |
| memory_m64 | 64 | 84.16% | -0.45% |
| memory_m128 | 128 | TBD | - |

Linear memory didn't help - may need different approach.

---

## 6. Recommendations

### Short-term

1. **Increase max_token_len to 24**: Avoid truncation of complex texts
2. **Clean text data**: Fix typos in training data
3. **Text augmentation**: Paraphrase to increase diversity

### Medium-term

1. **Location-aware text encoding**: Add explicit position encoding for spatial words
2. **Separate left/right embeddings**: Force model to distinguish laterality
3. **Auxiliary task**: Predict location as classification head

### Long-term

1. **Try different dataset**: MosMedData+ may have better text quality
2. **Spatial text grounding**: Pre-train on text-location pairs

---

## 7. Next Steps

1. [ ] Complete text ablation (fixed, empty)
2. [ ] Increase max_token_len and retrain
3. [ ] Analyze per-location Dice (left vs right, upper vs lower)
4. [ ] Try explicit location encoding

---

*Last updated: Based on exp/linear-memory branch experiments*
