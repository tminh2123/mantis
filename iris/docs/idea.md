# README: Delta-BCAM — Linear Associative Memory for Text-Guided Medical Image Segmentation

## 1. Project Overview & Architectural Vision
Delta-BCAM is a specialized cross-modal interaction module designed to replace the conventional Bidirectional Complementary Attention Module (BCAM) within the C2Seg medical segmentation framework. By reformulating dense pixel-token cross-attention into an online associative memory governed by the Widrow-Hoff Delta rule and matched-filter readout, Delta-BCAM resolves semantic leakage in asymmetric anatomical guidance while reducing cross-attention computational complexity from quadratic to linear relative to spatial resolution.

```
       [Text L (N x C)] ──> Text KAN Projections ──> S_0 (Text-Anchored State, d x d)
                                                               │
                                                               ▼
[Visual Features V (P x C)] ──> Visual KAN Projections ──> [Spatial Scan / Chunk Scan]
                                                               │
                     ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
                     ▼                                                                                   ▼
            [Readout Phase]                                                                     [Delta State Update]
       q_p = γ_p ⊙ KAN_q(v_p)                                                             \tilde{v}_p = v_p - k_p S_{p-1}
       F_v[p] = q_p · S_{p-1}                                                             S_p = (I - k_p^T k_p)S_{p-1} + k_p^T \tilde{v}_p
                     │                                                                                   │
                     ▼                                                                                   ▼
       F_v: Image Conditioned on Text                                                      F_l: Differential Spatial Mask
                     └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                                               ▼
                                                 [K-Gate Bilateral Suppression]
```

---

## 2. Theoretical Motivation: The "Semantic Leakage" Defect in Softmax Attention
In text-guided referring segmentation, dense Softmax Cross-Attention generates feature representations through weighted sums across token vocabularies:

$$F_{\text{out}}[p] = \sum_{j=1}^N \frac{\exp(q_p k_j^\top / \sqrt{d})}{\sum_{m=1}^N \exp(q_p k_m^\top / \sqrt{d})} v_j$$

### The Concrete Failure Case (Bilateral Entity vs. Unilateral Directive)
Consider an image showing bilateral structures (e.g., both left and right lungs, or both hands) paired with a clinical text prompt specifying a unilateral lesion (e.g., `"consolidation in left lower lobe"`):
1. **Entity Token Activation**: The anatomical noun token (`"lobe"` or `"hand"`) exhibits high visual-semantic affinity with both sides of the patient body. Consequently, key-query similarity scores spike symmetrically across both left and right regions.
2. **Localization Token Attenuation**: The spatial modifier token (`"left"`) focuses attention on the left hemisphere. However, in the weighted sum calculation over the token dimension, the right hemisphere continues to accumulate substantial projection magnitude from the noun token (`"lobe"`).
3. **Semantic Leakage / False Positives**: Softmax normalization fails to subtract or zero out the erroneous entity activation in non-target regions. The decoder receives non-zero lesion representations at the contralateral site, yielding spurious false-positive masks or boundary confusion.

### The Delta Rule Solution
The Widrow-Hoff Delta rule frames memory updates as an orthogonal projection contraction:

$$S_t = (\mathbf{I} - k_t^\top k_t) S_{t-1} + k_t^\top v_t$$

When an anatomical key vector $k_t$ is processed under a conflicting spatial constraint, the operator $(\mathbf{I} - k_t^\top k_t)$ explicitly erases prior correlated memory energy along that directional subspace before writing updated residual innovations.

---

## 3. Mathematical Formulation of Delta-BCAM

Delta-BCAM takes two inputs:
* Visual features: $V \in \mathbb{R}^{B \times P \times C}$, where $P = H \times W$ from the multi-scale visual backbone.
* Clinical text representations: $L \in \mathbb{R}^{B \times N \times C}$, extracted from biomedical text encoders.

### Step 1: Text-Anchored Associative Initialization ($S_0$)
Because medical prompts contain compact token sequences ($N \ll P$, typically $N \in [15, 45]$), linguistic tokens are compiled into a compact state matrix $S_0 \in \mathbb{R}^{B \times d \times d}$:

$$K_l = \text{KAN}_k(L) \in \mathbb{R}^{B \times N \times d}, \quad V_l = \text{KAN}_v(L) \in \mathbb{R}^{B \times N \times d}$$

Normalize key vectors under Euclidean norm:

$$\bar{k}_j = \frac{K_{l, j}}{\|K_{l, j}\|_2 + \epsilon}$$

Accumulate linguistic concepts into the initial associative state:

$$S_0 = \sum_{j=1}^N \bar{k}_j^\top V_{l, j} \in \mathbb{R}^{B \times d \times d}$$

### Step 2: Matched-Filter Readout ($F_v$)
Pixels are traversed sequentially or in 2D spatial chunks. For spatial coordinate $p \in [1, P]$:

$$q_p = \text{KAN}_q(V_p) \in \mathbb{R}^{B \times 1 \times d}$$

To prevent slow-decaying or high-norm associative channels from dominating low-frequency clinical clues, apply a matched-filter dynamic gate $\gamma_p$:

$$\gamma_p = \sigma(W_\gamma V_p) \in \mathbb{R}^{B \times 1 \times d}$$

$$\hat{q}_p = \gamma_p \odot q_p$$

Extract language-steered visual representation:

$$F_v[p] = \hat{q}_p \cdot S_{p-1} \in \mathbb{R}^{B \times 1 \times d}$$

### Step 3: Progressive Delta State Update ($F_l$)
Visual features are projected to construct instantaneous pixel-level keys and values:

$$k_p = \frac{\text{KAN}_{k,\text{img}}(V_p)}{\|\text{KAN}_{k,\text{img}}(V_p)\|_2 + \epsilon}, \quad v_p = \text{KAN}_{v,\text{img}}(V_p)$$

Compute residual innovation (unaccounted visual variance):

$$\tilde{v}_p = v_p - k_p S_{p-1} \in \mathbb{R}^{B \times 1 \times d}$$

Update associative state matrix via rank-1 orthogonal erasure and selective write:

$$S_p = (\mathbf{I} - k_p^\top k_p) S_{p-1} + k_p^\top \tilde{v}_p \in \mathbb{R}^{B \times d \times d}$$

The spatial localization map reflects innovation magnitudes:

$$F_l[p] = \tilde{v}_p \in \mathbb{R}^{B \times 1 \times d}$$

---

## 4. Integration with C2Seg Decoder & K-Gate
Delta-BCAM preserves strict tensor interface compatibility with standard C2Seg pipelines. Outputs $F_v, F_l \in \mathbb{R}^{B \times P \times d}$ feed into Kolmogorov-Arnold Network Attention Gates:

$$g_v = \tanh(\text{KAN}_g(F_v)), \quad g_l = \tanh(\text{KAN}_g(F_l))$$

$$F_{\text{fused}} = \text{Conv}_{1 \times 1}\big(\text{concat}(F_v \odot g_v, \, F_l \odot g_l)\big)$$

---

## 5. Dataset Exploration Checklist
Before benchmark execution, verify target dataset characteristics across two benchmark corpora:

| Dataset | Modality | Primary Evaluation Focus | Key Exploration Target |
| :--- | :--- | :--- | :--- |
| **QaTa-COV19** | Chest X-ray (2D) | Ground-glass & bilateral consolidation | Proportion of unilateral prompts (`left`, `right`, `lower`, `upper`) |
| **MosMedData+** | CT Slices (2D/3D) | Multi-lobe viral pneumonia lesions | Cross-talk between symmetrical pulmonary segments |

**Target Verification Script**: Extract regex-based clinical tags from annotation files (`left`, `right`, `bilateral`, `single`, `multiple`). Compute false-positive lesion mask rates in contralateral non-target lung lobes comparing standard BCAM versus Delta-BCAM.

---

## 6. Implementation Specification for Coding Agents

### PyTorch Module Blueprint
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeltaBCAM(nn.Module):
    def __init__(self, in_channels: int, state_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        
        # Text Projections
        self.kan_text_k = nn.Linear(in_channels, state_dim)
        self.kan_text_v = nn.Linear(in_channels, state_dim)
        
        # Visual Projections
        self.kan_vis_q = nn.Linear(in_channels, state_dim)
        self.kan_vis_k = nn.Linear(in_channels, state_dim)
        self.kan_vis_v = nn.Linear(in_channels, state_dim)
        
        # Matched-Filter Readout Projection
        self.gamma_proj = nn.Linear(in_channels, state_dim)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)

    def forward(self, V: torch.Tensor, L: torch.Tensor):
        # V: [B, P, C], L: [B, N, C]
        B, P, C = V.shape
        N = L.shape[1]
        
        # Step 1: Text-Anchored Memory Initialization
        Kl = F.normalize(self.kan_text_k(L), p=2, dim=-1) # [B, N, d]
        Vl = self.kan_text_v(L)                            # [B, N, d]
        S = torch.bmm(Kl.transpose(1, 2), Vl)              # [B, d, d]
        
        # Visual Projections
        Qv = self.kan_vis_q(V)                             # [B, P, d]
        Kv = F.normalize(self.kan_vis_k(V), p=2, dim=-1)   # [B, P, d]
        Vv = self.kan_vis_v(V)                             # [B, P, d]
        Gamma = torch.sigmoid(self.gamma_proj(V))          # [B, P, d]
        
        F_v = []
        F_l = []
        
        # Step 2 & 3: Sequential or Chunkwise Associative Recurrence
        for p in range(P):
            q_p = (Gamma[:, p:p+1, :] * Qv[:, p:p+1, :])  # [B, 1, d]
            k_p = Kv[:, p:p+1, :]                          # [B, 1, d]
            v_p = Vv[:, p:p+1, :]                          # [B, 1, d]
            
            # Readout
            f_v_p = torch.bmm(q_p, S)                      # [B, 1, d]
            F_v.append(f_v_p)
            
            # Delta Residual Innovation
            pred_v = torch.bmm(k_p, S)                     # [B, 1, d]
            v_tilde = v_p - pred_v                         # [B, 1, d]
            F_l.append(v_tilde)
            
            # Orthogonal Erasure & Write
            S = S - torch.bmm(k_p.transpose(1, 2), pred_v) + torch.bmm(k_p.transpose(1, 2), v_p)
            
        F_v = torch.cat(F_v, dim=1) # [B, P, d]
        F_l = torch.cat(F_l, dim=1) # [B, P, d]
        
        return F_v, F_l
```
