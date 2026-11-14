# Delta-BCAM: Linear Associative Delta-Memory with Matched-Filter Readout for Multimodal Medical Image Segmentation

## 1. Problem Formulation: Limitations of Softmax Cross-Attention in BCAM

In the baseline C2Seg framework, multimodal interaction between medical image features and clinical linguistic representations at decoder scale $\omega$ is governed by the Bidirectional Complementary Attention Module (BCAM). Let the spatial visual feature map be denoted as $V \in \mathbb{R}^{P \times C}$, where $P = H_\omega \times W_\omega$ represents the number of spatial pixels and $C$ denotes the channel dimension. Let the clinical textual sequence representation be denoted as $L \in \mathbb{R}^{N \times C}$, where $N$ is the number of linguistic tokens.

BCAM projects both modalities through Kolmogorov-Arnold Network (KAN) layers to generate key and value representations:

$$V_{\text{key}} = \text{KAN}(V), \quad V_{\text{value}} = \text{KAN}(V) \quad \in \mathbb{R}^{P \times C}$$

$$L_{\text{key}} = \text{KAN}(L), \quad L_{\text{value}} = \text{KAN}(L) \quad \in \mathbb{R}^{N \times C}$$

The multimodal affinity matrix $A$ is computed via full dot-product cross-correlation:

$$A = \frac{1}{\sqrt{C}} V_{\text{key}} (L_{\text{key}})^\top \in \mathbb{R}^{P \times N}$$

From this single affinity matrix, two directional representations are derived:

$$A^V = \text{Softmax}_{\text{row}}(A) \in \mathbb{R}^{P \times N}, \quad F_v = A^V L_{\text{value}} \in \mathbb{R}^{P \times C}$$

$$A^L = \text{Softmax}_{\text{col}}(A)^\top \in \mathbb{R}^{N \times P}, \quad F_l = \text{Conv}_{1 \times 1}\big(\text{reshape}(A^L V_{\text{value}})\big) \in \mathbb{R}^{P \times C}$$

While BCAM establishes bidirectional alignment, it exhibits critical theoretical and computational drawbacks:

* **Quadratic Cross-Attention Footprint**: The generation and storage of matrix $A$ incurs $\mathcal{O}(P \cdot N)$ computational complexity and memory overhead. At higher spatial resolutions (large $P$) or when scaling to 3D volumetric scans ($P = D \times H \times W$) and extended clinical reports (large $N$), this footprint poses substantial GPU memory pressure.
* **Diffusion of Rare Clinical Descriptors**: Normalized row-wise softmax across $N$ tokens enforces a weighted convex sum over linguistic concepts. In medical contexts where critical diagnostic indicators (such as subtle margin irregularities or localized ground-glass opacities) are represented by only one or two rare tokens, their signal becomes diffused when averaged against frequent non-discriminative clinical phrasing.
* **Absence of Progressive State Retention**: Every pixel independently computes its dot product with the entire token vocabulary without maintaining an associative memory state of what anatomical anomalies have already been detected across adjacent spatial coordinates.

---

## 2. Mathematical Foundations: Delta Rule and Matched-Filter Readout

To overcome the scaling and diffusion limitations of standard cross-attention, the Delta-BCAM module reformulates multimodal alignment through an online linear associative memory matrix $S \in \mathbb{R}^{d \times d}$.

### 2.1 The Gated Delta Rule

In contrast to conventional additive linear attention mechanisms that continuously accumulate outer products ($S_t = S_{t-1} + k_t^\top v_t$) leading to representation saturation, the Delta rule (Widrow-Hoff learning rule) treats memory updating as an online error-correcting gradient descent step.

Given an input key vector $k_t \in \mathbb{R}^{1 \times d}$ normalized such that $\|k_t\|_2 = 1$, and target value vector $v_t \in \mathbb{R}^{1 \times d}$, the reconstruction loss on the current memory state is formulated as:

$$\mathcal{L}_t(S) = \frac{1}{2} \| k_t S - v_t \|_2^2$$

Evaluating the gradient of $\mathcal{L}_t$ with respect to the state matrix $S$ yields:

$$\nabla_S \mathcal{L}_t(S) = k_t^\top (k_t S - v_t)$$

Performing a step of gradient descent with a unit learning rate produces the Delta update rule:

$$S_t = S_{t-1} - k_t^\top (k_t S_{t-1} - v_t) = \left(\mathbf{I} - k_t^\top k_t\right) S_{t-1} + k_t^\top v_t$$

By defining the effective write error $\tilde{v}_t$ as:

$$\tilde{v}_t = v_t - k_t S_{t-1}$$

the update decomposes into two explicit mathematical phases:

$$S_t = S_{t-1} + k_t^\top \tilde{v}_t$$

* **Contraction / Erasure Phase**: The projection operator $(\mathbf{I} - k_t^\top k_t)$ contracts the memory state along the subspace spanned by $k_t$, explicitly purging obsolete or redundant associations stored under that key.
* **Innovation / Write Phase**: The outer product $k_t^\top \tilde{v}_t$ injects solely the unpredicted novelty vector $\tilde{v}_t$ into memory. If the state already reconstructs the incoming value accurately ($k_t S_{t-1} \approx v_t$), $\tilde{v}_t \to 0$ and the memory remains stable.

### 2.2 Sensitivity-Allocated Gating and Matched-Filter Readout

In recurrent gating architectures, state dynamics are governed by channel-wise decay factors $\alpha_t \in (0, 1)^d$, erase gates $b_t \in (0, 1)^d$, and write gates $w_t \in (0, 1)^d$:

$$S_t = \left(\mathbf{I} - k_t^\top (b_t \odot k_t)\right) \text{Diag}(\alpha_t) S_{t-1} + k_t^\top (w_t \odot v_t)$$

Theoretical analysis of the Delta recurrence establishes that state trajectories are substantially more sensitive to perturbations in the write pathway than in the erasure pathway. Because key activations after non-linear gating concentrate on a sparse subset of coordinates (approximately 5% of channels), the effective contraction rank is strictly bounded. Consequently, the erase parameterization can be constrained to a low-rank subspace while reserving full expressivity for the write transformations.

Furthermore, state channels decay at heterogeneous rates $\alpha_{t, j}$. Over long recurrence sequences, slow-decaying channels accumulate norm energies orders of magnitude larger than fast-decaying channels, introducing severe row-energy imbalance. To ensure that readout queries extract unbiased associative signals without prematurely triggering aggressive state erasure, the readout query $q_t$ is dynamically modulated via a matched-filter gate $\gamma_t$:

$$\hat{q}_t = \gamma_t \odot q_t$$

The matched-filter coefficients follow a fractional power-law scaling $\gamma_{t, j} \propto \rho_j^{-p}$ with $p \in (0, 1)$, attenuating high-energy saturated channels while amplifying subtle high-frequency signals.

---

## 3. The Delta-BCAM Module Architecture

Delta-BCAM replaces the dense $P \times N$ cross-attention matrix by structuring the multimodal interaction into two coordinated mathematical stages: Linguistic State Anchoring and Spatial Delta Scanning with Matched-Filter Readout.

### 3.1 Linguistic State Anchoring (Text-to-Memory Initialization)

Because clinical descriptions contain a compact set of tokens ($N \ll P$), linguistic features are directly compiled into a grounded clinical memory matrix $S_0 \in \mathbb{R}^{d \times d}$.

The linguistic token embeddings $L \in \mathbb{R}^{N \times C}$ are projected through KAN transformation layers into key and value spaces:

$$K_l = \text{KAN}_k^L(L) \in \mathbb{R}^{N \times d}, \quad V_l = \text{KAN}_v^L(L) \in \mathbb{R}^{N \times d}$$

Each key vector is normalized under the $L_2$ norm along the channel dimension to ensure numeric stability and bounded spectrum:

$$\bar{k}_{l, j} = \frac{K_{l, j}}{\|K_{l, j}\|_2 + \epsilon}, \quad \forall j \in \{1, \dots, N\}$$

The initial state matrix $S_0$ is synthesized via the associative outer-product sum:

$$S_0 = \sum_{j=1}^N \bar{k}_{l, j}^\top V_{l, j} \in \mathbb{R}^{d \times d}$$

The matrix $S_0$ serves as an explicit semantic dictionary encoding the full spectrum of pathological descriptors, anatomical locations, and structural relations present in the patient notes.

### 3.2 Spatial Delta Scanning and Matched-Filter Readout

The spatial feature map $V \in \mathbb{R}^{H_\omega \times W_\omega \times C}$ is serialized into a spatial sequence of pixels $V = \{v_p\}_{p=1}^P$, where $P = H_\omega W_\omega$. Serialization can follow a standard raster scan order, a Hilbert space-filling curve, or multi-directional sweeps (horizontal, vertical, and diagonal).

At each spatial coordinate $p \in \{1, \dots, P\}$:

1. **Query and Gate Projection**:
   The pixel feature $v_p \in \mathbb{R}^{1 \times C}$ is mapped to a query representation $q_p$ and a matched-filter modulation vector $\gamma_p$:
   $$q_p = \text{KAN}_q^V(v_p) \in \mathbb{R}^{1 \times d}$$
   $$\gamma_p = \sigma\big(W_\gamma v_p + b_\gamma\big) \in \mathbb{R}^{1 \times d}$$
   where $\sigma(\cdot)$ denotes the sigmoid activation function. The modulated query is computed as:
   $$\hat{q}_p = \gamma_p \odot q_p$$

2. **Vision-Dominant Path ($F_v$ Readout)**:
   The modulated query extracts the corresponding pathological guidance directly from the active associative state $S_{p-1}$:
   $$F_v[p] = \hat{q}_p S_{p-1} \in \mathbb{R}^{1 \times d}$$
   This vector represents the visual coordinate $p$ conditioned on the clinical guidance stored in memory.

3. **Spatial Key-Value Projections and Delta Error**:
   The pixel is concurrently mapped to its key and target value representations:
   $$k_p = \frac{\text{KAN}_k^V(v_p)}{\|\text{KAN}_k^V(v_p)\|_2 + \epsilon} \in \mathbb{R}^{1 \times d}, \quad v_p^{\text{val}} = \text{KAN}_v^V(v_p) \in \mathbb{R}^{1 \times d}$$
   The expected value predicted by the existing memory state is retrieved as $k_p S_{p-1}$. The innovation vector (effective write error) is computed as:
   $$\tilde{v}_p = v_p^{\text{val}} - k_p S_{p-1} \in \mathbb{R}^{1 \times d}$$

4. **Language-Dominant Path ($F_l$ Discovery)**:
   In the original BCAM, the language-dominant feature $F_l$ maps token significance back across the pixel grid. In Delta-BCAM, the innovation vector $\tilde{v}_p$ directly measures the anatomical discrepancy between the observed visual patch and the semantic memory:
   $$F_l[p] = \tilde{v}_p \in \mathbb{R}^{1 \times d}$$
   When a pixel matches an expected lesion specified in $S_0$, $\tilde{v}_p$ exhibits high selective activation. For background anatomical structures where visual features align with neutral expectations, $\tilde{v}_p \to 0$.

5. **State Recurrence Update**:
   Using dynamic channel-wise decay $\alpha_p = \sigma(W_\alpha v_p)$ and write gate $w_p = \sigma(W_w v_p)$, the associative state is updated for the subsequent spatial location:
   $$S_p = \left(\mathbf{I} - k_p^\top k_p\right) \text{Diag}(\alpha_p) S_{p-1} + k_p^\top (w_p \odot \tilde{v}_p) \in \mathbb{R}^{d \times d}$$

Upon completing the traversal over all $P$ positions, the resulting sequence tensors are reshaped back to the 2D spatial domain:

$$F_v \in \mathbb{R}^{H_\omega \times W_\omega \times d}, \quad F_l \in \mathbb{R}^{H_\omega \times W_\omega \times d}$$

---

## 4. Downstream Integration with K-Gate and Decoder

The outputs $F_v$ and $F_l$ produced by Delta-BCAM maintain the exact dimensional interface required by the subsequent K-Gate module of C2Seg. The integration proceeds through three sequential operations:

1. **Input Handoff**: The clinical text representation $L$ is compiled into the initial state $S_0$, and the visual tensor $V$ undergoes serialized traversal via the Delta-BCAM engine. The module yields two complementary spatial tensors: vision-dominant readout $F_v \in \mathbb{R}^{H_\omega \times W_\omega \times d}$ and language-dominant innovation $F_l \in \mathbb{R}^{H_\omega \times W_\omega \times d}$.
2. **Nonlinear Attentive Filtering (K-Gate)**: The features are routed through independent dual KAN gating heads to suppress background noise and irrelevant anatomical regions:
   $$g_v = \tanh\left(\text{KAN}_v^{(2)}\left(\text{ReLU}\left(\text{KAN}_v^{(1)}(F_v)\right)\right)\right) \in \mathbb{R}^{H_\omega \times W_\omega \times d}$$
   $$g_l = \tanh\left(\text{KAN}_l^{(2)}\left(\text{ReLU}\left(\text{KAN}_l^{(1)}(F_l)\right)\right)\right) \in \mathbb{R}^{H_\omega \times W_\omega \times d}$$
3. **Channel Fusion and Multiscale Merging**: The gated features are modulated via element-wise multiplication, concatenated along the channel axis, and projected back to the decoder channel dimension $C_\omega$:
   $$F_{\text{gated}} = \left(F_v \odot g_v\right) \mathbin{\Vert} \left(F_l \odot g_l\right) \in \mathbb{R}^{H_\omega \times W_\omega \times 2d}$$
   $$F_{\text{fused}} = \text{Conv}_{1 \times 1}(F_{\text{gated}}) \in \mathbb{R}^{H_\omega \times W_\omega \times C_\omega}$$
   The resulting tensor $F_{\text{fused}}$ is directly summed with the upsampled representation from decoder stage $\omega + 1$ before advancing to subsequent convolution blocks.

---

## 5. Comparative Complexity and Analytical Assessment

The structural differences between the baseline BCAM and the proposed Delta-BCAM module are summarized below:

| Dimension / Metric | Baseline BCAM (Softmax Attention) | Proposed Delta-BCAM (Linear Associative State) |
| :--- | :--- | :--- |
| **Computational Complexity** | $\mathcal{O}(P \cdot N \cdot C)$ | $\mathcal{O}(P \cdot d^2 + N \cdot d^2)$ |
| **Peak Activation Memory** | $\mathcal{O}(P \cdot N)$ (Full affinity matrix $A$) | $\mathcal{O}(d^2)$ (Fixed associative state $S$) |
| **Scaling to 3D Scans ($P \to 10^7$)** | Prohibitive ($\mathcal{O}(P \cdot N)$ memory explosion) | Strict Linear $\mathcal{O}(P)$ with $\mathcal{O}(1)$ working state |
| **Linguistic Signal Retention** | Vulnerable to diffusion via row-wise softmax averaging | Preserved via orthogonal projection $(\mathbf{I} - k^\top k)$ |
| **Channel Energy Imbalance** | Unmitigated (prone to gradient dominance) | Dynamically compensated by Matched-Filter gate $\gamma_p$ |
| **Parameter Efficiency** | Symmetric projection layers | Sensitivity-allocated gating (low-rank erase, high-rank write) |

By replacing the dense affinity matrix with an online error-correcting memory recurrence, Delta-BCAM preserves fine-grained linguistic guidance across arbitrarily large spatial dimensions, achieving high fidelity in anatomical localization while maintaining a constant memory footprint.
