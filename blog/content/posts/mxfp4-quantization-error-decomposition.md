---
title: "Decomposing MXFP4 quantization error for LLM RL: reducible bias, recoverable deadzone, and an irreducible floor"
date: 2026-05-27T10:00:00-05:00
draft: false
tags: ["Quantization", "Reinforcement Learning", "Large Language Models", "MXFP4", "Low-precision Training"]
categories: ["Research"]
description: "MXFP4 quantization error is not one noise term but three with disjoint structure. Each one breaks a different part of RL training, so corrections must be matched to the mechanism."
summary: "An exact three-way decomposition of MXFP4 quantization error into scale bias, deadzone truncation, and grid noise — and the mechanism-targeted corrections (MBS, OF, AQN) that follow."
math: true
---

## TL;DR

MXFP4 promises **4× throughput and 4× memory** for LLM training on Blackwell/MI350-class accelerators, but naively dropping it into RL post-training **destroys accuracy**. Existing work treats the quantization error as one monolithic noise term, which makes it impossible to explain *why* the same MXFP4 hurts dense and MoE models in completely different ways.

We prove that the MXFP4 error is **exactly the sum of three components with disjoint structure**:

- **Scale bias** from the E8M0 power-of-two scale → corrupts the **gradient**
- **Deadzone truncation** from zeroing small values → corrupts the **rollout**
- **Grid noise** from rounding to the E2M1 grid → raises **policy entropy**

Each correction targets one mechanism: **MBS** (macro-block scaling) kills the scale bias, **OF** (outlier fallback with blend $\alpha=0.5$) recovers the deadzone, and **AQN** controls the residual grid noise. Validated on Qwen2.5-3B and Qwen3-30B-A3B-Base:

- **Dense**: recovers BF16 within $-0.7$ pp (81.3% vs. 82.0%)
- **MoE**: **MBS + OF exceeds BF16 by $+1.0$ pp** (92.49% vs. 91.51%)

---

## The Problem: One Monolithic Noise Term Hides Three Mechanisms

The MXFP4 format (OCP MX standard) stores each block of 32 elements as:

1. A shared **E8M0** scale — 0 mantissa bits, only powers of two
2. **E2M1** per-element values on the grid $\mathcal{G} = \\{0, \pm 0.5, \pm 1, \pm 1.5, \pm 2, \pm 3, \pm 4, \pm 6\\}$

Replace BF16 with this in RL post-training and accuracy collapses:

- Qwen2.5-3B (dense): **−21.9 pp** on GSM8K
- Qwen3-30B-A3B-Base (MoE): **−2.7 pp**

The standard PTQ/QAT literature (GPTQ, AWQ, QuIP, QeRL, …) treats the quantization error $e = Q(x) - x$ as a single noise term and patches the symptoms. That hides the structural question: *why* does dense lose 21.9 pp but MoE only 2.7 pp? Different mechanisms must be dominant in each architecture.

---

## The Key Insight: Three Disjoint Error Components

Introduce an **ideal-scale quantizer** $Q^* $ that keeps the unquantized scale $s_b^* = \max_i |x_{b,i}| / q_{\max}$ instead of E8M0's ceiling-rounded $s_b$. Then the total error decomposes exactly:

$$e_{b,i} = \underbrace{Q(x_{b,i}) - Q^\*(x_{b,i})} \_{e^{\mathrm{scale}}\_{b,i} \text{ (scale bias)}} + \underbrace{[Q^\*(x_{b,i}) - x_{b,i}] \cdot \mathbb{1}\_{\mathcal{D}\_b}(i)}\_{e^{\mathrm{DZ}}\_{b,i} \text{ (deadzone)}} + \underbrace{[Q^\*(x_{b,i}) - x_{b,i}] \cdot \mathbb{1}\_{\mathcal{D}\_b^c}(i)}\_{e^{\mathrm{grid}}\_{b,i} \text{ (grid noise)}}$$

where $\mathcal{D}\_b$  is the deadzone (elements with $|x_{b,i}| < m_b / 24$, mapped to 0).

This decomposition has two formal properties that drive everything that follows.

### Property 1: Exact Orthogonality

**Lemma (pointwise, no distributional assumptions):**

$$\langle \mathbf{e}^{\mathrm{DZ}}, \mathbf{e}^{\mathrm{scale}} \rangle = \langle \mathbf{e}^{\mathrm{DZ}}, \mathbf{e}^{\mathrm{grid}} \rangle = 0$$

*Proof sketch:* On the deadzone, $|x/s_b^\*| < q_{\min}/2$, so $Q^\*(x) = 0$. Ceiling rounding guarantees $s_b \geq s_b^\*$, so $|x/s_b| \leq |x/s_b^\*| < q_{\min}/2$ and $Q(x) = 0$ as well. Hence $e^{\mathrm{scale}} = Q - Q^\* = 0$ identically on the deadzone. $\square$

The MSE identity then has just **one surviving cross term**:

$$
\\|\mathbf{e}\\|^2 = \\|\mathbf{e}^{\mathrm{scale}}\\|^2 + \\|\mathbf{e}^{\mathrm{DZ}}\\|^2 + \\|\mathbf{e}^{\mathrm{grid}}\\|^2 + 2 \langle \mathbf{e}^{\mathrm{scale}}, \mathbf{e}^{\mathrm{grid}} \rangle
$$

Empirically that cross term has $\cos(\mathbf{e}^{\mathrm{scale}}, \mathbf{e}^{\mathrm{grid}}) \approx -0.66$ across 18,876 weight tensors in two model scales — a structural anti-correlation from the ceiling property $s_b \geq s_b^*$.

{{< figure src="/blog/images/mxfp4-decomposition/figA1_orthogonality.png" alt="Pairwise cosine similarities" caption="Figure 1: Pairwise error-component cosine similarities across 18,624 Qwen3-30B-A3B-Base weight tensors. DZ is exactly orthogonal to both scale and grid (point masses at 0); scale and grid are anti-correlated with cos ≈ −0.66 and minimal variance." >}}

### Property 2: Grid Noise Is Scale-Invariant ⇒ Irreducible Floor

The grid error $e^{\mathrm{grid}}$ depends on the weights and the *ideal* scale $s_b^*$, not on the actual E8M0 scale $s_b$. As you improve scale precision (E8M0 → E8M$k$):

$$
\\|\mathbf{e}^{\mathrm{scale}}\\|^2 \to 0, \quad
\langle \mathbf{e}^{\mathrm{scale}}, \mathbf{e}^{\mathrm{grid}} \rangle \to 0, \quad
\\|\mathbf{e}\\|^2 \to \underbrace{\\|\mathbf{e}^{\mathrm{grid}}\\|^2 + \\|\mathbf{e}^{\mathrm{DZ}}\\|^2}\_{\text{irreducible floor}}
$$

Total error converges to a floor set by the E2M1 grid itself. The only way to go below it is to change the grid (e.g., NVFP4's larger format) or to recover deadzone values directly (outlier fallback).

{{< figure src="/blog/images/mxfp4-decomposition/fig5_scale_sweep.png" alt="Scale precision sweep" caption="Figure 2: As we improve the block-scale mantissa precision, the scale component vanishes but the grid component does not. Total MSE converges to the irreducible floor ‖e_grid‖² + ‖e_DZ‖²." >}}

The ratios are remarkably stable across model scales:

| Model | $\\|\mathbf{e}^{\mathrm{scale}}\\|^2/\\|\mathbf{e}\\|^2$ | $\\|\mathbf{e}^{\mathrm{DZ}}\\|^2/\\|\mathbf{e}\\|^2$ | $\\|\mathbf{e}^{\mathrm{grid}}\\|^2/\\|\mathbf{e}\\|^2$ | $\cos$ |
|---|---:|---:|---:|---:|
| Qwen2.5-3B | 1.725 | 0.026 | 0.703 | −0.657 |
| Qwen3-30B-A3B-Base | 1.726 | 0.022 | 0.712 | −0.658 |

These numbers are properties of **the MXFP4 format itself**, not of any particular model.

{{< figure src="/blog/images/mxfp4-decomposition/fig2_error_decomp.png" alt="Per-layer-type error decomposition" caption="Figure 3: Per-layer-type error decomposition (Qwen3-30B-A3B-Base). The same three-way ratios hold across attention, MLP, expert, and shared layers — the decomposition is universal." >}}

---

## Each Component Dominates One RL Failure Mode

The decomposition matters because each component breaks RL training through a *different* mechanism.

### Scale bias → gradient accuracy (backward-pass-only pathology)

E8M0 ceiling rounding gives a per-block scale ratio $\gamma_b = s_b / s_b^* = 2^{\delta_b}$ with $\delta_b \sim \mathrm{Uniform}[0, 1)$, i.e. a $\sim$44% systematic bias per layer. In the **forward pass**, LayerNorm resets activations every layer, preventing accumulation. But in the **backward pass**, the STE chain rule multiplies $L$ scale factors with no normalization:

$$
\log \frac{\\|\hat{\nabla}\\|}{\\|\nabla\_{\mathrm{true}}\\|} \approx \sum_{l=1}^L \delta_{b(l)}, \qquad \delta_b \sim \mathrm{Uniform}[0, 1)
$$

For 48 layers, the centred sum has std $\sqrt{L/12} = 2.0$. Within one standard deviation the gradient magnitude ratio spans roughly $[0.18\times, 5.6\times]$. We measure $\sigma_{\mathrm{emp}} = 1.97$ on Qwen3-30B-A3B-Base, in tight agreement with theory.

{{< figure src="/blog/images/mxfp4-decomposition/figA4_grad_fluctuation.png" alt="Scale bias accumulation in the backward pass" caption="Figure 4: (a) Per-layer ceiling residual δ_b is approximately Uniform(0, 1) with mean 0.546. (b) Cumulative scale bias across L=48 layers matches the theoretical std of 2.0 with empirical std 1.97 — validating the backward-pass amplification." >}}

### Deadzone → rollout quality (forward-pass-only pathology)

Deadzone is **information loss**: about 9% of weights are mapped to zero. The forward pass loses rank, producing blander rollouts. But the STE backward pass **ignores the deadzone** — it passes gradients through as if the full-precision value were present. So deadzone damage is fully visible in rollouts but largely invisible in gradients.

This asymmetry is what makes deadzone primarily a *rollout quality* problem, not a gradient problem.

### Grid noise → policy entropy (effective temperature)

Grid noise is approximately zero-mean Gaussian on logits, $\boldsymbol{\eta} \sim \mathcal{N}(0, \sigma_\eta^2 \mathbf{I})$. By matching the noisy softmax to a deterministic temperature-scaled softmax via the classical probit–logit identity (convolving a probit with a Gaussian gives another probit with summed variances),

$$
T_{\mathrm{eff}} \approx \sqrt{1 + \dfrac{2 \sigma_\eta^2}{\mathrm{Var}(\Delta \ell)}} \\; > \\; 1
$$

The factor of 2 comes from $\mathrm{Var}(\eta_a - \eta_b) = 2\sigma_\eta^2$ — the variance of a pairwise logit difference. The noise level is **constant throughout training**, so it provides no annealing mechanism — exploration is widened uniformly. This causes premature entropy collapse if not actively managed.

---

## Three Mechanism-Targeted Corrections

### MBS — Macro Block Scaling

E8M0 wastes half a magnitude because it has zero mantissa bits. **MBS** adds an 8-bit mantissa correction at a coarser macro-block granularity ($B_M = 128$, four MXFP4 blocks):

$$
s^{\mathrm{MBS}} = 2^{e_M} \cdot (1 + m_{\mathrm{MBS}}), \qquad m_{\mathrm{MBS}} \in [0, 1)
$$

Implemented in training as a prescale–quantize–postscale wrapper:

$$
\hat{x}\_i = \frac{1}{1 + m_{\mathrm{MBS}}} \cdot Q\\!\bigl((1 + m_{\mathrm{MBS}}) \cdot x_i\bigr)
$$

Cost: **1 byte / 128 elements ≈ <0.1 bit/element**, with prescale/postscale folded into the GEMM epilogue at near-zero compute cost. Effect: $\mathrm{Var}(\gamma)$ shrinks by $\sim (256)^{-2}$, the scale component vanishes, and the cross term with grid noise vanishes too — total MSE converges to the irreducible floor.

### OF — Outlier Fallback (with residual blend α)

Deadzone is information loss; no rescaling recovers it. **OF** uses a two-pass residual quantization:

$$
\hat{\mathbf{x}}_1 = Q(\mathbf{x}), \quad
\hat{\mathbf{x}}_2 = Q(\mathbf{x} - \hat{\mathbf{x}}_1), \quad
\hat{\mathbf{x}}\_{\mathrm{OF}} = \hat{\mathbf{x}}_1 + \alpha \\, \hat{\mathbf{x}}_2
$$

Pass 1 quantizes outliers accurately (they set the block scale) and sends small values to the deadzone. Pass 2 has small dynamic range, so previously dead values are now representable on the E2M1 grid.

We introduce the **residual-blend coefficient $\alpha$**. Setting $\alpha = 1$ adds the full residual back, but $\hat{\mathbf{x}}_2$ is itself an MXFP4-quantized approximation and inherits its own grid+deadzone error. Adding it at **half strength ($\alpha = 0.5$) outperforms $\alpha = 1$ on GSM8K by ~1 pp**:

| $\alpha$ | GSM8K final | GSM8K peak |
|---|---:|---:|
| **0.5** (default) | **91.58%** | **92.49%** |
| 1.0 | 90.45% | 90.45% |

### AQN — Adaptive Quantization Noise

After MBS and OF, the residual error is dominated by grid noise — static and unannealed. **AQN** injects controlled Gaussian noise on weights before each rollout with an exponential decay schedule from $\sigma_{\mathrm{start}} = 1\%$ to $\sigma_{\mathrm{end}} = 0.1\%$. This converts the constant grid temperature into a *annealable* exploration signal.

A subtle but important point: AQN only works **after MBS** removes the first-moment scale bias. Injecting AQN on top of biased MXFP4 logits diverges — the systematic bias contaminates the noise. This is the difference between our W4A4 full-parameter setting and QeRL's W4A16 + LoRA setting, where MBS isn't necessary.

---

## Experiments: Does the Recipe Actually Hold Together?

We evaluate on GSM8K with verifiable rewards, using GRPO + Truncated Importance Sampling. Models:

- **Qwen2.5-3B dense** (36 layers, 2× H100, FSDP2)
- **Qwen3-30B-A3B-Base MoE** (48 layers, 3B active, 8× H100, Megatron)

All experiments are W4A4 QDQ emulation — numerically faithful to native MXFP4 (same E2M1 grid, same E8M0 block scale, FP32 accumulation).

### MoE results (Qwen3-30B-A3B-Base, BF16 = 91.51%)

| Configuration | AQN | MBS | OF | GSM8K (%) | Gap |
|---|:-:|:-:|:-:|---:|---:|
| MXFP4 baseline | | | | 88.8 | −2.7 |
| +MBS | | ✓ | | 90.1 | −1.4 |
| +OF | | | ✓ | 90.3 | −1.2 |
| +AQN(1%) | ✓ | | | 89.2 | −2.3 |
| +AQN+MBS | ✓ | ✓ | | 90.5 | −1.0 |
| +AQN+MBS+OF | ✓ | ✓ | ✓ | 91.1 | −0.4 |
| **+MBS+OF ($\alpha=0.5$)** | | ✓ | ✓ | **92.49** | **+1.0** |

**MBS + OF (no AQN, $\alpha = 0.5$) exceeds BF16 by +1.0 pp** — the new best recipe.

### Dense results (Qwen2.5-3B, BF16 = 82.0%)

| Configuration | MBS | AQN | OF | GSM8K (%) | Gap |
|---|:-:|:-:|:-:|---:|---:|
| MXFP4 baseline | | | | 60.1 | −21.9 |
| +MBS | ✓ | | | 75.2 | −6.8 |
| +OF | | | ✓ | 77.6 | −4.4 |
| +MBS+OF | ✓ | | ✓ | 80.8 | −1.2 |
| **+MBS+AQN(1%)+OF** | ✓ | ✓ | ✓ | **81.3** | **−0.7** |

Naive MXFP4 loses 21.9 pp; MBS+AQN+OF recovers to within 0.7 pp. Notably, **OF alone provides +17.5 pp on dense** but only +1.5 pp on MoE.

{{< figure src="/blog/images/mxfp4-decomposition/fig4_ablation.png" alt="Ablation results" caption="Figure 5: Ablation across MoE (left) and dense (right). MoE gains are near-additive; on dense, OF alone dominates because there is no expert-routing redundancy to mask the deadzone." >}}

### The dense vs. MoE dichotomy is predicted by the decomposition

Why does OF give $+17.5$ pp on dense but only $+1.5$ pp on MoE? Expert routing in MoE acts as a natural error-correcting code: if one expert's small weights are pruned, other experts compensate. Dense models have no such redundancy, so the deadzone severs the forward path directly.

This is exactly the prediction of the decomposition — deadzone is a *forward-pass* problem, and forward-pass redundancy modulates its severity. Without this view, the asymmetry looks arbitrary.

### Training dynamics: AQN prevents entropy collapse

{{< figure src="/blog/images/mxfp4-decomposition/fig3_training_dynamics.png" alt="Training dynamics" caption="Figure 6: MoE GSM8K training dynamics. Baseline policy entropy collapses from 1.71 to 0.35 by step 50 (premature convergence). AQN+MBS sustains entropy at 0.61, with gradient norm 0.24 vs. baseline 0.16. The grid-noise → temperature widening predicted by theory is confirmed empirically." >}}

---

## Takeaways

1. **MXFP4 quantization error is not one noise term.** It is the exact sum of scale bias, deadzone, and grid noise — three components with disjoint structure, two formal properties (orthogonality + grid invariance), and consistent ratios across model scales.
2. **Each component breaks a different part of RL training.** Scale bias compounds *exponentially in the backward pass* but is forward-invariant; deadzone is the opposite (forward-fatal, gradient-invisible); grid noise raises *effective temperature* uniformly.
3. **Corrections must be mechanism-targeted, not symptom-targeted.** MBS removes scale bias, OF recovers the deadzone (with $\alpha = 0.5$ outperforming the naive $\alpha = 1$ by ~1 pp), AQN controls residual grid noise. The recipe **exceeds BF16 on MoE** and matches it within 0.7 pp on dense.
4. **The dense/MoE dichotomy is no longer mysterious.** Deadzone is a forward-pass problem; MoE expert routing provides forward-pass redundancy that masks it. Dense models cannot — so OF is the load-bearing correction on dense, while MBS+AQN is what carries MoE.

The broader principle: **when accuracy collapses under aggressive quantization, decompose the error before patching symptoms.** A single-noise view of MXFP4 hides exactly the structure that points to a working recipe.

---

## References & Code

If you find our work useful, please kindly cite our work:

```bibtex
@misc{li2026decomposingmxfp4quantizationerror,
      title={Decomposing MXFP4 quantization error for LLM reinforcement learning: reducible bias, recoverable deadzone, and an irreducible floor}, 
      author={Xiaocan Li and Shiliang Wu and Zheng Shen},
      year={2026},
      eprint={2605.20402},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.20402}, 
}
```

*Questions, comments, or want to try the recipe on your own MXFP4 stack? Feel free to reach out.*
