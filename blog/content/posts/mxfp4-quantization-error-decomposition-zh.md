---
title: "分解 MXFP4 量化误差：为大模型强化学习对症下药"
date: 2026-05-27T10:30:00-05:00
draft: false
tags: ["Quantization", "Reinforcement Learning", "Large Language Models", "MXFP4", "Low-precision Training", "中文"]
categories: ["Research"]
description: "MXFP4 的量化误差不是单一噪声，而是由三种性质完全不同的成分叠加而成；每一种主导一种 RL 训练失效模式，因此需要对症下药的修正策略。"
summary: "MXFP4 量化误差的精确三分解（缩放偏置 / 死区截断 / 网格噪声）以及对症下药的三件套修正（MBS / OF / AQN）。"
math: true
---

## TL;DR

MXFP4 在 NVIDIA Blackwell / AMD MI350 等加速器上能给大模型训练带来 **4× 吞吐 + 4× 显存** 的红利，但**直接拿来做 RL 后训练，精度会塌方**。现有工作把量化误差当成单一噪声来打补丁，这就解释不了一个最关键的现象：*同样的 MXFP4，为什么对 dense 和 MoE 的伤害方式完全不同？*

我们证明：MXFP4 误差**精确等于**三种结构完全不交的成分之和：

- **缩放偏置（Scale bias）**—— E8M0 缩放只能取 $2^n$ → 损害**梯度**
- **死区截断（Deadzone truncation）**—— 小值被映射为零 → 损害 **rollout**
- **网格噪声（Grid noise）**—— E2M1 网格的舍入 → 抬高**策略熵**

每个修正只针对一种机制：**MBS**（宏块缩放）消除缩放偏置；**OF**（离群值回退，残差权重 $\alpha=0.5$）找回死区；**AQN**（自适应量化噪声）控制残余网格噪声。在 Qwen2.5-3B 与 Qwen3-30B-A3B-Base 上验证：

- **Dense**：恢复 BF16 在 $-0.7$ pp 以内（81.3% vs. 82.0%）
- **MoE**：**AQN+MBS+OF（$\alpha=0.5$）比 BF16 还高 $+1.0$ pp**（92.49% vs. 91.51%）

---

## 痛点：把误差当成"一团噪声"，就解释不了 dense / MoE 的差异

MXFP4（OCP MX 标准）的存储格式是每 32 个元素一个 block：

1. 共享的 **E8M0** 缩放——0 mantissa bit，只能取 $2^n$
2. **E2M1** 元素值——网格 $\mathcal{G} = \\{0, \pm 0.5, \pm 1, \pm 1.5, \pm 2, \pm 3, \pm 4, \pm 6\\}$

把 BF16 直接换成它做 RL 后训练，精度立刻崩：

- Qwen2.5-3B（dense）：GSM8K 上掉 **−21.9 pp**
- Qwen3-30B-A3B-Base（MoE）：掉 **−2.7 pp**

传统的 PTQ / QAT 工作（GPTQ、AWQ、QuIP、QeRL……）都把量化误差 $e = Q(x) - x$ 视为单一噪声、然后打补丁。这就回避了一个结构性问题：*为什么 dense 掉 21.9 pp，MoE 只掉 2.7 pp？* 两种架构里**主导**的失效机制必然不同。

---

## 核心洞察：把误差精确拆成三个互不交叠的分量

引入**理想缩放量化器** $Q^*$ ：它使用未量化的理想缩放 $s_b^* = \max_i |x_{b,i}| / q_{\max}$ ，而不是 E8M0 的天花板舍入 $s_b$ 。这样总误差可以**精确分解**：

$$e_{b,i} = \underbrace{Q(x_{b,i}) - Q^\*(x_{b,i})}\_{e^{\mathrm{scale}}\_{b,i} \text{ 缩放偏置}} + \underbrace{[Q^\*(x_{b,i}) - x_{b,i}] \cdot \mathbb{1}\_{\mathcal{D}\_b}(i)}\_{e^{\mathrm{DZ}}\_{b,i} \text{ 死区截断}} + \underbrace{[Q^\*(x_{b,i}) - x_{b,i}] \cdot \mathbb{1}\_{\mathcal{D}\_b^c}(i)}\_{e^{\mathrm{grid}}\_{b,i} \text{ 网格噪声}}$$


其中死区 $\mathcal{D}\_b$ 是被映射到 0 的元素集合，即满足 $|x_{b,i}| < m_b / 24$ 的部分。

这个分解有两个**形式性质**支撑了后面所有结论。

### 性质 1：精确正交

**引理（点态严格，无任何分布假设）：**

$$\langle \mathbf{e}^{\mathrm{DZ}}, \mathbf{e}^{\mathrm{scale}} \rangle = \langle \mathbf{e}^{\mathrm{DZ}}, \mathbf{e}^{\mathrm{grid}} \rangle = 0$$

*证明思路：* 在死区上 $|x/s_b^\*| < q_{\min}/2$ ，所以 $Q^\*(x) = 0$ 。天花板舍入保证 $s_b \geq s_b^\*$ ，因此 $|x/s_b| \leq |x/s_b^\*| < q_{\min}/2$ ，于是 $Q(x) = 0$ 。所以 $e^{\mathrm{scale}} = Q - Q^\* = 0$ 在整个死区上恒为零。$\square$

由此 MSE 恒等式只剩**一个交叉项**：

$$
\\|\mathbf{e}\\|^2 = \\|\mathbf{e}^{\mathrm{scale}}\\|^2 + \\|\mathbf{e}^{\mathrm{DZ}}\\|^2 + \\|\mathbf{e}^{\mathrm{grid}}\\|^2 + 2 \langle \mathbf{e}^{\mathrm{scale}}, \mathbf{e}^{\mathrm{grid}} \rangle
$$

实测上这个交叉项 $\cos(\mathbf{e}^{\mathrm{scale}}, \mathbf{e}^{\mathrm{grid}}) \approx -0.66$ ——在 18,876 张权重张量上跨两种模型规模都成立。这是 ceiling 舍入 $s_b \geq s_b^\*$ 强制造成的结构性反相关。

{{< figure src="/blog/images/mxfp4-decomposition/figA1_orthogonality.png" alt="两两余弦相似度" caption="图 1：Qwen3-30B-A3B-Base 18,624 张权重张量上的两两误差分量余弦相似度分布。死区与 scale / grid 严格正交（点质量集中在 0）；scale 与 grid 反相关于 cos ≈ −0.66，方差极小。" >}}

### 性质 2：网格噪声对缩放精度免疫 ⇒ 不可约下界

网格误差 $e^{\mathrm{grid}}$ 只依赖权重和**理想缩放** $s_b^*$ ，与实际的 E8M0 $s_b$ **无关**。当我们把缩放精度从 E8M0 提到 E8M$k$ 时：

$$
\\|\mathbf{e}^{\mathrm{scale}}\\|^2 \to 0, \quad
\langle \mathbf{e}^{\mathrm{scale}}, \mathbf{e}^{\mathrm{grid}} \rangle \to 0, \quad
\\|\mathbf{e}\\|^2 \to \underbrace{\\|\mathbf{e}^{\mathrm{grid}}\\|^2 + \\|\mathbf{e}^{\mathrm{DZ}}\\|^2}\_{\text{不可约下界}}
$$

总误差收敛到一个**完全由 E2M1 网格本身决定**的下界。要突破它，只能换掉 E2M1（比如 NVFP4 更宽的格式），或者直接把死区元素挽救回来（这就是 OF 干的事）。

{{< figure src="/blog/images/mxfp4-decomposition/fig5_scale_sweep.png" alt="缩放精度扫描" caption="图 2：随着 block-scale mantissa 精度提升，scale 分量逐渐归零，但 grid 分量始终不动。总 MSE 收敛到不可约下界 ‖e_grid‖² + ‖e_DZ‖²。" >}}

跨模型的实证比例惊人地一致：

| 模型 | $\\|\mathbf{e}^{\mathrm{scale}}\\|^2/\\|\mathbf{e}\\|^2$ | $\\|\mathbf{e}^{\mathrm{DZ}}\\|^2/\\|\mathbf{e}\\|^2$ | $\\|\mathbf{e}^{\mathrm{grid}}\\|^2/\\|\mathbf{e}\\|^2$ | $\cos$ |
|---|---:|---:|---:|---:|
| Qwen2.5-3B | 1.725 | 0.026 | 0.703 | −0.657 |
| Qwen3-30B-A3B-Base | 1.726 | 0.022 | 0.712 | −0.658 |

这组数字是 **MXFP4 格式本身**的统计指纹，跟具体模型无关。

{{< figure src="/blog/images/mxfp4-decomposition/fig2_error_decomp.png" alt="逐 layer-type 的误差分解" caption="图 3：Qwen3-30B-A3B-Base 各 layer-type（attention / MLP / expert / shared）上的三分量比例几乎一致——三分解是普适的。" >}}

---

## 每个分量主导 RL 训练的一种失效模式

三分解之所以重要，是因为每个分量通过**完全不同的机制**去破坏 RL 训练。

### 缩放偏置 → 梯度精度（反传专属损伤）

E8M0 ceiling 舍入给出每块的缩放比 $\gamma_b = s_b / s_b^* = 2^{\delta_b}$ ，其中 $\delta_b \sim \mathrm{Uniform}[0, 1)$ ，对应每层约 44% 的系统性偏置。在**前向**，LayerNorm 每层都把激活重新归一化，阻止偏置累积。但在**反传**，STE 链式法则把 $L$ 个缩放因子串起来相乘，没有任何归一化：

$$
\log \frac{\\|\hat{\nabla}\\|}{\\|\nabla\_{\mathrm{true}}\\|} \approx \sum_{l=1}^L \delta_{b(l)}, \qquad \delta_b \sim \mathrm{Uniform}[0, 1)
$$

48 层时，中心化和的标准差是 $\sqrt{L/12} = 2.0$ 。一个标准差内，梯度幅值比就横跨 $[0.18\times, 5.6\times]$ 。实测 $\sigma_{\mathrm{emp}} = 1.97$ ，与理论紧密一致。

{{< figure src="/blog/images/mxfp4-decomposition/figA4_grad_fluctuation.png" alt="缩放偏置的反传累积" caption="图 4：(a) 每层 ceiling 残差 δ_b 近似 Uniform(0, 1)，均值 0.546。(b) L=48 层的累积偏置实测 σ=1.97，理论 √(48/12)=2.0——验证了反传中的指数级放大。" >}}

### 死区截断 → Rollout 质量（前向专属损伤）

死区是**信息丢失**：约 9% 的权重被映射到零。前向有效秩下降，rollout 输出变"钝"。但 STE 反传**看不见**死区——它当成全精度值一样把梯度直通过去。所以死区的破坏在 rollout 里看得清清楚楚，却在梯度里几乎隐形。

这种"前向坏、反向看不见"的不对称，正是死区主要是 **rollout 质量问题**而非梯度问题的原因。

### 网格噪声 → 策略熵（等价温度缩放）

网格噪声近似零均值高斯，作用在 logits 上：$\boldsymbol{\eta} \sim \mathcal{N}(0, \sigma_\eta^2 \mathbf{I})$ 。通过经典的 probit–logit 高斯边缘化恒等式（probit 与高斯卷积仍是 probit，方差相加），可以把加噪 softmax 匹配到一个确定性的温度缩放 softmax：

$$
T_{\mathrm{eff}} \approx \sqrt{1 + \dfrac{2 \sigma_\eta^2}{\mathrm{Var}(\Delta \ell)}} \\; > \\; 1
$$

平方根里的 2 来自 $\mathrm{Var}(\eta_a - \eta_b) = 2\sigma_\eta^2$ ——即两个独立高斯之差的方差。噪声强度**整训练恒定**，没有任何退火机制——探索被均匀地"加宽"。如果不主动管理，会导致**策略熵过早坍塌**。

---

## 对症下药的三件套修正

### MBS —— 宏块缩放（Macro Block Scaling）

E8M0 没有 mantissa bit，浪费了半个量级的动态范围。**MBS** 在更粗的宏块（$B_M = 128$，相当于 4 个 MXFP4 block）上加一个 8-bit mantissa 修正：

$$
s^{\mathrm{MBS}} = 2^{e_M} \cdot (1 + m_{\mathrm{MBS}}), \qquad m_{\mathrm{MBS}} \in [0, 1)
$$

训练里以 prescale–quantize–postscale 的形式包裹标准量化器：

$$
\hat{x}\_i = \frac{1}{1 + m_{\mathrm{MBS}}} \cdot Q\\!\bigl((1 + m_{\mathrm{MBS}}) \cdot x_i\bigr)
$$

开销：**每 128 个元素只多 1 byte ≈ <0.1 bit/element**，prescale 和 postscale 都能融进 GEMM 的 epilogue，几乎零额外算力。效果：$\mathrm{Var}(\gamma)$ 缩小约 $(256)^{-2}$ 倍；scale 分量归零，与 grid 的交叉项也归零；总 MSE 收敛到不可约下界。

### OF —— 离群值回退（Outlier Fallback，带残差权重 $\alpha$）

死区是**信息丢失**，缩放调不回来。**OF** 用两步残差量化把它救回来：

$$
\hat{\mathbf{x}}_1 = Q(\mathbf{x}), \quad
\hat{\mathbf{x}}_2 = Q(\mathbf{x} - \hat{\mathbf{x}}_1), \quad
\hat{\mathbf{x}}\_{\mathrm{OF}} = \hat{\mathbf{x}}_1 + \alpha \\, \hat{\mathbf{x}}_2
$$

第一遍正常量化：outlier 设定 block scale，小值进死区。第二遍对残差量化：残差量级被压缩到很小，原死区元素现在可以落到 E2M1 网格上了。

我们引入**残差权重 $\alpha$** 。直觉上 $\alpha = 1$ 是把残差完整加回，但 $\hat{\mathbf{x}}_2$ 本身也是 MXFP4 量化的近似，它自己也有 grid + deadzone 误差。以**半幅强度 $\alpha = 0.5$** 加回反而比 $\alpha = 1$ 在 GSM8K 上**高约 1 pp**：

| $\alpha$ | GSM8K final | GSM8K peak |
|---|---:|---:|
| **0.5**（默认） | **91.58%** | **92.49%** |
| 1.0 | 90.45% | 90.45% |

更进一步：$\alpha = 0.5$ 还**最小化 rollout 与 training 之间的数值漂移**。在一个受控的单步隔离实验里（仅启用 OF，关闭 MBS/AQN，学习率 0，固定提示，让 $\alpha$ 是唯一变化的量），rollout 与 training 的散度对 $\alpha$ 呈 **U 形，在 $0.5$ 处取得清晰的极小值**——两端（$\alpha=0$ 与 $\alpha=1$）的漂移都是 $\alpha=0.5$ 的约 $1.5$ 到 $2$ 倍。半幅残差最能让 rollout 的伪量化网格与 training 的网格对齐；由于 RL 后训练对漂移高度敏感，让漂移最小的 $\alpha$ 同时也是让下游精度最大的 $\alpha$。

### AQN —— 自适应量化噪声（Adaptive Quantization Noise）

MBS + OF 之后，残余误差被 grid 噪声主导——**静态、恒温**，没有退火机制。**AQN** 在每次 rollout 前给权重注入受控的高斯噪声，按指数衰减从 $\sigma_{\mathrm{start}} = 1\%$ 降到 $\sigma_{\mathrm{end}} = 0.1\%$ 。这就把网格噪声的"恒定温度"改造成了**可退火的探索信号**。

一个微妙但重要的点：**AQN 必须在 MBS 之后才有效**。直接在带偏置的 MXFP4 logits 上注入 AQN 会发散——系统偏置会污染本应零均值的噪声。这也是我们 W4A4 全参数训练设定与 QeRL 的 W4A16 + LoRA 设定的关键区别（QeRL 因为权重精度更高，不需要 MBS 兜底）。

---

## 实验：这套配方真的成立吗？

我们在 GSM8K（可验证奖励的数学推理）上评估，算法是 GRPO + Truncated Importance Sampling。模型：

- **Qwen2.5-3B dense**（36 层，FSDP2）
- **Qwen3-30B-A3B-Base MoE**（48 层，3B 激活，Megatron）

所有实验都是 W4A4 QDQ 仿真——与原生 MXFP4 数值等价（同样的 E2M1 网格，同样的 E8M0 block scale，FP32 累加）。

### MoE 结果（Qwen3-30B-A3B-Base，BF16 = 91.51%）

| 配置 | AQN | MBS | OF | GSM8K (%) | Gap |
|---|:-:|:-:|:-:|---:|---:|
| MXFP4 baseline | | | | 88.8 | −2.7 |
| +MBS | | ✓ | | 90.1 | −1.4 |
| +OF | | | ✓ | 90.3 | −1.2 |
| +AQN(1%) | ✓ | | | 89.2 | −2.3 |
| +AQN+MBS | ✓ | ✓ | | 90.5 | −1.0 |
| +MBS+OF | | ✓ | ✓ | 91.1 | −0.4 |
| **+AQN+MBS+OF ($\alpha=0.5$)** | ✓ | ✓ | ✓ | **92.49** | **+1.0** |

**AQN+MBS+OF（$\alpha = 0.5$）比 BF16 还高 +1.0 pp**——是 MoE 上新的最优配方。

### Dense 结果（Qwen2.5-3B，BF16 = 82.0%）

| 配置 | MBS | AQN | OF | GSM8K (%) | Gap |
|---|:-:|:-:|:-:|---:|---:|
| MXFP4 baseline | | | | 60.1 | −21.9 |
| +MBS | ✓ | | | 75.2 | −6.8 |
| +OF | | | ✓ | 77.6 | −4.4 |
| +MBS+OF | ✓ | | ✓ | 80.8 | −1.2 |
| **+MBS+AQN(1%)+OF** | ✓ | ✓ | ✓ | **81.3** | **−0.7** |

朴素 MXFP4 在 dense 上掉 21.9 pp；MBS+AQN+OF 恢复到 0.7 pp 以内。值得注意的是：**OF 单独在 dense 上贡献 +17.5 pp**，但在 MoE 上只贡献 +1.5 pp。

{{< figure src="/blog/images/mxfp4-decomposition/fig4_ablation.png" alt="消融实验" caption="图 5：MoE（左）与 dense（右）的消融。MoE 上三种修正近似可加；dense 上 OF 单独占主导——因为没有 expert routing 提供的冗余去掩盖死区。" >}}

### Dense vs. MoE 的"病理二象性"是三分解直接预测出来的

为什么 OF 在 dense 上贡献 +17.5 pp，在 MoE 上只贡献 +1.5 pp？因为 MoE 的 expert routing 本身就是一种**天然的纠错码**：即使一个专家的小权重被剪掉，其他专家会代偿。Dense 模型没有这种冗余，死区直接切断信息流。

这正是三分解的预测——**死区是前向问题**，而前向冗余度调节它的严重程度。如果不做这种分解，这种不对称看上去就是凭空出现的现象。

### 训练动力学：AQN 防止策略熵过早坍塌

{{< figure src="/blog/images/mxfp4-decomposition/fig3_training_dynamics.png" alt="训练动力学" caption="图 6：MoE GSM8K 训练动态。Baseline 策略熵在 50 步内从 1.71 暴跌到 0.35（过早收敛）。AQN+MBS 维持熵在 0.61，梯度范数 0.24 vs. baseline 0.16——三分解理论中 grid 噪声 ⇒ 等价温度缩放的预测被实验证实。" >}}

### 适用范围：短回复里"被平均掉"的下界在长回复里会"累积"

§4.3 中的"网格噪声在整段回复上被平均掉"对 GSM8K 这种短到中等长度的回复（约 280 token / 1024-token 上限）成立，但随着回复长度增加而逼近极限。把配方家族固定，*每步*的 rollout–training Pearson 相关性在短任务（GSM8K）和长任务（DAPO-MATH，约 795 token）上几乎完全一致；但端到端表现却分道扬镳——在多千 token 长 CoT 上，同一配方明显欠恢复。机制是**确定性贪婪解码下的自回归轨迹分叉**：单个早期 arg-max 翻转（来自量化噪声）会顺着自回归链放大为完全不同的解题路径。长 CoT 上的保真我们当作本配方的范围外未来工作（很可能需要更高 activation 精度，例如 W4A8）。

---

## 总结

1. **MXFP4 量化误差不是单一噪声项。** 它精确等于缩放偏置、死区截断、网格噪声三个**结构不交**的分量之和，背靠两个形式性质（正交性 + 网格对缩放免疫），跨模型规模比例几乎不变。
2. **每个分量打击 RL 训练的不同部位。** 缩放偏置在**反传中指数累积**但前向不可见；死区正相反（前向致命，反传隐形）；网格噪声以等价温度的方式均匀抬高策略熵。
3. **修正必须机制定向，而不是症状定向。** MBS 消除缩放偏置；OF 救回死区（$\alpha = 0.5$ 比朴素的 $\alpha = 1$ 高约 1 pp）；AQN 控制残余的网格噪声。完整配方在 MoE 上**超过 BF16**，在 dense 上恢复到 0.7 pp 以内。
4. **Dense / MoE 的差异不再神秘。** 死区是前向问题，MoE 的 expert routing 提供了前向冗余把它掩盖；dense 没有这种冗余，所以 OF 是 dense 的"救命药"，MBS + AQN 则是 MoE 的主力。

更广的方法论：**当极低精度量化让精度塌方时，先做误差分解，再打补丁。** 把 MXFP4 误差当成单一噪声的视角，恰好掩盖了指向可工作配方的那部分结构。

---

## 引用 & 资源

论文BibTeX：

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

*欢迎讨论，或想在你自己的 MXFP4 训练栈里试这套配方，欢迎随时联系。*
