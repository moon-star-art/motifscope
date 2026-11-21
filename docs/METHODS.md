# 方法说明

本文档说明 `motifscope` 背后的方法学，包括数据构造、模型设计、归因算法与评估指标。
阅读论文正文前，建议先看本文。

---

## 1. 为什么"归因是否可信"难以评估

给定模型 $f$ 和输入序列 $x$，归因方法给出每个碱基的重要性分数 $\varphi_i$。
理想情况下 $\varphi_i$ 应满足若干公理（Sundararajan et al., 2017）：

1. **完整性（Completeness）**：$\sum_i \varphi_i = f(x) - f(x')$，其中 $x'$ 是基线。
2. **敏感性（Sensitivity）**：若某个特征变化导致 $f$ 变化，它必须得到非零归因。
3. **实现不变性（Implementation Invariance）**：功能等价的模型应得到相同归因。

真实生物数据的问题在于：**无法验证第 1、2 条**。我们知道某段 DNA 是增强子，
但不知道哪些碱基"真正"驱动了模型的判断。

### 合成数据的解法

用已知 PWM 生成序列并植入，可以得到精确的逐碱基真值掩码 $T \in \{0,1\}^L$。
于是"归因是否指向正确位置"变成了可计算的分类问题。

---

## 2. 数据构造

### 2.1 基序采样

对长度为 $L_m$ 的 PWM，第 $j$ 位碱基按 $P(b \mid j)$ 独立采样：

$$x_j \sim \text{Categorical}\big(P(\cdot \mid j)\big)$$

**为什么采样而不是直接用一致序列**：一致序列是每个位置概率最大的碱基组合，
在生物学上极少出现。若用一致序列训练，模型可能只学会"匹配一致序列"，
在真实的基序变体上失效。

### 2.2 植入策略

- **位置**：在 $[0, L - L_m]$ 上均匀随机
- **方向**：以 0.5 概率取反向互补（双链结合）
- **重叠**：默认允许，模拟真实的基序簇；可关闭以获得无重叠数据

**真值掩码**：$T_i = 1$ 当且仅当位置 $i$ 被任一植入的基序覆盖。

**已知局限**：允许重叠时，后植入的基序会覆盖先植入基序的部分碱基。因此条目中
同时记录 `instance`（原始实例）与 `written`（最终序列中的实际片段），
并给出 `intact` 标志。这保证了记录与序列始终一致。

### 2.3 负样本

生成一定比例（默认 25%）完全不含基序的纯背景序列，使任务成为二分类问题。

---

## 3. 模型

### 3.1 PWM 扫描模型

$$f(x) = \sum_{m=1}^{M} \text{pool}_i \Big( \sum_{j} W_m[b_{i+j}, j] \Big) - \beta$$

其中 `pool` 可选 `max`（每基序取最优窗口）或 `sum`（所有窗口求和）。

**为什么默认 max**：实测 sum 池化的准确率仅 0.488（低于随机）。原因是 200 bp
序列上有 ~190 个窗口，背景噪声的累加完全淹没了单个基序的信号。

### 3.2 k-mer 逻辑回归

把序列表示成所有 k-mer 的指示向量，用 L2 正则逻辑回归分类。

这是**严格线性**模型，因此是检验归因方法收敛性的"标准答案"：
ISM、IG 等近似方法都应收敛到它的精确解。

**精确分解的推导**：

对 one-hot 输入，$f(x) = \sum_w \text{coef}[w, \text{code}_w] + \text{intercept}$。
把每个窗口的系数均摊给它覆盖的 $k$ 个位置：

$$\varphi_j = \sum_{w:\, j \in \text{window } w} \frac{\text{coef}[w, \text{code}_w]}{k}$$

均摊是满足"守恒"与"位置对称"的自然选择（单个 k-mer 系数无法唯一拆到其 k 个碱基上）。
再减去全零基线的同一量，即得 $f(x) - f(0)$。

### 3.3 卷积 + 全局 max 池化

$$\text{score}_m(i) = \sum_{b,o} W[m, b, o] \cdot x[i+o, b] + b_m$$

$$f(x) = \sum_m v_m \cdot \max_i \text{score}_m(i) + c$$

用纯 numpy 实现，Adam 优化器，无需 PyTorch/TensorFlow。

**关键性质**：max 池化使梯度只在 argmax 窗口内非零，因此

$$\varphi_i = x_i \cdot \frac{\partial f}{\partial x_i} = \begin{cases} v_m W[m, b, i - i^*_m] & i \in \text{argmax window of } m \\ 0 & \text{otherwise}\end{cases}$$

恰好等于 $f(x) - f(0)$，即**精确分解**，而非近似。

---

## 4. 归因方法

| 方法 | 原理 | 复杂度 |
|---|---|---|
| `exact` | 模型的解析分解 | $O(1)$ 前向 |
| `grad_input` | $\nabla_x f \odot x$ | $O(1)$ 前向 |
| `ig` | $\int_0^1 \nabla_x f(x' + \alpha(x-x'))\,d\alpha$ | $O(\text{steps})$ 前向 |
| `smoothgrad` | 加噪输入多次求梯度平均 | $O(n)$ 前向 |
| `ism` | 逐位置饱和突变 | $O(L)$ 前向 |

**梯度是显式推导的**，不依赖自动微分框架。例如对 PWM max 模型，
只有最优窗口内的碱基有非零梯度，值为对应 PWM 条目。

---

## 5. 评估指标

### 5.1 回收型指标（直接比较归因与真值）

**位置级 ROC-AUC**：把"归因分数能否区分真值掩码"当作二分类问题。
用 Mann-Whitney U 统计量计算，正确处理并列值。

$$P(\varphi_{\text{in}} > \varphi_{\text{out}})$$

**Top-k% 命中率**：归因最高的 $k$% 位置中，落在真值内的比例。
随机基线约等于真值占比（默认约 8%）。

**峰值命中率**：归因峰值位置是否落在真值区域内。

### 5.2 扰动型指标（观察模型行为）

**删除曲线**：按归因从高到低把位置替换为背景，记录预测概率变化。

**AOPC**（Area Over the Perturbation Curve）：

$$\text{AOPC} = \underbrace{\frac{1 + p_{\text{norm}}(1)}{2}}_{\text{随机删除的面积}} - \int_0^1 p_{\text{norm}}(f)\,df$$

- AOPC > 0：删除高分区使预测下降**快于**随机 → 归因忠实
- AOPC < 0：比随机更差 → 归因方向错误，或模型根本没学到信号

**充分性**：只保留少量高归因位置时已恢复的预测比例。

### 5.3 稳定性指标

**跨方法 Spearman 相关**：两种方法在同一序列上的排序相关性。
**Top-k Jaccard**：两种方法选出的高分区重合度。

---

## 6. 指标的适用边界（重要）

实验中观察到：**扰动型指标在弱模型上会失去区分度**。

当模型尚未学到基序信号时，即使按真值删除位置，预测也不会明显变化。
此时 AOPC 对真值排序与随机排序给出的数值几乎相同，甚至真值更差。

| 训练轮数 | 测试准确率 | AOPC(真值) | AOPC(随机) | 真值更优？ |
|---|---|---|---|---|
| 2 | 0.750 | −0.094 | −0.066 | 否 |
| 10 | 0.750 | −0.120 | −0.102 | 否 |
| 40 | 0.800 | +0.249 | +0.247 | 是 |
| 80 | 0.750 | +0.304 | +0.315 | 否 |

**结论**：
- 回收型指标衡量的是"归因方法本身"，更稳定
- 扰动型指标衡量的是"模型 + 归因的联合行为"，受模型质量影响大
- 因此**必须同时报告两类指标**，不能只用其中一种

---

## 7. 统计显著性

### 7.1 经验零分布的分辨率问题

采样 $n$ 条随机序列得到的经验分布，最小可分辨 p 值约为 $1/(n+1)$。
用 $n = 400$ 时下限是 2.5e-3。

### 7.2 精确 p 值

PWM 得分是各位置独立贡献之和：

$$\text{score} = \sum_{j=0}^{L-1} w[b_j, j], \quad b_j \sim \text{Background}$$

其分布是 $L$ 个 4 点离散分布的**卷积**，可以精确计算。实现上把每个位置的得分
量化到公共网格，用 `np.convolve` 逐列卷积概率质量函数。

实测与经验分布高度一致（以 ETS1 基序为例）：

| 得分 | 精确 p | 经验 p（n=390000） |
|---|---|---|
| 8.434 | 4.88e-4 | 4.74e-4 |
| 3.226 | 1.07e-2 | 1.09e-2 |
| −13.016 | 4.95e-1 | 5.07e-1 |

分辨率下限约 $1/\text{n\_bins} = 2.5\times10^{-4}$（默认 4000 bins）。

### 7.3 多重检验校正

扫描 $M$ 条基序、长度 $L$ 的序列、双链时，共做 $M \times 2 \times (L - L_m + 1)$ 次检验。
未校正时随机序列上也会出现若干"显著"命中。

本工具默认做 Bonferroni 校正：$\alpha_{\text{adj}} = \alpha / n_{\text{tests}}$。

实测效果（200 bp 随机序列，30 个种子）：校正后 **0/30** 条序列出现假阳性。

### 7.4 重叠命中抑制

相邻窗口往往都超过阈值，但它们描述同一个结合位点。本工具按得分做非极大值抑制
（NMS），并额外合并间隔 ≤2 bp 的同基序命中，以处理串联重复。

未做 NMS 时，一条 666 bp 的序列曾报出 286 个冗余命中；处理后降到 2 个真实位点。

---

## 参考文献

1. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *ICML*.
2. Smilkov, D., et al. (2017). SmoothGrad: removing noise by adding noise. *arXiv:1706.03825*.
3. Shrikumar, A., Greenside, P., & Kundaje, A. (2017). Learning important features through propagating activation differences. *ICML*.
4. Alipanahi, B., et al. (2015). Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. *Nature Biotechnology*.
5. Zhou, J., & Troyanskaya, O. G. (2015). Predicting effects of noncoding variants with deep learning–based sequence model. *Nature Methods*.
6. Avsec, Ž., et al. (2021). Effective gene expression prediction from sequence by integrating long-range interactions. *Nature Methods*.
7. Castro, M. A. S., et al. (2016). Cardinal: an R package for faithfulness of explanations. *arXiv:1906.09754*.
8. Hooker, S., Erhan, D., Kindermans, P.-J., & Kim, B. (2019). A benchmark for interpretability methods in deep neural networks. *NeurIPS*.
