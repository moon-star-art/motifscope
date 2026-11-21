# motifscope

**DNA 调控序列模型的可解释性验证工具包**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-158%20passed-brightgreen)](#测试)

---

## 这个工具解决什么问题

调控基因组学中的深度模型（DeepSEA、Basset、Enformer 等）能较准确地预测某段 DNA
是否为增强子、剪接位点或转录因子结合位点。但一个更关键的问题长期被忽视：

> **模型究竟是"看懂了"生物学基序，还是只记住了训练集的噪声？**

回答这个问题要依赖归因方法（gradient × input、ISM、integrated gradients、
SmoothGrad 等），而这些方法本身**几乎没有被定量验证过**——因为真实数据没有
"哪个碱基真正重要"的标准答案。

`motifscope` 的思路是：**用受控植入的合成数据制造真值**，从而把
"归因是否可信"变成一个可以算出数字的实验问题。

---

## 核心特性

| 特性 | 说明 |
|---|---|
| **真值可算** | 从 JASPAR PWM 按概率采样基序实例并植入背景序列，逐碱基记录真值掩码 |
| **三个可解释模型** | 多基序 PWM 扫描模型、k-mer 词袋逻辑回归、卷积 + 全局 max 池化（纯 numpy 实现，无需深度学习框架） |
| **完整性可验证** | 三个模型均满足 Σφ = f(x) − f(0)，实测误差 ~1e-15（见下） |
| **四种归因方法** | 精确分解、ISM、梯度×输入、积分梯度、SmoothGrad |
| **双层评估** | 回收型（位置 AUC / top-k 命中）+ 扰动型（删除曲线 / AOPC）+ 稳定性（跨方法 Spearman） |
| **真实数据支持** | UCI 剪接位点数据集（3190 条，EI/IE/N 三类） |
| **实用功能** | `motifscope scan` 在任意 FASTA 上注释 TF 结合位点，输出 BED6 |
| **精确统计** | 用卷积计算 PWM 得分的**解析分布**，p 值分辨率达 1e-15，支持 Bonferroni 校正 |
| **离线可用** | 内置 19 条 JASPAR 基序，无网络也能完整跑通 `demo` |

---

## 安装

```bash
git clone https://github.com/moon-star-art/motifscope.git
cd motifscope
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e .
```

依赖：`numpy`、`pandas`、`scipy`、`scikit-learn`、`matplotlib`、`jinja2`、`requests`。
**纯 CPU 即可运行**，不需要 GPU。

---

## 快速开始

### 一键演示（约 1 分钟）

```bash
motifscope demo
```

会依次完成：加载基序 → 生成合成数据 → 训练三个模型 → 计算归因 →
评测忠实度 → 生成中文 HTML 报告 `motifscope_report.html`。

### 注释 TF 结合位点

```bash
motifscope scan examples/example.fa --pvalue 1e-2 --bed hits.bed
```

输出示例：

```
读入 3 条序列，使用 19 条基序，p 值阈值 0.01
多重检验校正: 开启(Bonferroni)  重叠命中抑制: 开启
共找到 3 个显著命中

EGFR_promoter_region (长度 666): 2 个命中
    351-362   +  NFE2L2         (MA0150.1  )  score= 17.68  p=0.00e+00
    577-588   +  NFKB1          (MA0105.3  )  score= 17.35  p=2.38e-07

CTCF_binding_site (长度 180): 1 个命中
     60-79    +  CTCF           (MA0139.1  )  score= 26.11  p=3.64e-12

random_background (长度 177): 0 个命中
```

### 复现论文实验

```bash
motifscope bench --out results
```

产物：`benchmark.json`、`results.csv`、4 张图。

---

## 主要发现

### 1. 三个模型都满足完整性公理

归因的完整性（completeness）要求各位置贡献之和等于预测值相对基线的变化：

$$\sum_i \varphi_i = f(x) - f(0)$$

实测误差：

| 模型 | max\|Σφ − (f(x)−f(0))\| | 量级 |
|---|---|---|
| PWM-max | 7.1e-15 | 浮点精度 |
| k-mer LR | 2.8e-16 | 浮点精度 |
| CNN-max | 8.9e-16 | 浮点精度 |

这不是巧合：全局 max 池化使得梯度只在 argmax 窗口内非零，因此对 one-hot 输入
`grad × input` **就是**精确分解，而非近似。

### 2. 过拟合模型的归因会完全失效

k-mer 逻辑回归在训练集上准确率 1.000，但其归因的：

- 位置级 AUC ≈ **0.51**（与随机无异）
- AOPC = **−1.24**（负值意味着"按归因删除反而让预测上升"）

也就是说，**模型准确率与归因可信度是两回事**。一个在测试集上表现最好的模型，
其解释可能完全是噪声。这是本项目最值得强调的结论。

### 3. 卷积核宽度需要与基序长度匹配

固定核宽的 CNN 在不同长度基序上表现差异显著：

| 基序长度组 | 最优核宽 | 位置 AUC |
|---|---|---|
| 短 (6–8 bp) | w=6 | 0.623 |
| 中 (10–12 bp) | w=10 | **0.718** |
| 长 (15–17 bp) | w=12 | **0.831** |

核宽显著大于基序长度时，归因质量明显下降（短基序在 w=16 时降到 0.562）。

### 4. 扰动型指标本身也有局限

实验中发现：当模型未充分学到基序信号时，**按真值删除位置也不会使预测下降**，
AOPC 因此失去区分度（真值排序甚至可能不如随机排序）。

这说明单一指标不可靠。`motifscope` 因此同时报告回收型与扰动型两类指标，
并在文档中明确各自的适用条件。

---

## 项目结构

```
motifscope/
├── src/motifscope/
│   ├── seq.py                  # one-hot 编码、反向互补、k-mer
│   ├── data/
│   │   ├── synthetic.py        # 合成数据（带逐碱基真值）
│   │   └── splice.py           # UCI 剪接位点数据
│   ├── motifs/
│   │   ├── pwm.py              # PWM 模型、扫描、经验 p 值
│   │   ├── exact_pvalue.py     # 卷积法精确 p 值
│   │   └── database.py         # JASPAR 基序库（缓存 + 离线兜底）
│   ├── models/
│   │   ├── sequence.py         # PWM 扫描模型、k-mer 逻辑回归
│   │   └── conv.py             # 卷积 + 全局 max 池化（纯 numpy）
│   ├── attribution/            # 精确分解 / ISM / 梯度 / IG / SmoothGrad
│   ├── evaluate/
│   │   ├── faithfulness.py     # 删除曲线、AOPC、位置 AUC、稳定性
│   │   └── scan.py             # 基序扫描（NMS + 多重检验校正）
│   ├── report/                 # 中文 HTML 报告
│   ├── resources/              # 内置 19 条 JASPAR 基序
│   └── cli.py
├── tests/                      # 158 个测试，覆盖率 89%
├── examples/
│   ├── run_benchmark.py
│   └── example.fa
├── docs/
└── results/
```

---

## 方法说明

### 为什么用合成数据

真实基因组数据没有逐碱基的重要性标注。用 PWM 采样植入可以精确控制：
植入位置、基序种类、丰度、重叠程度、正负链方向，从而得到**可计算的真值**。

### 为什么需要三个模型

- **PWM 扫描模型**：生物学最可解释，提供强基线
- **k-mer 逻辑回归**：严格线性，是检验归因方法收敛性的"标准答案"
- **CNN-max**：接近真实深度模型的结构，检验方法在复杂模型上的表现

### 精确 p 值的必要性

经验零分布需要采样 n 条序列才能分辨 1/n 的 p 值。做 Bonferroni 校正时
（6700 次检验 → α ≈ 1.5e-7）经验方法完全失效，会出现"真位点也检不出"。
而 PWM 得分是各位置独立贡献之和，其分布是 4 点分布的卷积，可以**精确解析计算**。

---

## 测试

```bash
pip install -e ".[dev]"
pytest --cov=motifscope
```

```
158 passed in ~3s
TOTAL coverage: 89%
```

---

## 引用

如果你在研究中使用了本工具，请引用：

```bibtex
@software{motifscope2026,
  title  = {motifscope: A toolkit for benchmarking attribution faithfulness
            in regulatory DNA sequence models},
  author = {moon-star-art},
  year   = {2026},
  url    = {https://github.com/moon-star-art/motifscope}
}
```

## 许可

MIT License，详见 [LICENSE](LICENSE)。

## 致谢

基序数据来自 [JASPAR](https://jaspar.genereg.net/)；剪接位点数据来自
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/69/)。
