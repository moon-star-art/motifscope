"""ISM（In-silico Mutagenesis，单碱基饱和突变）归因。

原理
----
对序列的每个位置 j，把该位置替换为其他三个碱基，各计算一次预测变化：

    Δ(j, b) = f(x) - f(x with position j set to base b)

对二分类模型，"该位置重要性"通常取所有替换中预测变化的最大值或平均值。
本实现返回 (N, L, 4) 的矩阵：

- 对**当前碱基**通道，填入 0（替换为自身不变）；
- 对**其他碱基**通道，填入 f(x) - f(x_{j→b})。

这样与"梯度×输入"的语义对齐：正值表示"这个碱基的贡献为正"。

复杂度
------
每条序列需要 3L 次前向推理。对 L=200 即 600 次，模型若是 numpy 实现
（PWM / k-mer）可以批量向量化，速度可接受。
"""

from __future__ import annotations

import numpy as np


def in_silico_mutagenesis(
    model,
    onehot: np.ndarray,
    batch_size: int = 512,
    aggregate: str = "mean",
) -> np.ndarray:
    """计算 ISM 归因，返回 (N, L, 4)。

    Parameters
    ----------
    aggregate
        如何把多个替代碱基的效果汇总到"当前碱基"通道：
        - ``"mean"``：取其他三个碱基变化的平均（推荐，数值更稳）；
        - ``"max"``：取最大绝对值（更敏感，易受单点噪声影响）。
    """
    if onehot.ndim == 2:
        onehot = onehot[None, ...]
    x = onehot.astype(np.float32)
    n, length, _ = x.shape

    base_pred = model.decision_function(x).astype(np.float64)  # (N,)
    delta = np.zeros((n, length, 4), dtype=np.float64)

    # 构造所有单点突变，分批前向
    variants = []
    meta = []
    for i in range(n):
        for j in range(length):
            cur = int(x[i, j].argmax())
            for b in range(4):
                if b == cur:
                    continue
                variants.append((i, j, b, cur))
    if not variants:
        return delta

    arr = np.repeat(x, 3 * length, axis=0).reshape(len(variants), length, 4).copy()
    for idx, (i, j, b, _cur) in enumerate(variants):
        arr[idx, j, :] = 0.0
        arr[idx, j, b] = 1.0

    preds = np.empty(len(variants), dtype=np.float64)
    for start in range(0, len(variants), batch_size):
        stop = min(start + batch_size, len(variants))
        preds[start:stop] = model.decision_function(arr[start:stop])

    for idx, (i, j, b, _cur) in enumerate(variants):
        delta[i, j, b] = base_pred[i] - preds[idx]

    # 汇总到当前碱基通道
    out = np.zeros((n, length, 4), dtype=np.float64)
    for i in range(n):
        for j in range(length):
            cur = int(x[i, j].argmax())
            others = [delta[i, j, b] for b in range(4) if b != cur]
            out[i, j, cur] = np.mean(others) if aggregate == "mean" else max(others, key=abs)
    return out
