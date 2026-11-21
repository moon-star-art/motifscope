"""精确归因：直接调用模型提供的解析分解。"""

from __future__ import annotations

import numpy as np


def exact(model, onehot: np.ndarray) -> np.ndarray:
    """使用模型自带的精确分解，返回 (N, L, 4) 贡献。

    目前仅 :class:`~motifscope.models.PWMScoringModel` 与
    :class:`~motifscope.models.KmerLogisticModel` 支持精确分解。

    说明：由于精确分解是针对"当前碱基"的加性贡献，其他三个未出现在输入
    中的碱基通道记为 0（它们对当前预测没有贡献）。
    """
    if not hasattr(model, "exact_attribution"):
        raise TypeError(f"模型 {type(model).__name__} 不支持精确分解")
    per_pos = np.asarray(model.exact_attribution(onehot), dtype=np.float64)  # (N, L)
    out = np.zeros(onehot.shape, dtype=np.float64)
    idx = onehot.argmax(axis=2)  # (N, L)
    n, length = idx.shape
    out[np.arange(n)[:, None], np.arange(length)[None, :], idx] = per_pos
    return out
