"""归因方法的统一注册与分发。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

#: 方法名 -> (显示名, 是否为精确方法)
ATTRIBUTION_METHODS: dict[str, dict] = {
    "exact": {"label": "精确分解", "exact": True, "desc": "利用模型的解析形式直接分解预测值"},
    "ism": {"label": "ISM 单碱基饱和突变", "exact": False, "desc": "逐位置替换为其他碱基，观察预测变化"},
    "grad_input": {"label": "梯度×输入", "exact": False, "desc": "一阶泰勒近似的逐位置贡献"},
    "ig": {"label": "积分梯度", "exact": False, "desc": "沿背景到输入的直线路径积分梯度"},
    "smoothgrad": {"label": "SmoothGrad", "exact": False, "desc": "对加噪输入多次求梯度取平均"},
}


def get_attribution_fn(method: str) -> Callable:
    """按名称返回归因函数。"""
    m = method.lower()
    from .exact import exact
    from .gradient import gradient_input, integrated_gradients, smoothgrad
    from .ism import in_silico_mutagenesis

    table = {
        "exact": exact,
        "ism": in_silico_mutagenesis,
        "grad_input": gradient_input,
        "ig": integrated_gradients,
        "smoothgrad": smoothgrad,
    }
    if m not in table:
        raise ValueError(f"未知归因方法: {method}；可用: {sorted(table)}")
    return table[m]


def attribute(model, onehot: np.ndarray, method: str = "ism", **kwargs) -> np.ndarray:
    """对单条或批量输入计算归因，自动保持输入维度。"""
    fn = get_attribution_fn(method)
    squeeze = onehot.ndim == 2
    x = onehot[None, ...] if squeeze else onehot
    out = fn(model, x, **kwargs)
    return out[0] if squeeze else out


def position_scores(attr: np.ndarray) -> np.ndarray:
    """把 (N, L, 4) 的碱基级归因压缩为 (N, L) 的位置级分数。

    使用**绝对值和**而非直接求和：正负贡献都代表"该位置重要"，
    直接求和会因为正负抵消而低估重要性。
    """
    return np.abs(attr).sum(axis=-1)


def normalize_scores(scores: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """按序列把分数缩放到 [0, 1]，便于跨方法比较形状。"""
    s = np.asarray(scores, dtype=np.float64)
    lo = s.min(axis=-1, keepdims=True)
    hi = s.max(axis=-1, keepdims=True)
    return (s - lo) / np.maximum(hi - lo, eps)
