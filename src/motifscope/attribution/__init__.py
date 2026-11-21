"""归因方法：给定模型，计算每个碱基对预测的重要性。

统一接口
--------
所有归因函数签名为 ``attribute(model, onehot, **kwargs) -> ndarray``，
输入 (N, L, 4) one-hot（或单条 (L, 4)），输出 (N, L, 4) 或 (L, 4) 的重要性。

之所以保留 4 个通道而不是直接求和，是为了支持"这个位置换成 T 会怎样"这类
碱基级分析；下游的绝大部分指标都对通道求和得到逐位置标量。
"""

from __future__ import annotations

from .base import (
    ATTRIBUTION_METHODS,
    attribute,
    get_attribution_fn,
    normalize_scores,
    position_scores,
)
from .exact import exact
from .gradient import gradient_input, integrated_gradients, smoothgrad
from .ism import in_silico_mutagenesis

__all__ = [
    "ATTRIBUTION_METHODS",
    "attribute",
    "exact",
    "get_attribution_fn",
    "gradient_input",
    "in_silico_mutagenesis",
    "integrated_gradients",
    "normalize_scores",
    "position_scores",
    "smoothgrad",
]
