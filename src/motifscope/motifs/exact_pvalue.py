"""精确 PWM 得分分布：用卷积计算解析 p 值，替代经验零分布。

为什么需要
----------
经验零分布需要采样 n 条随机序列才能分辨 1/n 的 p 值。若多重检验校正要求
p < 1e-7，用 n=400 的采样完全无法分辨（最低只能到 1/401 ≈ 2.5e-3），
结果是"真位点也检不出来"。

精确方法
--------
PWM 得分是各位置独立贡献之和：
    score = Σ_{j=0}^{L-1} w[b_j, j]，其中 b_j ~ 背景分布。
因此其分布是 L 个离散分布的**卷积**。每个位置只有 4 个可能取值，
卷积可精确计算，得到完整的累积分布函数，p 值分辨率仅受浮点精度限制。

实现
----
把每个位置 4 个得分量化到公共网格（相对最小值的偏移），
用 ``np.convolve`` 逐位置卷积概率质量函数。
"""

from __future__ import annotations

import numpy as np

from ..motifs.pwm import DEFAULT_BACKGROUND, PWM


def exact_score_pmf(
    motif: PWM,
    background: np.ndarray | None = None,
    n_bins: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """用卷积计算 PWM 得分的离散概率分布。

    Returns
    -------
    (grid, pmf)
        ``grid`` 为得分取值网格（升序），``pmf`` 为对应概率质量，二者等长。
    """
    bg = DEFAULT_BACKGROUND if background is None else np.asarray(background, dtype=np.float64)
    bg = bg / bg.sum()
    pwm = motif.pwm(background=bg)  # (4, L)

    # 得分的最小可能值：每列取最小（最不利）碱基
    w_min = float(pwm.min(axis=0).sum())
    w_max = float(pwm.max(axis=0).sum())
    span = max(w_max - w_min, 1e-9)
    step = span / n_bins
    if step <= 0:
        step = 1e-6

    # 每列每碱基相对 w_min 的整数偏移
    offsets = np.rint((pwm - pwm.min(axis=0, keepdims=True)) / step).astype(np.int64)  # (4, L)
    max_off = int(offsets.max()) if offsets.size else 0

    # 逐列卷积（离散分布）
    pmf = np.array([1.0])
    for j in range(motif.length):
        # 该列的局部分布：偏移 -> 概率
        local = np.zeros(max_off + 1, dtype=np.float64)
        for b in range(4):
            if bg[b] > 0:
                local[offsets[b, j]] += bg[b]
        pmf = np.convolve(pmf, local)

    # 网格：第 k 个质量点对应 w_min + k*step
    grid = w_min + np.arange(len(pmf)) * step
    total = pmf.sum()
    if total > 0:
        pmf = pmf / total
    return grid, pmf


def exact_pvalues(
    scores: np.ndarray,
    motif: PWM,
    background: np.ndarray | None = None,
    n_bins: int = 4000,
    min_pvalue: float | None = None,
) -> np.ndarray:
    """给定观测得分，返回精确的右尾 p 值 P(Score >= s)。

    Parameters
    ----------
    min_pvalue
        p 值下限。由于离散量化，理论最小可分辨 p 值约等于最后一个网格点的
        概率质量；若有多个得分落在同一最高网格点，p 值会算成 0，不利于
        下游的对数运算与多重检验。默认将其钳制到 ``1/n_bins`` 量级。
    """
    grid, pmf = exact_score_pmf(motif, background=background, n_bins=n_bins)
    cdf = np.cumsum(pmf)
    s = np.atleast_1d(np.asarray(scores, dtype=np.float64))
    # 找到严格小于 s 的最后一个网格点
    pos = np.searchsorted(grid, s, side="left")  # 第一个 grid >= s 的位置
    left = np.where(pos > 0, cdf[np.clip(pos - 1, 0, len(cdf) - 1)], 0.0)
    p = np.clip(1.0 - left, 0.0, 1.0)
    floor = min_pvalue if min_pvalue is not None else 1.0 / max(n_bins, 1)
    return np.maximum(p, floor)
