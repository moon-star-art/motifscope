"""基于梯度的归因方法。

本项目中的模型是 numpy 实现的、无自动微分。因此这里**显式推导**梯度：

- 线性模型（k-mer 逻辑回归）：∂f/∂x_j[b] = w_j[b]；
- PWM max 模型：只有得分最高窗口内的碱基有非零梯度，值为对应 PWM 条目。

这样既保证正确性，又能独立于自动微分框架，便于审计与教学。
"""

from __future__ import annotations

import numpy as np

from ..motifs.pwm import DEFAULT_BACKGROUND
from ..models.conv import ConvMaxPoolModel
from ..models.sequence import KmerLogisticModel, PWMScoringModel


def _gradient(model, onehot: np.ndarray) -> np.ndarray:
    """计算 ∂f/∂x，形状与输入相同 (N, L, 4)。

    仅支持本项目自带的三个模型；其他模型会尝试有限的数值差分作为兜底。
    """
    n, length, _ = onehot.shape

    if isinstance(model, ConvMaxPoolModel):
        return model.input_gradient(onehot)

    if isinstance(model, KmerLogisticModel):
        # k-mer 模型对 one-hot 输入是线性的，但"逐碱基权重"没有唯一解
        # （一个 k-mer 系数无法拆到它的 k 个碱基上）。这里采用与
        # ``exact_attribution`` 一致的均摊约定，保证 grad x input 与
        # 精确分解同源、二者结果一致。
        # 返回 (N, L, 4)：当前碱基通道填该位置的均摊贡献，其余通道为 0。
        phi = model.exact_attribution(onehot)  # (N, L)
        grad = np.zeros((n, length, 4), dtype=np.float64)
        idx = onehot.argmax(axis=2)
        grad[np.arange(n)[:, None], np.arange(length)[None, :], idx] = phi
        return grad

    if isinstance(model, PWMScoringModel):
        grad = np.zeros((n, length, 4), dtype=np.float64)
        pwms = model._pwms  # list，元素 (4, L_m)
        for m, pwm in enumerate(pwms):
            L = pwm.shape[1]
            if model.pooling == "sum":
                # sum 池化：每个窗口都参与，梯度按位移累加
                for offset in range(L):
                    grad[:, offset : offset + (length - L + 1), :] += pwm[:, offset]
            else:
                windows = np.lib.stride_tricks.sliding_window_view(onehot, L, axis=1)
                per_pos = np.einsum("nwcl,cl->nwl", windows, pwm)  # (N, n_win, L)
                window_scores = per_pos.sum(axis=2)
                best = window_scores.argmax(axis=1)
                for i in range(n):
                    start = int(best[i])
                    grad[i, start : start + L, :] += pwm.T
        return grad

    # 兜底：数值差分
    eps = 1e-4
    grad = np.zeros((n, length, 4), dtype=np.float64)
    base = model.decision_function(onehot)
    for j in range(length):
        for b in range(4):
            pert = onehot.copy()
            pert[:, j, :] = 0.0
            pert[:, j, b] = 1.0
            grad[:, j, b] = (model.decision_function(pert) - base) / eps
    return grad


def gradient_input(model, onehot: np.ndarray) -> np.ndarray:
    """梯度×输入：``grad ⊙ x``。

    对 one-hot 输入，这等价于只保留"当前碱基"通道的梯度值，
    其他三个通道为 0（因为 x=0）。
    """
    g = _gradient(model, onehot)
    return g * onehot


def integrated_gradients(
    model,
    onehot: np.ndarray,
    baseline: np.ndarray | None = None,
    steps: int = 64,
) -> np.ndarray:
    """积分梯度（Sundararajan et al., 2017）。

    IG_i = (x_i - x'_i) * ∫_0^1 ∂f(x' + α(x - x')) / ∂x_i dα

    基线 x' 默认使用背景碱基分布构成的"平均序列"（4 个通道各 0.25），
    也可显式传入。积分用中点黎曼和近似。
    """
    if onehot.ndim == 2:
        onehot = onehot[None, ...]
    if baseline is None:
        bg = DEFAULT_BACKGROUND.copy()
        base = np.broadcast_to(bg[None, None, :], onehot.shape).astype(np.float64)
    else:
        base = np.asarray(baseline, dtype=np.float64)
        if base.ndim == 2:
            base = base[None, ...]
        base = np.broadcast_to(base, onehot.shape).astype(np.float64)

    x = onehot.astype(np.float64)
    diff = x - base
    total = np.zeros_like(x)

    for k in range(steps):
        alpha = (k + 0.5) / steps
        interp = (base + alpha * diff).astype(np.float32)
        total += _gradient(model, interp)
    total /= steps
    return diff * total


def smoothgrad(model, onehot: np.ndarray, n_samples: int = 20, sigma: float = 0.1, seed: int = 0) -> np.ndarray:
    """SmoothGrad（Smilkov et al., 2017）：对加噪输入多次求梯度取平均。

    对 one-hot 输入，噪声加在"有效概率"空间，即把 one-hot 视为分布后
    加入高斯噪声并重新归一化，避免产生非法的多通道同时为 1 的输入。
    """
    rng = np.random.default_rng(seed)
    x = onehot.astype(np.float64)
    acc = np.zeros_like(x)
    for _ in range(n_samples):
        noisy = np.clip(x + rng.normal(0.0, sigma, size=x.shape), 0.0, None)
        s = noisy.sum(axis=-1, keepdims=True)
        noisy = np.where(s > 0, noisy / np.maximum(s, 1e-12), 1.0 / 4.0)
        acc += _gradient(model, noisy.astype(np.float32)) * x
    return acc / n_samples
