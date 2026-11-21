"""卷积 + 全局 max 池化模型（纯 numpy 实现，无需深度学习框架）。

为什么需要这个模型
------------------
在"序列中是否存在某基序"这类任务上，**全局 max 池化是关键**：
- 若用 sum 池化（或线性模型），195 个窗口的背景噪声会累加并淹没单个基序信号
  （实测准确率 0.488，低于随机）；
- 全局 max 池化只取每个卷积核最响应的那一个窗口，天然对应"一个基序只有一个
  结合位点"的生物学事实，实测准确率显著提升。

架构
----
    x (N, L, 4)
      -> 一维卷积 M 个核，宽度 w        -> (N, M, L-w+1)
      -> 全局 max 池化                  -> (N, M)
      -> 线性输出 + sigmoid             -> (N,)

可解释性的关键性质
------------------
对 one-hot 输入，全局 max 池化使得**梯度只在 argmax 窗口内非零**：

    ∂f/∂x[i, b] = v_m * W[m, b, i - i*_m]   （当 i 落在第 m 个核的最优窗口内）

因此 ``grad x input`` 恰好等于 ``f(x) - f(0)``，即**精确满足完整性公理**
（completeness）。这是本项目用来自证"归因可信"的理论支点。
"""

from __future__ import annotations

import numpy as np

from .sequence import SequenceModel


class ConvMaxPoolModel(SequenceModel):
    """一维卷积 + 全局 max 池化 + 逻辑输出的二分类模型。"""

    name = "conv_maxpool"

    def __init__(
        self,
        n_filters: int = 16,
        width: int = 10,
        n_epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-2,
        l2: float = 1e-4,
        seed: int = 0,
        init_scale: float | None = None,
    ) -> None:
        self.n_filters = int(n_filters)
        self.width = int(width)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.init_scale = init_scale
        self.W: np.ndarray | None = None  # (M, 4, w)
        self.b: np.ndarray | None = None  # (M,)
        self.v: np.ndarray | None = None  # (M,)
        self.c: float = 0.0
        self.history_: list[float] = []

    # ---------------------------------------------------------------- 前向
    def _windows(self, x: np.ndarray) -> np.ndarray:
        """(N, L, 4) -> (N, n_win, 4, w)，满足 windows[n,i,b,o] = x[n, i+o, b]。

        注意 ``sliding_window_view(x, w, axis=1)`` 得到 (N, n_win, 4, w)：
        窗口索引在第 1 维，碱基在第 2 维，窗口内偏移在第 3 维。
        """
        n, length, _ = x.shape
        if length < self.width:
            raise ValueError(f"序列长度 {length} 小于卷积核宽度 {self.width}")
        return np.lib.stride_tricks.sliding_window_view(x, self.width, axis=1)

    def conv_scores(self, x: np.ndarray) -> np.ndarray:
        """卷积响应，形状 (N, M, n_win)。"""
        if self.W is None:
            raise RuntimeError("模型尚未初始化/训练")
        win = self._windows(x)  # (N, n_win, 4, w) -> 下标 n,i,b,o
        # win: n,i,b,o ; W: m,b,o ; out: n,m,i
        return np.einsum("nibo,mbo->nmi", win, self.W) + self.b.reshape(1, -1, 1)

    def pool(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """全局 max 池化，返回 (池化值 (N,M), argmax 位置 (N,M))。"""
        idx = scores.argmax(axis=2)
        vals = np.take_along_axis(scores, idx[:, :, None], axis=2)[:, :, 0]
        return vals, idx

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            x = x[None, ...]
        scores = self.conv_scores(x.astype(np.float32))
        pooled, _ = self.pool(scores)
        return pooled @ self.v + self.c

    # ---------------------------------------------------------------- 训练
    def _init_params(self, x: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        scale = self.init_scale if self.init_scale is not None else np.sqrt(2.0 / (4 * self.width))
        self.W = rng.normal(0.0, scale, size=(self.n_filters, 4, self.width)).astype(np.float64)
        self.b = np.zeros(self.n_filters, dtype=np.float64)
        self.v = np.full(self.n_filters, 1.0 / self.n_filters, dtype=np.float64)
        self.c = 0.0

    def fit(self, onehot: np.ndarray, labels: np.ndarray) -> "ConvMaxPoolModel":
        if onehot.ndim == 2:
            onehot = onehot[None, ...]
        x = onehot.astype(np.float32)
        y = np.asarray(labels, dtype=np.float64)
        n = x.shape[0]
        self._init_params(x)

        # Adam 状态
        params = {"W": self.W, "b": self.b, "v": self.v.copy()}
        m_adam = {k: np.zeros_like(v_) for k, v_ in params.items()}
        s_adam = {k: np.zeros_like(v_) for k, v_ in params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        rng = np.random.default_rng(self.seed + 1)

        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                xb, yb = x[idx], y[idx]
                bs = len(idx)

                # ---- 前向
                scores = self.conv_scores(xb)  # (bs, M, n_win)
                pooled, amax = self.pool(scores)  # (bs, M)
                z = pooled @ self.v + self.c  # (bs,)
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
                eps_l = 1e-12
                loss = -np.mean(yb * np.log(p + eps_l) + (1 - yb) * np.log(1 - p + eps_l))
                epoch_loss += loss * bs

                # ---- 反向
                dz = (p - yb) / bs  # (bs,)
                dv = pooled.T @ dz  # (M,)
                dc = float(dz.sum())
                dpooled = np.outer(dz, self.v)  # (bs, M)

                # max 池化的梯度只回流到 argmax 位置
                dscores = np.zeros_like(scores)  # (bs, M, n_win)
                np.put_along_axis(dscores, amax[:, :, None], dpooled[:, :, None], axis=2)

                win = self._windows(xb)  # (bs, n_win, 4, w) -> n,i,b,o
                dW = np.einsum("nmi,nibo->mbo", dscores, win) + self.l2 * self.W
                db = dscores.sum(axis=(0, 2))

                # ---- Adam 更新
                step += 1
                for key, grad in (("W", dW), ("b", db), ("v", dv)):
                    m_adam[key] = beta1 * m_adam[key] + (1 - beta1) * grad
                    s_adam[key] = beta2 * s_adam[key] + (1 - beta2) * (grad * grad)
                    m_hat = m_adam[key] / (1 - beta1**step)
                    s_hat = s_adam[key] / (1 - beta2**step)
                    params[key] -= self.lr * m_hat / (np.sqrt(s_hat) + eps)

                self.W, self.b, self.v = params["W"], params["b"], params["v"]
                self.c -= self.lr * dc

            self.history_.append(float(epoch_loss / n))
        return self

    # -------------------------------------------------------------- 归因
    def exact_attribution(self, onehot: np.ndarray) -> np.ndarray:
        """max 池化卷积模型的**精确**加性分解。

        由于 f 对 one-hot 输入在 argmax 分支上是局部线性的，

            f(x) - f(0) = Σ_m v_m * score_m(i*_m) = Σ_{i,b} x[i,b] * (∂f/∂x[i,b])

        因此 ``梯度×输入`` 本身就是精确分解，满足完整性公理。
        """
        if onehot.ndim == 2:
            onehot = onehot[None, ...]
        x = onehot.astype(np.float32)
        grad = self.input_gradient(x)  # (N, L, 4)
        return np.einsum("nlb,nlb->nl", grad, x)

    def input_gradient(self, onehot: np.ndarray) -> np.ndarray:
        """∂f/∂x，形状 (N, L, 4)；只在各卷积核的 argmax 窗口内非零。"""
        if onehot.ndim == 2:
            onehot = onehot[None, ...]
        x = onehot.astype(np.float32)
        n, length, _ = x.shape
        scores = self.conv_scores(x)
        _, amax = self.pool(scores)  # (N, M)

        grad = np.zeros((n, length, 4), dtype=np.float64)
        for m in range(self.n_filters):
            contrib = self.v[m] * self.W[m]  # (4, w)
            for i in range(n):
                start = int(amax[i, m])
                grad[i, start : start + self.width, :] += contrib.T
        return grad

    def learned_filters_as_pwm(self) -> list[np.ndarray]:
        """把学到的卷积核导出为 (4, w) 的类 PWM 矩阵，可用于画 sequence logo。"""
        if self.W is None:
            raise RuntimeError("模型尚未训练")
        return [self.W[m].astype(np.float64) for m in range(self.n_filters)]

    def filter_importance(self) -> np.ndarray:
        """每个卷积核对输出的贡献权重 |v_m|。"""
        if self.v is None:
            raise RuntimeError("模型尚未训练")
        return np.abs(self.v)
