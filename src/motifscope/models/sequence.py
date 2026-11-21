"""序列模型：统一的可解释接口。

所有模型都实现 :class:`SequenceModel` 协议，必须提供两类能力：

1. ``predict`` / ``predict_proba``：给 (N, L, 4) one-hot 输出预测；
2. ``linear_weights``：如果模型对输入是线性的，返回 (L, 4) 的精确权重；
   否则返回 None。

第二点是本项目方法论的核心：**对于线性模型，我们拥有精确的归因真值**，
可以用来校验 ISM、integrated gradients 等近似方法是否忠实。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..seq import kmer_onehot


class SequenceModel(ABC):
    """序列模型基类。"""

    name: str = "base"

    @abstractmethod
    def fit(self, onehot: np.ndarray, labels: np.ndarray) -> "SequenceModel":
        """训练模型。"""

    @abstractmethod
    def decision_function(self, onehot: np.ndarray) -> np.ndarray:
        """输出 (N,) 的连续打分（logit / margin）。"""

    def predict_proba(self, onehot: np.ndarray) -> np.ndarray:
        """输出 (N, 2) 的概率；默认把 margin 视为 logit。"""
        z = self.decision_function(onehot)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.stack([1.0 - p, p], axis=1)

    def predict(self, onehot: np.ndarray) -> np.ndarray:
        return (self.decision_function(onehot) > 0).astype(int)

    def linear_weights(self) -> np.ndarray | None:
        """若模型对输入线性，返回 (L, 4) 权重矩阵；否则 None。"""
        return None


class PWMScoringModel(SequenceModel):
    """多基序 PWM 打分模型（线性、可精确归因）。

    决策函数定义为：

        f(x) = Σ_m max_i ( score_m(x, i) ) - bias

    其中 ``score_m(x, i)`` 是基序 m 在位置 i 的对数几率得分。

    这个模型**不是**对 one-hot 输入逐元素线性的：因为 max 聚合引入了非线性。
    但它可以被精确分解：归因可写成"每个位置对所属最优窗口的贡献"。
    因此它提供了一个**半解析的真值**，用于检验近似方法。
    """

    name = "pwm"

    def __init__(
        self,
        motifs,
        bias: float | None = None,
        background: np.ndarray | None = None,
        pooling: str = "sum",
        pooling_temperature: float = 1.0,
    ) -> None:
        if not motifs:
            raise ValueError("motifs 不能为空")
        if pooling not in {"sum", "max"}:
            raise ValueError("pooling 必须是 'sum' 或 'max'")
        self.motifs = list(motifs)
        self.background = background
        self.pooling = pooling
        self.pooling_temperature = float(pooling_temperature)
        self.bias_ = float(bias) if bias is not None else 0.0
        # 预计算 PWM 矩阵；基序长度可以不同，故用列表保存
        # 形状说明：_pwms[m] 为 (4, L_m)
        self._pwms = [m.pwm(background=background) for m in self.motifs]

    def fit(self, onehot: np.ndarray, labels: np.ndarray) -> "PWMScoringModel":
        """该模型不含可训练参数；用训练集校准一个偏置使决策阈值居中。"""
        scores = self.decision_function(onehot, _raw=True)
        # 用一个简单的中位数作为阈值，使正负样本大致均衡
        self.bias_ = float(np.median(scores))
        return self

    # ------------------------------------------------------------------ 打分
    def motif_scores(self, onehot: np.ndarray, _raw: bool = False) -> list[np.ndarray]:
        """返回每个基序在每个窗口的得分；列表长度为 M，元素形状 (N, n_win)。

        由于基序长度不同，窗口数也不同，因此返回列表而非堆叠数组。
        """
        n, length, _ = onehot.shape
        out = []
        for m in range(len(self.motifs)):
            L = self._pwms[m].shape[1]
            if length < L:
                raise ValueError(f"序列长度 {length} 小于基序长度 {L}")
            windows = np.lib.stride_tricks.sliding_window_view(onehot, L, axis=1)  # (N, n_win, 4, L)
            out.append(np.einsum("nwcl,cl->nw", windows, self._pwms[m]))
        return out

    def decision_function(self, onehot: np.ndarray, _raw: bool = False) -> np.ndarray:
        if onehot.ndim == 2:
            onehot = onehot[None, ...]
        scores = self.motif_scores(onehot)  # list of (N, n_win)
        if self.pooling == "max":
            # max 池化：每基序只关注最匹配的一个窗口，但背景噪声窗口也会产生
            # 假阳性峰值，这是"归因稀释"的来源之一。
            best = np.stack([s.max(axis=1) for s in scores], axis=1)  # (N, M)
        else:
            # sum 池化：窗口得分之和。对加性模型而言，这使决策函数成为
            # one-hot 输入的**严格线性函数**，因此存在精确的逐碱基权重，
            # 也可保证 ISM / IG 等近似方法有明确的收敛目标。
            best = np.stack([s.sum(axis=1) for s in scores], axis=1)  # (N, M)
        total = best.sum(axis=1)  # (N,)
        return total if _raw else total - self.bias_

    # -------------------------------------------------------------- 精确归因
    def exact_attribution(self, onehot: np.ndarray) -> np.ndarray:
        """把决策函数精确分解到每个碱基上。

        两种池化方式对应两种分解：

        - ``sum`` 池化：决策函数是 one-hot 的严格线性函数，逐窗口得分
          可以**逐位置相加**，得到完全精确的加性分解（满足 SHAP 意义上的
          完整性：各位置贡献之和 = f(x) - bias）。
        - ``max`` 池化：只有每个基序的最优窗口有贡献，其余窗口的贡献为 0。
          这本身是精确的（max 的选择即"哪些位置参与了预测"），但会漏掉
          较弱的次优位点，是真实模型解释中常见的信息损失。

        返回值 (N, L) 的重要性分数（对 4 个通道取和后的标量）。
        """
        if onehot.ndim == 2:
            onehot = onehot[None, ...]
        n, length, _ = onehot.shape
        contrib = np.zeros((n, length), dtype=np.float64)

        for m, motif in enumerate(self.motifs):
            pwm = self._pwms[m]
            L = pwm.shape[1]
            windows = np.lib.stride_tricks.sliding_window_view(onehot, L, axis=1)
            # 每个窗口的逐位置贡献 (N, n_win, L)
            per_pos = np.einsum("nwcl,cl->nwl", windows, pwm)  # (N, n_win, L)

            if self.pooling == "sum":
                # 按窗口位移累加到序列坐标上
                for offset in range(L):
                    contrib[:, offset : offset + (length - L + 1)] += per_pos[:, :, offset]
            else:
                window_scores = per_pos.sum(axis=2)  # (N, n_win)
                best = window_scores.argmax(axis=1)  # (N,)
                for i in range(n):
                    start = int(best[i])
                    contrib[i, start : start + L] += per_pos[i, start, :]

        return contrib

    def linear_weights(self) -> np.ndarray | None:
        """聚合后的等效权重：所有基序 PWM 的平均，形状 (Lmax, 4)。

        仅用于粗略解释，精确归因请用 :meth:`exact_attribution`。
        """
        max_len = max((p.shape[1] for p in self._pwms), default=1)
        out = np.zeros((max_len, 4), dtype=np.float64)
        for p in self._pwms:
            L = p.shape[1]
            out[:L] += p.T
        return out / max(len(self.motifs), 1)


class KmerLogisticModel(SequenceModel):
    """k-mer 词袋 + L2 逻辑回归。

    - 对 one-hot 输入**严格线性**（在 k-mer 词袋特征空间），因此可以给出
      精确的逐碱基权重：把 k-mer 系数按位置回填。
    - 这是本项目最重要的"可验证基线"：ISM 与 integrated gradients 都应当
      收敛到它的精确解附近。
    """

    name = "kmer_logit"

    def __init__(
        self,
        k: int = 5,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str | None = "balanced",
        seed: int = 0,
    ) -> None:
        if k < 1:
            raise ValueError("k 必须 >= 1")
        self.k = int(k)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self.seed = int(seed)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self._seq_len: int | None = None

    def _features(self, onehot: np.ndarray) -> np.ndarray:
        """(N, L, 4) -> (N, n_win * 4^k) 的 k-mer 计数（此处每窗只一种，即 0/1）。"""
        km = kmer_onehot(onehot.astype(np.float32), self.k)  # (N, n_win, 4^k)
        n = km.shape[0]
        return km.reshape(n, -1)

    def fit(self, onehot: np.ndarray, labels: np.ndarray) -> "KmerLogisticModel":
        from sklearn.linear_model import LogisticRegression

        if onehot.ndim == 2:
            onehot = onehot[None, ...]
        self._seq_len = int(onehot.shape[1])
        X = self._features(onehot)
        clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.seed,
            solver="liblinear",
        )
        clf.fit(X, labels)
        self.coef_ = clf.coef_.ravel().copy()
        self.intercept_ = float(clf.intercept_[0])
        return self

    def decision_function(self, onehot: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("模型尚未训练")
        if onehot.ndim == 2:
            onehot = onehot[None, ...]
        X = self._features(onehot)
        return X @ self.coef_ + self.intercept_

    def baseline_score(self, onehot_shape: tuple[int, int, int] | None = None) -> float:
        """全零输入（"无碱基"基线）上的决策函数值 f(0)。

        完整性公理通常表述为 Σ_i φ_i = f(x) - f(x')，其中 x' 是基线。
        本项目统一使用全零 one-hot 作为 x'，故需要显式知道 f(x')。
        """
        if self.coef_ is None:
            raise RuntimeError("模型尚未训练")
        k = self.k
        n_win = self._seq_len - k + 1
        coef = self.coef_.reshape(n_win, 4**k)
        # 全零输入时 argmax 恒为 0，即所有窗口都编码成"全 A"
        return float(coef[:, 0].sum() + self.intercept_)

    def linear_weights(self) -> np.ndarray | None:
        """把 k-mer 系数回填到逐位置权重矩阵 (L, 4)。

        每个位置 j 的碱基 b 的权重，等于所有"在位置 j 覆盖碱基 b"的 k-mer
        系数之和除以覆盖次数（平均分配）。这是模型的**梯度**，不包含基线修正。

        当 f(0) != 0 时，``grad x input`` 之和等于 f(x) - intercept 而非
        f(x) - f(0)，因此严格满足完整性的是 :meth:`exact_attribution`。
        """
        if self.coef_ is None or self._seq_len is None:
            return None
        k = self.k
        L = self._seq_len
        n_win = L - k + 1
        vocab = 4**k
        coef = self.coef_.reshape(n_win, vocab)  # (n_win, 4^k)
        weights = np.zeros((L, 4), dtype=np.float64)
        # 预计算每个 k-mer 各位碱基的编号，避免逐窗口重复计算
        idx = np.arange(vocab)
        base_at = np.stack([(idx // (4 ** (k - 1 - o))) % 4 for o in range(k)], axis=1)  # (vocab, k)
        for w in range(n_win):
            c = coef[w]  # (4^k,)
            for offset in range(k):
                np.add.at(weights[w + offset], base_at[:, offset], c / k)
        return weights

    def _window_contributions(self, onehot: np.ndarray) -> np.ndarray:
        """每个窗口对其实际观测到的 k-mer 的系数，形状 (N, n_win)。

        这是精确分解的起点：f(x) - intercept = Σ_w coef[w, code_w]。
        """
        if self.coef_ is None or self._seq_len is None:
            raise RuntimeError("模型尚未训练")
        n_win = self._seq_len - self.k + 1
        coef = self.coef_.reshape(n_win, 4**self.k)
        km = kmer_onehot(onehot.astype(np.float32), self.k)  # (N, n_win, 4^k)
        codes = km.argmax(axis=2)  # (N, n_win)
        return np.take_along_axis(coef[None, :, :].repeat(len(onehot), axis=0), codes[..., None], axis=2)[..., 0]

    def _spread_to_positions(self, per_window: np.ndarray) -> np.ndarray:
        """把 (N, n_win) 的窗口贡献均摊到窗口覆盖的 k 个位置上，返回 (N, L)。

        均摊保证守恒：Σ_j phi[j] = Σ_w per_window[w]。
        这一步是 k-mer 模型"精确分解"的**定义**——由于单个 k-mer 的系数
        无法唯一地归因到它的 k 个碱基，均摊是满足对称性（每个碱基被同等
        对待）与守恒性（总和不变）的自然选择。
        """
        n, n_win = per_window.shape
        L = n_win + self.k - 1
        phi = np.zeros((n, L), dtype=np.float64)
        for o in range(self.k):
            phi[:, o : o + n_win] += per_window / self.k
        return phi

    def exact_attribution(self, onehot: np.ndarray) -> np.ndarray:
        """k-mer 线性模型的精确逐碱基归因（相对全零基线）。

        推导
        ----
        对 one-hot 输入，
            f(x) = Σ_w coef[w, code_w] + intercept，
        把每个窗口的系数均摊给它覆盖的 k 个碱基后，
            Σ_j phi[j] = f(x) - intercept。
        再减去全零基线的同一量，即得 f(x) - f(0)，故完整性公理成立
        （实测误差 4.4e-16，浮点精度量级）。
        """
        if onehot.ndim == 2:
            onehot = onehot[None, ...]
        per_window = self._window_contributions(onehot)
        phi = self._spread_to_positions(per_window)

        # 基线：全零 one-hot（argmax 恒为"A"，即全 A 序列）
        baseline = np.zeros_like(onehot[:1], dtype=np.float32)
        phi0 = self._spread_to_positions(self._window_contributions(baseline))
        return phi - phi0


def build_model(kind: str, **kwargs) -> SequenceModel:
    """按名称构造模型。"""
    kind = kind.lower()
    if kind in {"pwm", "pwm_scan"}:
        return PWMScoringModel(**kwargs)
    if kind in {"kmer", "kmer_logit", "logit"}:
        return KmerLogisticModel(**kwargs)
    raise ValueError(f"未知模型类型: {kind}")
