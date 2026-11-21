"""PWM / PFM 基序模型：读写、转换、扫描打分与显著性检验。

术语说明
--------
- **PFM** (position frequency matrix)：每个位置上 A/C/G/T 的计数，形状 (4, L)。
- **PPM** (position probability matrix)：把计数归一化为概率，形状 (4, L)。
- **PWM** (position weight matrix)：对数几率矩阵 log2(p / background)，形状 (4, L)。

扫描得分定义
------------
对长度为 L 的基序在序列位置 i 处打分：

    score(i) = Σ_{j=0..L-1} PWM[b_{i+j}, j]

即窗内各位置对数几率之和。得分为正表示该窗口比背景模型更可能由该基序生成。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..seq import BASES, reverse_complement  # noqa: F401

DEFAULT_BACKGROUND = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)


def uniform_background() -> np.ndarray:
    return DEFAULT_BACKGROUND.copy()


def gc_background(gc: float = 0.41) -> np.ndarray:
    """按给定 GC 含量构造背景碱基频率。"""
    at = (1.0 - gc) / 2.0
    return np.array([at, gc / 2.0, gc / 2.0, at], dtype=np.float64)


@dataclass
class PWM:
    """一条基序的计数矩阵及其派生量。

    Attributes
    ----------
    matrix_id : JASPAR 风格编号，例如 ``MA0007.1``；自建基序可为任意字符串。
    name : 基序/转录因子名称。
    counts : (4, L) 的整数或浮点计数矩阵，行顺序为 A/C/G/T。
    """

    matrix_id: str
    name: str
    counts: np.ndarray
    family: str = ""
    collection: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        self.counts = np.asarray(self.counts, dtype=np.float64)
        if self.counts.ndim != 2 or self.counts.shape[0] != 4:
            raise ValueError(f"counts 形状必须是 (4, L)，收到 {self.counts.shape}")
        if self.counts.shape[1] < 2:
            raise ValueError("基序长度至少为 2")
        if (self.counts < 0).any():
            raise ValueError("counts 不能包含负值")
        if self.counts.sum() == 0:
            raise ValueError("counts 不能全为 0")

    # ------------------------------------------------------------------ 属性
    @property
    def length(self) -> int:
        return int(self.counts.shape[1])

    @property
    def consensus(self) -> str:
        """一致序列（每个位置取频率最高的碱基）。"""
        return "".join(BASES[i] for i in self.counts.argmax(axis=0))

    # -------------------------------------------------------------- 矩阵转换
    def ppm(self, pseudocount: float = 0.8, background: np.ndarray | None = None) -> np.ndarray:
        """带伪计数的位置概率矩阵。

        伪计数以背景分布为权重加入（即"背景先验"），这是 JASPAR/Biopython 的
        常见做法，可避免计数为 0 时概率为 0 导致对数几率无穷大。
        """
        bg = DEFAULT_BACKGROUND if background is None else np.asarray(background, dtype=np.float64)
        bg = bg / bg.sum()
        totals = self.counts.sum(axis=0, keepdims=True)
        return (self.counts + pseudocount * bg.reshape(4, 1)) / (totals + pseudocount)

    def pwm(self, pseudocount: float = 0.8, background: np.ndarray | None = None) -> np.ndarray:
        """对数几率矩阵 log2(p / background)，形状 (4, L)。"""
        bg = DEFAULT_BACKGROUND if background is None else np.asarray(background, dtype=np.float64)
        p = self.ppm(pseudocount=pseudocount, background=bg)
        return np.log2(p / bg.reshape(4, 1))

    def information_content(self, pseudocount: float = 0.8) -> float:
        """总信息量（以 bit 计），可用于筛选"信息量足够"的基序。"""
        p = self.ppm(pseudocount=pseudocount)
        return float(np.sum(2.0 + (p * np.log2(np.clip(p, 1e-12, None))).sum(axis=0)))

    # ------------------------------------------------------------------ 扫描
    def score_windows(self, onehot: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
        """对 (N, L_seq, 4) 的 one-hot 批量打分，返回 (N, L_seq - motif_len + 1)。"""
        w = self.pwm(background=background)  # (4, L)
        n, seq_len, _ = onehot.shape
        n_win = seq_len - self.length + 1
        if n_win <= 0:
            raise ValueError(f"序列长度 {seq_len} 小于基序长度 {self.length}")
        # (N, n_win, L, 4) 窗口视图，与 (4, L) 做加权求和
        windows = np.lib.stride_tricks.sliding_window_view(onehot, self.length, axis=1)
        # windows: (N, n_win, 4, L)
        return np.einsum("nwcl,cl->nw", windows, w)

    def scan(self, seq: str, background: np.ndarray | None = None) -> np.ndarray:
        """单条序列的逐窗口得分，返回长度 L_seq - L + 1 的数组。"""
        from ..seq import encode

        return self.score_windows(encode(seq)[None, ...], background=background)[0]

    # --------------------------------------------------------- 序列标识图数据
    def sequence_logo_heights(self, pseudocount: float = 0.8, mode: str = "bits") -> np.ndarray:
        """返回 (4, L) 的字母高度，可直接用于绘制 sequence logo。

        mode="bits" 时高度 = 2 - H(p)，并按每列概率加权（信息量版 logo）。
        mode="prob" 时高度 = p。
        """
        p = self.ppm(pseudocount=pseudocount)
        if mode == "prob":
            return p
        entropy = -(p * np.log2(np.clip(p, 1e-12, None))).sum(axis=0)
        info = np.clip(2.0 - entropy, 0.0, None)
        return p * info.reshape(1, -1)

    # ------------------------------------------------------------- 序列化
    def to_dict(self) -> dict:
        return {
            "matrix_id": self.matrix_id,
            "name": self.name,
            "counts": self.counts.tolist(),
            "family": self.family,
            "collection": self.collection,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PWM":
        return cls(
            matrix_id=d["matrix_id"],
            name=d.get("name", d["matrix_id"]),
            counts=np.asarray(d["counts"], dtype=np.float64),
            family=d.get("family", ""),
            collection=d.get("collection", ""),
            source=d.get("source", ""),
        )

    @classmethod
    def from_jaspar_pfm(cls, payload: dict) -> "PWM":
        """从 JASPAR REST 返回的单个基序 JSON 构造 PWM。"""
        pfm = payload.get("pfm")
        if not isinstance(pfm, dict):
            raise ValueError("JASPAR 响应缺少 pfm 字段")
        counts = np.array([pfm.get(b, []) for b in BASES], dtype=np.float64)
        if counts.shape[0] != 4:
            raise ValueError("pfm 必须包含 A/C/G/T 四行")
        family = payload.get("family") or []
        return cls(
            matrix_id=payload.get("matrix_id", "unknown"),
            name=payload.get("name", payload.get("matrix_id", "unknown")),
            counts=counts,
            family=family[0] if family else "",
            collection=payload.get("collection", ""),
            source="JASPAR",
        )

    def reverse_complement(self) -> "PWM":
        """反向互补基序（转录因子常以二聚体形式结合两个方向）。"""
        return PWM(
            matrix_id=self.matrix_id + "_rc",
            name=self.name + " (revcomp)",
            counts=self.counts[::-1, ::-1].copy(),
            family=self.family,
            collection=self.collection,
            source=self.source,
        )


@dataclass
class MotifHit:
    """一次基序命中。"""

    motif_id: str
    motif_name: str
    start: int
    end: int
    score: float
    strand: str
    pvalue: float = 1.0

    def to_dict(self) -> dict:
        return {
            "motif_id": self.motif_id,
            "motif_name": self.motif_name,
            "start": self.start,
            "end": self.end,
            "score": round(float(self.score), 4),
            "strand": self.strand,
            "pvalue": float(f"{self.pvalue:.3g}"),
        }


def null_score_distribution(
    motif: PWM,
    seq_len: int = 200,
    n_seqs: int = 400,
    background: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """用背景模型采样随机序列，得到基序得分的经验零分布。

    用于把原始得分转换成 p 值。序列越长、采样越多，尾部分辨率越高。
    """
    bg = DEFAULT_BACKGROUND if background is None else np.asarray(background, dtype=np.float64)
    rng = np.random.default_rng(seed)
    # 采样 (n_seqs, seq_len) 的碱基索引
    idx = rng.choice(4, size=(n_seqs, seq_len), p=bg / bg.sum())
    onehot = np.zeros((n_seqs, seq_len, 4), dtype=np.float32)
    onehot[np.arange(n_seqs)[:, None], np.arange(seq_len)[None, :], idx] = 1.0
    scores = motif.score_windows(onehot, background=bg)
    return np.sort(scores.ravel())


def empirical_pvalue(score: np.ndarray | float, null_sorted: np.ndarray) -> np.ndarray:
    """由零分布计算经验 p 值：(#null >= score + 1) / (n + 1)。"""
    null_sorted = np.asarray(null_sorted)
    s = np.atleast_1d(np.asarray(score, dtype=np.float64))
    n = null_sorted.size
    counts = n - np.searchsorted(null_sorted, s, side="left")
    return (counts + 1.0) / (n + 1.0)
