"""合成数据生成：把已知 PWM 基序植入随机背景序列，且**记录逐碱基真值**。

为什么需要合成数据
------------------
真实基因组数据没有"哪个碱基真正重要"的标准答案，因此无法定量评估归因方法。
本模块通过受控植入来制造真值：先按背景分布采样随机序列，再把选定的基序
按 PWM 采样后写入指定窗口。写入的窗口位置即为"真值重要区域"。

真值约定
--------
`truth` 是与序列等长的 (L,) 数组，取值 0/1：
- 1 表示该碱基属于某个植入基序；
- 0 表示背景碱基。
多次叠加植入时取"或"（并集），重叠区域仍记为 1。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..motifs.pwm import PWM, DEFAULT_BACKGROUND
from ..seq import BASES, encode

#: 每条序列最多尝试植入的次数上限，防止宽度不足时死循环。
_MAX_ATTEMPTS_PER_SEQ = 200


@dataclass
class SyntheticDataset:
    """一次合成任务的全部产物。

    Attributes
    ----------
    sequences : 长度 N 的字符串列表。
    onehot : (N, L, 4) 的 one-hot 张量。
    truth : (N, L) 的 0/1 真值掩码。
    labels : (N,) 的标签，1 表示"包含基序"，0 表示纯背景。
    planted : 每条序列的植入记录，用于解释与统计。
    """

    sequences: list[str]
    onehot: np.ndarray
    truth: np.ndarray
    labels: np.ndarray
    planted: list[list[dict]] = field(default_factory=list)
    background: np.ndarray = field(default_factory=lambda: DEFAULT_BACKGROUND.copy())
    seed: int = 0

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def seq_len(self) -> int:
        return int(self.onehot.shape[1])

    def summary(self) -> dict:
        n_pos = int((self.truth.sum(axis=1) > 0).sum())
        return {
            "n_sequences": len(self),
            "seq_len": self.seq_len,
            "n_positive": n_pos,
            "n_negative": len(self) - n_pos,
            "truth_fraction": float(self.truth.mean()),
            "n_planted": int(sum(len(p) for p in self.planted)),
        }


def _sample_background(rng: np.random.Generator, length: int, background: np.ndarray) -> str:
    idx = rng.choice(4, size=length, p=background / background.sum())
    return "".join(BASES[i] for i in idx)


def _sample_motif_instance(rng: np.random.Generator, motif: PWM, pseudocount: float = 0.8) -> str:
    """按 PWM 的概率分布采样一条具体的基序实例。

    注意：采样而非直接取一致序列，可避免模型"只学会看一致序列"的假象。
    """
    ppm = motif.ppm(pseudocount=pseudocount)
    out = []
    for pos in range(motif.length):
        p = ppm[:, pos]
        p = p / p.sum()
        out.append(BASES[int(rng.choice(4, p=p))])
    return "".join(out)


def make_synthetic(
    motifs: list[PWM],
    n_sequences: int = 2000,
    seq_len: int = 200,
    motifs_per_seq: tuple[int, int] = (1, 3),
    negative_fraction: float = 0.25,
    background: np.ndarray | None = None,
    reverse_complement_prob: float = 0.5,
    allow_overlap: bool = True,
    seed: int = 0,
) -> SyntheticDataset:
    """生成带真值的合成数据集。

    Parameters
    ----------
    motifs
        用于植入的基序列表。
    n_sequences
        序列总条数（含负样本）。
    seq_len
        每条序列长度。
    motifs_per_seq
        正样本中每条序列植入的基序个数区间 [lo, hi]（含端点）。
    negative_fraction
        完全不含基序的负样本比例，用于给模型提供判别任务。
    background
        背景碱基分布；默认均匀。
    reverse_complement_prob
        每个植入实例以该概率取反向互补（更贴近真实的双链结合）。
    allow_overlap
        是否允许基序之间重叠；False 时强制互不重叠。
    seed
        随机种子，保证可复现。
    """
    if not motifs:
        raise ValueError("motifs 不能为空")
    bg = (DEFAULT_BACKGROUND if background is None else np.asarray(background, dtype=np.float64))
    bg = bg / bg.sum()
    rng = np.random.default_rng(seed)

    lo, hi = motifs_per_seq
    if lo < 1 or hi < lo:
        raise ValueError(f"motifs_per_seq 非法: {motifs_per_seq}")

    seqs: list[str] = []
    truths: list[np.ndarray] = []
    labels: list[int] = []
    planted_all: list[list[dict]] = []

    n_negative = int(round(n_sequences * negative_fraction))
    n_positive = n_sequences - n_negative

    for i in range(n_sequences):
        seq = _sample_background(rng, seq_len, bg)
        truth = np.zeros(seq_len, dtype=np.int8)
        planted: list[dict] = []

        if i < n_positive:
            n_plant = int(rng.integers(lo, hi + 1))
            occupied: list[tuple[int, int]] = []
            for _ in range(n_plant):
                motif = motifs[int(rng.integers(0, len(motifs)))]
                mlen = motif.length
                if mlen > seq_len:
                    continue
                placed = False
                for _attempt in range(_MAX_ATTEMPTS_PER_SEQ):
                    start = int(rng.integers(0, seq_len - mlen + 1))
                    end = start + mlen
                    if not allow_overlap and any(not (end <= s or start >= e) for s, e in occupied):
                        continue
                    occupied.append((start, end))
                    placed = True
                    break
                if not placed:
                    continue

                instance = _sample_motif_instance(rng, motif)
                strand = "+"
                if rng.random() < reverse_complement_prob:
                    from ..seq import reverse_complement as _rc

                    instance = _rc(instance)
                    strand = "-"

                seq = seq[:start] + instance + seq[end:]
                truth[start:end] = 1
                planted.append(
                    {
                        "motif_id": motif.matrix_id,
                        "motif_name": motif.name,
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "instance": instance,
                    }
                )

        # 全部植入完成后，回填每条记录"实际留在序列里的片段"。
        # 允许重叠时，后植入的基序会覆盖先前的部分碱基，此时原始 instance
        # 与最终序列不再一致；记录 written 片段可让下游（测试、报告、
        # 可视化）始终拿到与序列相符的碱基串。
        for p in planted:
            p["written"] = seq[p["start"] : p["end"]]
            p["intact"] = p["written"] == p["instance"]

        seqs.append(seq)
        truths.append(truth)
        labels.append(1 if planted else 0)
        planted_all.append(planted)

    onehot = np.stack([encode(s) for s in seqs], axis=0).astype(np.float32)
    return SyntheticDataset(
        sequences=seqs,
        onehot=onehot,
        truth=np.stack(truths, axis=0).astype(np.int8),
        labels=np.asarray(labels, dtype=np.int8),
        planted=planted_all,
        background=bg,
        seed=seed,
    )


def make_heldout_motifs(
    motifs: list[PWM],
    n_sequences: int = 400,
    seq_len: int = 200,
    background: np.ndarray | None = None,
    seed: int = 99,
) -> SyntheticDataset:
    """生成"只含单条已知基序"的集合，用于归因回收率评估。

    每条序列只植入**一条**基序、且位置均匀随机，这样真值区域唯一，
    position-level AUC 才有清晰含义。
    """
    bg = (DEFAULT_BACKGROUND if background is None else np.asarray(background, dtype=np.float64))
    bg = bg / bg.sum()
    rng = np.random.default_rng(seed)

    seqs, truths, planted_all = [], [], []
    for i in range(n_sequences):
        motif = motifs[i % len(motifs)]
        mlen = motif.length
        if mlen > seq_len:
            raise ValueError(f"基序 {motif.matrix_id} 长度 {mlen} 超过序列长度 {seq_len}")
        seq = _sample_background(rng, seq_len, bg)
        start = int(rng.integers(0, seq_len - mlen + 1))
        end = start + mlen
        instance = _sample_motif_instance(rng, motif)
        strand = "+"
        if rng.random() < 0.5:
            from ..seq import reverse_complement as _rc

            instance = _rc(instance)
            strand = "-"
        seq = seq[:start] + instance + seq[end:]
        truth = np.zeros(seq_len, dtype=np.int8)
        truth[start:end] = 1
        seqs.append(seq)
        truths.append(truth)
        planted_all.append(
            [
                {
                    "motif_id": motif.matrix_id,
                    "motif_name": motif.name,
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "instance": instance,
                }
            ]
        )

    onehot = np.stack([encode(s) for s in seqs], axis=0).astype(np.float32)
    return SyntheticDataset(
        sequences=seqs,
        onehot=onehot,
        truth=np.stack(truths, axis=0).astype(np.int8),
        labels=np.ones(len(seqs), dtype=np.int8),
        planted=planted_all,
        background=bg,
        seed=seed,
    )
