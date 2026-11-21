"""序列编码工具：one-hot、反向互补、k-mer 词汇表。"""

from __future__ import annotations

import numpy as np

BASES = "ACGT"
BASE_INDEX = {b: i for i, b in enumerate(BASES)}
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def encode(seq: str) -> np.ndarray:
    """把 DNA 字符串编码成 (L, 4) 的 one-hot 矩阵。

    非 ACGT 的字符（如 N）编码为全 0，表示"未知碱基"。
    """
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, ch in enumerate(seq.upper()):
        j = BASE_INDEX.get(ch)
        if j is not None:
            arr[i, j] = 1.0
    return arr


def encode_batch(seqs: list[str]) -> np.ndarray:
    """把字符串列表编码成 (N, L, 4) 的 one-hot 张量。"""
    if not seqs:
        raise ValueError("seqs 不能为空")
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"所有序列长度必须一致，收到 {sorted(lengths)}")
    return np.stack([encode(s) for s in seqs], axis=0)


def decode(onehot: np.ndarray) -> str:
    """把 (L, 4) one-hot 还原成字符串；全 0 位置记为 N。"""
    out = []
    for row in onehot:
        idx = int(np.argmax(row))
        out.append(BASES[idx] if row[idx] > 0.5 else "N")
    return "".join(out)


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def kmer_vocabulary(k: int) -> list[str]:
    """按字典序生成全部 4^k 个 k-mer。"""
    vocab = [""]
    for _ in range(k):
        vocab = [v + b for v in vocab for b in BASES]
    return vocab


def kmer_index(k: int) -> dict[str, int]:
    return {km: i for i, km in enumerate(kmer_vocabulary(k))}


def kmer_onehot(onehot: np.ndarray, k: int) -> np.ndarray:
    """把 (N, L, 4) 的 one-hot 转换成 (N, L-k+1, 4^k) 的 k-mer one-hot。

    使用矩阵乘法实现：每个窗口的 4^k 维指示向量等于窗口内 k 个碱基
    one-hot 的外积展开，可用逐位索引相乘得到。
    """
    n, length, _ = onehot.shape
    if length < k:
        raise ValueError(f"序列长度 {length} 小于 k={k}")
    n_win = length - k + 1
    # 逐位编码：把每个位置的 one-hot 变成索引
    idx = onehot.argmax(axis=2)  # (N, L)
    out = np.zeros((n, n_win, 4**k), dtype=np.float32)
    # 混合基数编码
    code = np.zeros((n, n_win), dtype=np.int64)
    for offset in range(k):
        code = code * 4 + idx[:, offset : offset + n_win]
    rows = np.repeat(np.arange(n), n_win)
    cols = np.tile(np.arange(n_win), n)
    out[rows, cols, code.ravel()] = 1.0
    return out
