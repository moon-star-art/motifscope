"""真实数据加载器：UCI 剪接位点数据集。

数据说明
--------
UCI "Molecular Biology (Splice-junction Gene Sequences)"（ID 69）。
- 3190 条灵长类 DNA 序列，每条 **60 nt**；
- 三类：EI（外显子→内含子，供体位点）、IE（内含子→外显子，受体位点）、N（两者都不是）；
- 文件格式：`类别, 序列名, 序列` 三列逗号分隔。

生物学意义
----------
剪接位点是 mRNA 成熟过程中的关键调控元件。EI/IE 位点附近存在保守的
供体（GT）与受体（AG）二核苷酸以及分支点等弱信号，是经典的序列分类任务，
也是我们检验"模型是否学到了已知生物学信号"的真实数据场景。
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

UCI_SPLICE_URL = (
    "https://archive.ics.uci.edu/static/public/69/"
    "molecular+biology+splice+junction+gene+sequences.zip"
)

#: 类别到整数的映射（顺序固定，便于复现）。
SPLICE_CLASSES = {"EI": 0, "IE": 1, "N": 2}
SPLICE_LABELS = {v: k for k, v in SPLICE_CLASSES.items()}


@dataclass
class SpliceDataset:
    """剪接位点数据集。"""

    sequences: list[str]
    labels: np.ndarray  # 0=EI, 1=IE, 2=N
    names: list[str]

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def seq_len(self) -> int:
        return len(self.sequences[0])

    @property
    def label_names(self) -> list[str]:
        return [SPLICE_LABELS[int(i)] for i in sorted(set(self.labels.tolist()))]

    def class_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for lab in self.labels:
            key = SPLICE_LABELS[int(lab)]
            out[key] = out.get(key, 0) + 1
        return out

    def summary(self) -> dict:
        return {
            "n_sequences": len(self),
            "seq_len": self.seq_len,
            "class_counts": self.class_counts(),
        }

    def subset(self, n: int, seed: int = 0) -> "SpliceDataset":
        """分层抽样出一个子集（小样本实验用）。"""
        rng = np.random.default_rng(seed)
        idx = []
        for lab in sorted(set(self.labels.tolist())):
            pool = np.flatnonzero(self.labels == lab)
            take = max(1, int(round(n * len(pool) / len(self))))
            take = min(take, len(pool))
            idx.extend(rng.choice(pool, size=take, replace=False).tolist())
        idx_arr = np.array(sorted(idx))
        return SpliceDataset(
            sequences=[self.sequences[i] for i in idx_arr],
            labels=self.labels[idx_arr],
            names=[self.names[i] for i in idx_arr],
        )


def parse_splice(raw: str) -> SpliceDataset:
    """解析 splice.data 文本内容。"""
    seqs, labels, names = [], [], []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip().rstrip(",") for p in line.split(",")]
        parts = [p for p in parts if p]
        if len(parts) < 3:
            log.warning("第 %d 行字段不足，已跳过: %r", lineno, line[:60])
            continue
        cls, name, seq = parts[0], parts[1], parts[2]
        if cls not in SPLICE_CLASSES:
            continue
        seq = seq.upper()
        if any(ch not in "ACGT" for ch in seq):
            continue
        seqs.append(seq)
        labels.append(SPLICE_CLASSES[cls])
        names.append(name)
    if not seqs:
        raise ValueError("未解析出任何序列")
    return SpliceDataset(sequences=seqs, labels=np.asarray(labels, dtype=np.int64), names=names)


def load_splice(
    cache_dir: str | Path | None = None,
    allow_network: bool = True,
    timeout: float = 60.0,
) -> SpliceDataset:
    """加载 UCI 剪接位点数据集，带磁盘缓存与离线降级。"""
    cache_dir = Path(cache_dir or Path.cwd() / "data_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "uci_splice.data"

    if cache_file.exists():
        log.info("从缓存读取剪接位点数据: %s", cache_file)
        return parse_splice(cache_file.read_text(encoding="utf-8"))

    if not allow_network:
        raise FileNotFoundError(f"缓存不存在且已禁用网络: {cache_file}")

    log.info("下载 UCI 剪接位点数据 ...")
    with urllib.request.urlopen(UCI_SPLICE_URL, timeout=timeout) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        raw = zf.read("splice.data").decode("latin-1")

    cache_file.write_text(raw, encoding="utf-8")
    log.info("已缓存到 %s", cache_file)
    return parse_splice(raw)
