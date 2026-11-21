"""motifscope.data —— 合成数据与真实数据加载。"""

from __future__ import annotations

from .splice import (
    SPLICE_CLASSES,
    SPLICE_LABELS,
    SpliceDataset,
    load_splice,
    parse_splice,
)
from .synthetic import SyntheticDataset, make_heldout_motifs, make_synthetic

__all__ = [
    "SPLICE_CLASSES",
    "SPLICE_LABELS",
    "SpliceDataset",
    "SyntheticDataset",
    "load_splice",
    "make_heldout_motifs",
    "make_synthetic",
    "parse_splice",
]
