"""motifscope.evaluate —— 忠实度、回收率与稳定性指标。"""

from __future__ import annotations

from .faithfulness import (
    aopc,
    deletion_curve,
    evaluate_attribution,
    insertion_curve,
    method_agreement,
    motif_topk_hit,
    peak_inside_truth,
    position_auc,
    rank_stability,
    recovery_report,
    sufficiency,
)
from .scan import read_fasta, scan_many, scan_sequence, write_bed

__all__ = [
    "aopc",
    "deletion_curve",
    "evaluate_attribution",
    "insertion_curve",
    "method_agreement",
    "motif_topk_hit",
    "peak_inside_truth",
    "position_auc",
    "rank_stability",
    "read_fasta",
    "recovery_report",
    "scan_many",
    "scan_sequence",
    "sufficiency",
    "write_bed",
]
