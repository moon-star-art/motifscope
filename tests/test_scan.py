"""基序扫描与精确 p 值测试。"""

from __future__ import annotations

import numpy as np
import pytest

from motifscope.evaluate.scan import (
    _nms,
    read_fasta,
    scan_many,
    scan_sequence,
    write_bed,
)
from motifscope.motifs.exact_pvalue import exact_pvalues, exact_score_pmf
from motifscope.motifs.pwm import PWM, MotifHit, null_score_distribution


def make_motif() -> PWM:
    counts = np.array(
        [
            [40, 0, 5, 0, 0, 10],
            [5, 40, 5, 0, 40, 5],
            [5, 0, 5, 40, 0, 5],
            [10, 0, 25, 0, 0, 20],
        ],
        dtype=float,
    )
    return PWM(matrix_id="MA9999.1", name="TestTF", counts=counts)


# ------------------------------------------------------------ 精确 p 值
def test_exact_pmf_sums_to_one():
    m = make_motif()
    grid, pmf = exact_score_pmf(m)
    assert len(grid) == len(pmf)
    assert np.isclose(pmf.sum(), 1.0)


def test_exact_pmf_mean_matches_expectation():
    """卷积分布的均值应等于各列期望得分之和。"""
    m = make_motif()
    grid, pmf = exact_score_pmf(m, n_bins=8000)
    mean_conv = float(np.sum(grid * pmf))
    mean_theory = float((m.pwm() * 0.25).sum())
    assert abs(mean_conv - mean_theory) < 0.05


def test_exact_pvalue_high_score_is_small():
    m = make_motif()
    best = float(m.pwm().max(axis=0).sum())
    p = exact_pvalues([best], m)[0]
    assert 0 < p < 0.05


def test_exact_pvalue_has_positive_floor():
    """极端高分不应给出 p=0，否则下游取对数会出错。"""
    m = make_motif()
    very_high = float(m.pwm().max(axis=0).sum()) + 100
    assert exact_pvalues([very_high], m)[0] > 0


def test_exact_pvalue_low_score_is_large():
    m = make_motif()
    worst = float(m.pwm().min(axis=0).sum())
    assert exact_pvalues([worst], m)[0] > 0.9


def test_exact_pvalue_monotone_decreasing():
    """得分越高，p 值应越小。"""
    m = make_motif()
    scores = np.linspace(m.pwm().min(), m.pwm().max(axis=0).sum(), 15)
    p = exact_pvalues(scores, m)
    assert np.all(np.diff(p) <= 1e-9)


def test_exact_pvalue_agrees_with_empirical():
    """精确 p 值应与大样本经验 p 值大体一致。"""
    m = make_motif()
    null = null_score_distribution(m, seq_len=200, n_seqs=2000, seed=0)
    probe = np.array([np.percentile(null, 99), np.median(null)])
    pe = exact_pvalues(probe, m)
    from motifscope.motifs.pwm import empirical_pvalue

    em = empirical_pvalue(probe, null)
    # 允许 3 倍误差（经验分布尾部分辨率有限）
    ratio = np.maximum(pe, 1e-9) / np.maximum(em, 1e-9)
    assert np.all((ratio > 0.2) & (ratio < 5.0)), f"{pe} vs {em}"


# ------------------------------------------------------------ NMS
def test_nms_merges_identical():
    h1 = MotifHit("M1", "A", 10, 20, score=10.0, strand="+")
    h2 = MotifHit("M1", "A", 11, 21, score=9.0, strand="+")
    kept = _nms([h1, h2])
    assert len(kept) == 1
    assert kept[0].score == 10.0


def test_nms_keeps_disjoint():
    h1 = MotifHit("M1", "A", 10, 20, score=10.0, strand="+")
    h2 = MotifHit("M1", "A", 100, 110, score=9.0, strand="+")
    assert len(_nms([h1, h2])) == 2


def test_nms_merges_tandem():
    """紧邻的串联重复应被合并。"""
    h1 = MotifHit("M1", "A", 10, 20, score=10.0, strand="+")
    h2 = MotifHit("M1", "A", 20, 30, score=9.5, strand="+")
    assert len(_nms([h1, h2])) == 1


def test_nms_empty():
    assert _nms([]) == []


# ------------------------------------------------------------ 扫描
def test_scan_finds_planted_site():
    m = make_motif()
    cons = m.consensus
    seq = "ACGT" * 10 + cons + "TGCA" * 10
    hits = scan_sequence(seq, [m], pvalue_threshold=0.05, apply_nms=True)
    assert len(hits) >= 1
    top = hits[0]
    start = len("ACGT") * 10
    assert abs(top.start - start) <= 2


def test_scan_no_hits_on_random_with_correction():
    """多重检验校正后，随机序列不应有显著命中。"""
    m = make_motif()
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), size=200))
    hits = scan_sequence(
        seq, [m], pvalue_threshold=0.01, correct_multiple_testing=True
    )
    assert len(hits) == 0


def test_scan_without_correction_has_more_hits():
    """关闭校正后命中年数应不少于开启校正。"""
    m = make_motif()
    cons = m.consensus
    seq = "ACGT" * 5 + cons + "ACGT" * 5
    strict = scan_sequence(seq, [m], pvalue_threshold=0.01, correct_multiple_testing=True)
    loose = scan_sequence(
        seq, [m], pvalue_threshold=0.01, correct_multiple_testing=False, apply_nms=False
    )
    assert len(loose) >= len(strict)


def test_scan_skips_too_long_motif():
    m = PWM(matrix_id="L", name="L", counts=np.ones((4, 500)))
    hits = scan_sequence("ACGT" * 10, [m], pvalue_threshold=0.05)
    assert hits == []


def test_scan_results_sorted_by_pvalue():
    m = make_motif()
    cons = m.consensus
    seq = "ACGT" * 5 + cons + "TGCA" * 5 + cons
    hits = scan_sequence(seq, [m], pvalue_threshold=0.5, apply_nms=False)
    if len(hits) > 1:
        ps = [h.pvalue for h in hits]
        assert ps == sorted(ps)


def test_scan_hit_fields_within_sequence():
    m = make_motif()
    cons = m.consensus
    seq = "ACGT" * 10 + cons + "TGCA" * 10
    for h in scan_sequence(seq, [m], pvalue_threshold=0.5):
        assert 0 <= h.start < len(seq)
        assert h.end <= len(seq)
        assert h.strand in {"+", "-"}


def test_scan_many_returns_all_keys():
    m = make_motif()
    seqs = {"a": "ACGT" * 30, "b": "TGCA" * 30}
    out = scan_many(seqs, [m], pvalue_threshold=0.5)
    assert set(out) == set(seqs)


# ------------------------------------------------------------ FASTA / BED
def test_read_fasta(tmp_dir):
    p = tmp_dir / "t.fa"
    p.write_text(">seq1 desc\nACGT\nACGT\n>seq2\nTTTT\n", encoding="utf-8")
    seqs = read_fasta(str(p))
    assert seqs["seq1"] == "ACGTACGT"
    assert seqs["seq2"] == "TTTT"


def test_read_fasta_empty_raises(tmp_dir):
    p = tmp_dir / "empty.fa"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="未从"):
        read_fasta(str(p))


def test_write_bed(tmp_dir):
    hits = {"chr1": [MotifHit("M1", "TF1", 10, 20, score=5.5, strand="+")]}
    out = tmp_dir / "h.bed"
    write_bed(hits, str(out))
    text = out.read_text(encoding="utf-8")
    assert "chr1" in text
    assert "TF1" in text
    assert text.startswith("#chrom")
