"""PWM 基序模型测试。"""

from __future__ import annotations

import numpy as np
import pytest

from motifscope.motifs.pwm import (
    PWM,
    empirical_pvalue,
    gc_background,
    null_score_distribution,
)


def make_motif(counts: list[list[float]] | None = None) -> PWM:
    if counts is None:
        # 强 A 偏好基序：4 行 (A/C/G/T) × 4 列
        counts = [
            [10, 0, 0, 10],
            [0, 10, 0, 0],
            [0, 0, 10, 0],
            [0, 0, 0, 0.001],
        ]
    return PWM(matrix_id="TEST.1", name="TestMotif", counts=np.array(counts, dtype=float))


def test_pwm_shape():
    m = make_motif()
    assert m.length == 4
    assert m.counts.shape == (4, 4)


def test_pwm_rejects_bad_shape():
    with pytest.raises(ValueError, match="形状"):
        PWM(matrix_id="X", name="X", counts=np.ones((3, 5)))


def test_pwm_rejects_negative():
    with pytest.raises(ValueError, match="负值"):
        PWM(matrix_id="X", name="X", counts=-np.ones((4, 3)))


def test_pwm_rejects_all_zero():
    with pytest.raises(ValueError, match="全为 0"):
        PWM(matrix_id="X", name="X", counts=np.zeros((4, 3)))


def test_pwm_rejects_too_short():
    with pytest.raises(ValueError, match="至少为 2"):
        PWM(matrix_id="X", name="X", counts=np.ones((4, 1)))


def test_ppm_sums_to_one():
    m = make_motif()
    p = m.ppm()
    assert np.allclose(p.sum(axis=0), 1.0)


def test_pwm_log_odds_are_negative_for_background():
    """背景频率的碱基，其对数几率应为负（p < bg）。"""
    m = make_motif()
    pwm = m.pwm()
    # 第 0 列强偏好 A，则 C/G/T 的对数几率应为负
    assert pwm[1, 0] < 0
    assert pwm[2, 0] < 0
    assert pwm[3, 0] < 0
    assert pwm[0, 0] > 0


def test_pwm_scores_consensus_highest():
    """一致序列应得到比任意随机序列更高的分数。"""
    m = make_motif()
    cons = m.consensus
    best = m.scan(cons)[0]
    rng = np.random.default_rng(0)
    for _ in range(20):
        rand = "".join(rng.choice(list("ACGT"), size=4))
        assert m.scan(rand)[0] <= best + 1e-9


def test_scan_windows_count():
    m = make_motif()
    scores = m.scan("ACGTACGTAC")
    assert scores.shape == (10 - 4 + 1,)


def test_scan_rejects_short_sequence():
    m = PWM(matrix_id="X", name="X", counts=np.ones((4, 10)))
    with pytest.raises(ValueError, match="小于基序长度"):
        m.scan("ACG")


def test_reverse_complement_motif():
    m = make_motif()
    rc = m.reverse_complement()
    assert rc.length == m.length
    assert rc.matrix_id == m.matrix_id + "_rc"
    # 反向互补的计数矩阵应翻转
    assert np.allclose(rc.counts, m.counts[::-1, ::-1])


def test_information_content_positive():
    m = make_motif()
    assert m.information_content() > 0


def test_uniform_motif_has_low_information():
    uniform = PWM(matrix_id="U", name="U", counts=np.full((4, 5), 10.0))
    assert uniform.information_content() < 0.5


def test_sequence_logo_heights_shape():
    m = make_motif()
    h = m.sequence_logo_heights()
    assert h.shape == (4, m.length)
    assert (h >= 0).all()


def test_sequence_logo_prob_mode_sums_to_one():
    m = make_motif()
    h = m.sequence_logo_heights(mode="prob")
    assert np.allclose(h.sum(axis=0), 1.0)


def test_jaspar_roundtrip():
    payload = {
        "matrix_id": "MA0001.1",
        "name": "TestTF",
        "collection": "CORE",
        "family": ["Test family"],
        "pfm": {"A": [1, 2, 3], "C": [4, 5, 6], "G": [7, 8, 9], "T": [10, 11, 12]},
    }
    m = PWM.from_jaspar_pfm(payload)
    assert m.matrix_id == "MA0001.1"
    assert m.name == "TestTF"
    assert m.family == "Test family"
    assert m.length == 3


def test_jaspar_missing_pfm_raises():
    with pytest.raises(ValueError, match="pfm"):
        PWM.from_jaspar_pfm({"matrix_id": "X"})


def test_to_dict_from_dict_roundtrip():
    m = make_motif()
    m2 = PWM.from_dict(m.to_dict())
    assert m2.matrix_id == m.matrix_id
    assert np.allclose(m2.counts, m.counts)


def test_gc_background():
    bg = gc_background(0.6)
    assert np.isclose(bg.sum(), 1.0)
    assert np.isclose(bg[1] + bg[2], 0.6)


def test_null_distribution_is_sorted():
    m = make_motif()
    null = null_score_distribution(m, seq_len=100, n_seqs=50, seed=0)
    assert np.all(np.diff(null) >= 0)


def test_empirical_pvalue_bounds():
    null = np.arange(100.0)
    p_hi = empirical_pvalue(1000.0, null)
    p_lo = empirical_pvalue(-1000.0, null)
    assert p_hi < 0.01
    assert p_lo > 0.99
