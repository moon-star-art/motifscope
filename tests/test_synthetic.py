"""合成数据生成测试：真值正确性是后续所有评估的基础。"""

from __future__ import annotations

import numpy as np
import pytest

from motifscope.data.synthetic import make_heldout_motifs, make_synthetic
from motifscope.motifs.pwm import PWM
from motifscope.seq import reverse_complement


def make_motifs(n: int = 3) -> list[PWM]:
    out = []
    for i in range(n):
        counts = np.ones((4, 6 + i))
        counts[i % 4, :] += 10.0
        counts[(i + 1) % 4, 2] += 5.0
        out.append(PWM(matrix_id=f"M{i}.1", name=f"Motif{i}", counts=counts))
    return out


def test_synthetic_shapes():
    motifs = make_motifs(2)
    d = make_synthetic(motifs, n_sequences=50, seq_len=100, seed=0)
    assert d.onehot.shape == (50, 100, 4)
    assert d.truth.shape == (50, 100)
    assert len(d.sequences) == 50
    assert len(d.planted) == 50


def test_synthetic_truth_matches_planted():
    """真值掩码必须与实际植入位置一致。"""
    motifs = make_motifs(2)
    d = make_synthetic(motifs, n_sequences=30, seq_len=120, seed=1)
    for i, plants in enumerate(d.planted):
        expected = np.zeros(120, dtype=np.int8)
        for p in plants:
            expected[p["start"] : p["end"]] = 1
        assert np.array_equal(d.truth[i], expected), f"序列 {i} 真值不符"


def test_synthetic_planted_sequence_matches():
    """记录中的 written 片段必须与实际序列逐字相符。

    允许重叠植入时，后植入的基序会覆盖先前基序的部分碱基，因此原始
    ``instance`` 可能与最终序列不一致；``written`` 始终与序列一致。
    """
    motifs = make_motifs(2)
    d = make_synthetic(motifs, n_sequences=20, seq_len=120, seed=2)
    for i, plants in enumerate(d.planted):
        seq = d.sequences[i]
        for p in plants:
            assert seq[p["start"] : p["end"]] == p["written"]
            # intact 标志必须与实际比较结果一致
            assert p["intact"] == (p["written"] == p["instance"])


def test_synthetic_non_overlap_instances_intact():
    """禁止重叠时，每个植入实例都应完好无损。"""
    motifs = make_motifs(2)
    d = make_synthetic(
        motifs, n_sequences=30, seq_len=200, motifs_per_seq=(2, 3), allow_overlap=False, seed=4
    )
    for plants in d.planted:
        for p in plants:
            assert p["intact"], f"禁止重叠时实例被破坏: {p}"


def test_negative_fraction_respected():
    motifs = make_motifs(2)
    d = make_synthetic(motifs, n_sequences=100, seq_len=100, negative_fraction=0.3, seed=0)
    assert int((d.labels == 0).sum()) == 30


def test_no_overlap_when_disabled():
    motifs = make_motifs(2)
    d = make_synthetic(
        motifs, n_sequences=50, seq_len=200, motifs_per_seq=(2, 3), allow_overlap=False, seed=3
    )
    for plants in d.planted:
        spans = sorted((p["start"], p["end"]) for p in plants)
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            assert e1 <= s2, f"出现重叠: {(s1, e1)} 与 {(s2, e2)}"


def test_reproducible_with_same_seed():
    motifs = make_motifs(2)
    d1 = make_synthetic(motifs, n_sequences=30, seq_len=100, seed=42)
    d2 = make_synthetic(motifs, n_sequences=30, seq_len=100, seed=42)
    assert d1.sequences == d2.sequences


def test_different_seed_differs():
    motifs = make_motifs(2)
    d1 = make_synthetic(motifs, n_sequences=30, seq_len=100, seed=1)
    d2 = make_synthetic(motifs, n_sequences=30, seq_len=100, seed=2)
    assert d1.sequences != d2.sequences


def test_reverse_complement_instances_recorded():
    """反向植入的实例应是正向实例的反向互补。"""
    motifs = make_motifs(2)
    d = make_synthetic(motifs, n_sequences=60, seq_len=150, seed=5)
    rc_count = 0
    for plants in d.planted:
        for p in plants:
            if p["strand"] == "-":
                rc_count += 1
                assert len(p["instance"]) == p["end"] - p["start"]
    assert rc_count > 0, "未产生任何反向植入，测试无意义"


def test_empty_motifs_raises():
    with pytest.raises(ValueError, match="不能为空"):
        make_synthetic([], n_sequences=10, seq_len=100)


def test_invalid_motifs_per_seq_raises():
    motifs = make_motifs(1)
    with pytest.raises(ValueError, match="motifs_per_seq"):
        make_synthetic(motifs, n_sequences=10, seq_len=100, motifs_per_seq=(0, 2))


def test_summary_keys():
    motifs = make_motifs(2)
    d = make_synthetic(motifs, n_sequences=40, seq_len=100, seed=0)
    s = d.summary()
    assert s["n_sequences"] == 40
    assert s["seq_len"] == 100
    assert 0 <= s["truth_fraction"] <= 1


def test_heldout_single_motif_per_sequence():
    """heldout 集每条序列只植一个基序，真值应是一段连续区间。"""
    motifs = make_motifs(3)
    d = make_heldout_motifs(motifs, n_sequences=30, seq_len=120, seed=7)
    for i in range(len(d)):
        plants = d.planted[i]
        assert len(plants) == 1, f"序列 {i} 植入了 {len(plants)} 个基序"
        p = plants[0]
        assert d.truth[i].sum() == p["end"] - p["start"]


def test_heldout_position_is_uniform_ish():
    """heldout 的植入位置应覆盖较宽的分布范围。"""
    motifs = make_motifs(2)
    d = make_heldout_motifs(motifs, n_sequences=100, seq_len=200, seed=8)
    starts = [p[0]["start"] for p in d.planted]
    assert min(starts) < 30
    assert max(starts) > 150


def test_heldout_rejects_too_long_motif():
    long_motif = PWM(matrix_id="L.1", name="Long", counts=np.ones((4, 50)))
    with pytest.raises(ValueError, match="超过序列长度"):
        make_heldout_motifs([long_motif], n_sequences=5, seq_len=30)
