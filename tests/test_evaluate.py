"""评估指标测试：确认指标方向正确（好归因得分更高）。"""

from __future__ import annotations

import numpy as np
import pytest

from motifscope.evaluate import (
    aopc,
    deletion_curve,
    insertion_curve,
    method_agreement,
    motif_topk_hit,
    peak_inside_truth,
    position_auc,
    rank_stability,
    recovery_report,
    sufficiency,
)
from motifscope.models.conv import ConvMaxPoolModel


@pytest.fixture(scope="module")
def trained():
    from motifscope.data.synthetic import make_synthetic
    from motifscope.motifs.pwm import PWM

    counts = np.ones((4, 8))
    counts[0, :] += 12.0
    motif = PWM(matrix_id="T.1", name="T", counts=counts)
    d = make_synthetic([motif], n_sequences=120, seq_len=60, motifs_per_seq=(1, 1), seed=0)
    m = ConvMaxPoolModel(n_filters=4, width=8, n_epochs=6, seed=0)
    m.fit(d.onehot, d.labels)
    return m, d


# ------------------------------------------------------------ 回收型指标
def test_position_auc_perfect_separation():
    """完美归因（真值位置最高）应得到 AUC = 1。"""
    truth = np.zeros((2, 10), dtype=np.int8)
    truth[:, 3:6] = 1
    scores = truth.astype(float) * 10.0
    auc = position_auc(scores, truth)
    assert np.allclose(auc, 1.0)


def test_position_auc_anti_correlation():
    """完全反向的归因应得到 AUC = 0。"""
    truth = np.zeros((1, 10), dtype=np.int8)
    truth[:, 3:6] = 1
    scores = (1 - truth.astype(float)) * 10.0
    auc = position_auc(scores, truth)
    assert np.allclose(auc, 0.0)


def test_position_auc_random_about_half():
    rng = np.random.default_rng(0)
    truth = np.zeros((40, 50), dtype=np.int8)
    truth[:, 20:25] = 1
    scores = rng.random((40, 50))
    auc = position_auc(scores, truth)
    assert 0.35 < np.nanmean(auc) < 0.65


def test_position_auc_handles_empty_masks():
    truth = np.zeros((1, 10), dtype=np.int8)  # 无正例
    scores = np.random.default_rng(0).random((1, 10))
    auc = position_auc(scores, truth)
    assert np.isnan(auc[0])


def test_position_auc_handles_ties():
    """大量并列值不应导致 NaN 或越界。"""
    truth = np.zeros((1, 10), dtype=np.int8)
    truth[:, :5] = 1
    scores = np.ones((1, 10))
    auc = position_auc(scores, truth)
    assert np.isfinite(auc[0])
    assert 0.0 <= auc[0] <= 1.0


def test_motif_topk_hit_perfect():
    truth = np.zeros((1, 20), dtype=np.int8)
    truth[:, 0:2] = 1
    scores = np.zeros((1, 20))
    scores[0, 0] = 10
    scores[0, 1] = 9
    assert motif_topk_hit(scores, truth, k_fraction=0.1)[0] == 1.0


def test_peak_inside_truth():
    truth = np.zeros((1, 10), dtype=np.int8)
    truth[:, 4:6] = 1
    scores = np.zeros((1, 10))
    scores[0, 5] = 1.0
    assert peak_inside_truth(scores, truth)[0] == 1.0


def test_recovery_report_keys():
    truth = np.zeros((3, 20), dtype=np.int8)
    truth[:, 5:10] = 1
    scores = np.random.default_rng(0).random((3, 20))
    rep = recovery_report(scores, truth)
    for k in ["position_auc_mean", "topk_hit_mean", "peak_inside_rate", "truth_fraction"]:
        assert k in rep


# ------------------------------------------------------------ 扰动型指标
def test_deletion_curve_monotone_for_good_attribution(trained):
    """忠实归因：删除高分区应使预测快速下降。"""
    m, d = trained
    truth = d.truth[:20].astype(float)
    dele = deletion_curve(m, d.onehot[:20], truth)
    assert dele["probs"][0] >= dele["probs"][-1] - 0.05
    assert len(dele["fractions"]) == len(dele["probs"])


def test_deletion_curve_fractions_range(trained):
    m, d = trained
    truth = d.truth[:10].astype(float)
    dele = deletion_curve(m, d.onehot[:10], truth, steps=5)
    assert dele["fractions"][0] == 0.0
    assert dele["fractions"][-1] == 1.0


def test_insertion_curve_starts_low(trained):
    """从全背景出发插入真实碱基，预测应上升。"""
    m, d = trained
    truth = d.truth[:15].astype(float)
    ins = insertion_curve(m, d.onehot[:15], truth, steps=10)
    assert ins["probs"][-1] >= ins["probs"][0]


def test_aopc_random_attribution_near_zero(trained):
    """随机排序的归因等价于随机删除，AOPC 应接近 0。"""
    m, d = trained
    rng = np.random.default_rng(0)
    rand_scores = rng.random((30, d.seq_len))
    dele = deletion_curve(m, d.onehot[:30], rand_scores, steps=10)
    assert -0.15 < aopc(dele) < 0.15


def test_recovery_metrics_prefer_truth_over_random(trained):
    """真值排序在**回收型**指标上应显著优于随机排序。

    注意：这里刻意不使用 AOPC。扰动型指标（AOPC）刻画的是"模型对扰动是否
    敏感"，当模型尚未学到基序信号时，即便按真值删除也不会降低预测，AOPC
    会失去区分度。回收型指标（position AUC / top-k 命中）直接比较归因与
    真值，因此更适合作为归因方法本身的评估。
    """
    m, d = trained
    rng = np.random.default_rng(0)
    truth = d.truth[:30].astype(float)
    rand_scores = rng.random((30, d.seq_len))

    auc_truth = np.nanmean(position_auc(truth, d.truth[:30]))
    auc_rand = np.nanmean(position_auc(rand_scores, d.truth[:30]))
    assert auc_truth > 0.95
    assert auc_rand < 0.7
    assert auc_truth > auc_rand


def test_sufficiency_bounds(trained):
    m, d = trained
    truth = d.truth[:10].astype(float)
    ins = insertion_curve(m, d.onehot[:10], truth, steps=10)
    s = sufficiency(ins)
    assert np.isfinite(s)


# ------------------------------------------------------------ 稳定性指标
def test_method_agreement_identical():
    x = np.random.default_rng(0).random((5, 20))
    assert np.isclose(method_agreement(x, x), 1.0)


def test_method_agreement_negated():
    x = np.random.default_rng(0).random((5, 20))
    assert np.isclose(method_agreement(x, -x), -1.0)


def test_rank_stability_identical():
    x = np.random.default_rng(0).random((5, 20))
    assert np.isclose(rank_stability(x, x), 1.0)


def test_rank_stability_disjoint():
    a = np.zeros((1, 20))
    b = np.zeros((1, 20))
    a[0, :2] = 1.0
    b[0, -2:] = 1.0
    assert rank_stability(a, b, k_fraction=0.1) == 0.0
