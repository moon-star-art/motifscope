"""模型测试：重点是精确分解的**完整性公理**。

这是本项目最重要的不变量：
    Σ_i φ_i = f(x) - f(0)
若该式不成立，则该模型的"精确归因"名不副实，整个基准评测失去根基。
"""

from __future__ import annotations

import numpy as np
import pytest

from motifscope.data.synthetic import make_synthetic
from motifscope.models.conv import ConvMaxPoolModel
from motifscope.models.sequence import KmerLogisticModel, PWMScoringModel, build_model
from motifscope.motifs.pwm import PWM


def make_motifs(n: int = 3) -> list[PWM]:
    out = []
    for i in range(n):
        counts = np.ones((4, 6 + i))
        counts[i % 4, :] += 10.0
        counts[(i + 1) % 4, 2] += 5.0
        out.append(PWM(matrix_id=f"M{i}.1", name=f"Motif{i}", counts=counts))
    return out


@pytest.fixture(scope="module")
def dataset():
    motifs = make_motifs(3)
    train = make_synthetic(motifs, n_sequences=160, seq_len=60, motifs_per_seq=(1, 2), seed=0)
    return motifs, train


# ---------------------------------------------------------------- PWM 模型
def test_pwm_model_rejects_empty():
    with pytest.raises(ValueError, match="不能为空"):
        PWMScoringModel([])


def test_pwm_model_rejects_bad_pooling():
    with pytest.raises(ValueError, match="pooling"):
        PWMScoringModel(make_motifs(1), pooling="median")


def test_pwm_model_fit_predict(dataset):
    motifs, train = dataset
    m = PWMScoringModel(motifs, pooling="max").fit(train.onehot, train.labels)
    pred = m.predict(train.onehot)
    assert pred.shape == (len(train),)
    assert set(np.unique(pred)).issubset({0, 1})


def test_pwm_model_proba_bounds(dataset):
    motifs, train = dataset
    m = PWMScoringModel(motifs).fit(train.onehot, train.labels)
    p = m.predict_proba(train.onehot)
    assert p.shape == (len(train), 2)
    assert (p >= 0).all() and (p <= 1).all()
    assert np.allclose(p.sum(axis=1), 1.0)


def test_pwm_model_completeness_max_pooling(dataset):
    """max 池化模型：Σφ 必须精确等于 f(x) - f(0)。"""
    motifs, train = dataset
    m = PWMScoringModel(motifs, pooling="max").fit(train.onehot, train.labels)
    x = train.onehot[:20]
    phi = m.exact_attribution(x).sum(axis=1)
    f0 = float(m.decision_function(np.zeros_like(x[:1]))[0])
    assert np.allclose(phi, m.decision_function(x) - f0, atol=1e-8)


def test_pwm_model_completeness_sum_pooling(dataset):
    """sum 池化模型同样是精确线性分解。"""
    motifs, train = dataset
    m = PWMScoringModel(motifs, pooling="sum").fit(train.onehot, train.labels)
    x = train.onehot[:20]
    phi = m.exact_attribution(x).sum(axis=1)
    f0 = float(m.decision_function(np.zeros_like(x[:1]))[0])
    assert np.allclose(phi, m.decision_function(x) - f0, atol=1e-8)


def test_pwm_model_handles_unequal_motif_lengths(dataset):
    """不同长度的基序必须能共存（真实 JASPAR 库就是混合长度）。"""
    motifs, train = dataset
    lengths = {m.length for m in motifs}
    assert len(lengths) > 1
    m = PWMScoringModel(motifs).fit(train.onehot, train.labels)
    assert m.decision_function(train.onehot[:3]).shape == (3,)


def test_pwm_model_rejects_too_short_sequence(dataset):
    motifs, _ = dataset
    long_motif = PWM(matrix_id="L", name="L", counts=np.ones((4, 100)))
    m = PWMScoringModel([long_motif])
    with pytest.raises(ValueError, match="小于基序长度"):
        m.decision_function(np.zeros((1, 50, 4), dtype=np.float32))


# ---------------------------------------------------------------- k-mer 模型
def test_kmer_model_invalid_k():
    with pytest.raises(ValueError, match="k 必须"):
        KmerLogisticModel(k=0)


def test_kmer_model_fit_predict(dataset):
    _, train = dataset
    m = KmerLogisticModel(k=3, C=0.5).fit(train.onehot, train.labels)
    assert m.decision_function(train.onehot).shape == (len(train),)


def test_kmer_model_requires_fit():
    m = KmerLogisticModel(k=3)
    with pytest.raises(RuntimeError, match="尚未训练"):
        m.decision_function(np.zeros((1, 10, 4), dtype=np.float32))


def test_kmer_model_completeness(dataset):
    """k-mer 线性模型：均摊分解必须满足完整性公理。"""
    _, train = dataset
    m = KmerLogisticModel(k=3, C=0.5).fit(train.onehot, train.labels)
    x = train.onehot[:20]
    phi = m.exact_attribution(x).sum(axis=1)
    f0 = float(m.decision_function(np.zeros_like(x[:1]))[0])
    assert np.allclose(phi, m.decision_function(x) - f0, atol=1e-8)


def test_kmer_model_attribution_shape(dataset):
    _, train = dataset
    m = KmerLogisticModel(k=3).fit(train.onehot, train.labels)
    phi = m.exact_attribution(train.onehot[:5])
    assert phi.shape == (5, train.seq_len)


def test_kmer_linear_weights_shape(dataset):
    _, train = dataset
    m = KmerLogisticModel(k=3).fit(train.onehot, train.labels)
    w = m.linear_weights()
    assert w is not None
    assert w.shape == (train.seq_len, 4)


# ---------------------------------------------------------------- 卷积模型
def test_conv_model_fit_predict(dataset):
    _, train = dataset
    m = ConvMaxPoolModel(n_filters=4, width=6, n_epochs=3, seed=0)
    m.fit(train.onehot, train.labels)
    assert m.decision_function(train.onehot).shape == (len(train),)


def test_conv_model_completeness(dataset):
    """max 池化卷积：grad×input 是精确分解。"""
    _, train = dataset
    m = ConvMaxPoolModel(n_filters=4, width=6, n_epochs=5, seed=0)
    m.fit(train.onehot, train.labels)
    x = train.onehot[:10]
    phi = m.exact_attribution(x).sum(axis=1)
    f0 = float(m.decision_function(np.zeros_like(x[:1]))[0])
    assert np.allclose(phi, m.decision_function(x) - f0, atol=1e-6)


def test_conv_model_rejects_too_short():
    m = ConvMaxPoolModel(n_filters=2, width=20)
    m._init_params(None)
    with pytest.raises(ValueError, match="小于卷积核宽度"):
        m.decision_function(np.zeros((1, 10, 4), dtype=np.float32))


def test_conv_model_requires_fit():
    m = ConvMaxPoolModel(n_filters=2, width=5)
    with pytest.raises(RuntimeError, match="尚未初始化"):
        m.decision_function(np.zeros((1, 20, 4), dtype=np.float32))


def test_conv_model_filter_export(dataset):
    _, train = dataset
    m = ConvMaxPoolModel(n_filters=3, width=6, n_epochs=2, seed=0)
    m.fit(train.onehot, train.labels)
    filters = m.learned_filters_as_pwm()
    assert len(filters) == 3
    assert filters[0].shape == (4, 6)
    assert m.filter_importance().shape == (3,)


def test_conv_model_history_recorded(dataset):
    _, train = dataset
    m = ConvMaxPoolModel(n_filters=2, width=5, n_epochs=3, seed=0)
    m.fit(train.onehot, train.labels)
    assert len(m.history_) == 3
    assert all(np.isfinite(v) for v in m.history_)


# ---------------------------------------------------------------- 工厂
def test_build_model_pwm():
    assert isinstance(build_model("pwm", motifs=make_motifs(1)), PWMScoringModel)


def test_build_model_kmer():
    assert isinstance(build_model("kmer", k=3), KmerLogisticModel)


def test_build_model_unknown():
    with pytest.raises(ValueError, match="未知模型"):
        build_model("transformer")
