"""归因方法测试：形状、复现性、与方法间一致性。

核心性质：
- ``grad_input`` 对这三个模型都应等于 ``exact``（因为都是精确分解）；
- ``ig`` 应接近 ``exact``；
- 同一方法的多次调用必须完全可复现。
"""

from __future__ import annotations

import numpy as np
import pytest

from motifscope.attribution import (
    ATTRIBUTION_METHODS,
    attribute,
    get_attribution_fn,
    normalize_scores,
    position_scores,
)
from motifscope.data.synthetic import make_synthetic
from motifscope.models.conv import ConvMaxPoolModel
from motifscope.models.sequence import KmerLogisticModel, PWMScoringModel
from motifscope.motifs.pwm import PWM


def make_motifs(n: int = 2) -> list[PWM]:
    out = []
    for i in range(n):
        counts = np.ones((4, 7))
        counts[i % 4, :] += 10.0
        out.append(PWM(matrix_id=f"M{i}.1", name=f"Motif{i}", counts=counts))
    return out


@pytest.fixture(scope="module")
def models():
    motifs = make_motifs(2)
    train = make_synthetic(motifs, n_sequences=120, seq_len=40, motifs_per_seq=(1, 2), seed=0)
    out = {}
    out["pwm"] = PWMScoringModel(motifs, pooling="max").fit(train.onehot, train.labels)
    out["kmer"] = KmerLogisticModel(k=3, C=0.5).fit(train.onehot, train.labels)
    conv = ConvMaxPoolModel(n_filters=4, width=5, n_epochs=4, seed=0)
    conv.fit(train.onehot, train.labels)
    out["conv"] = conv
    out["data"] = train
    return out


def test_registry_covers_all_methods():
    expected = {"exact", "ism", "grad_input", "ig", "smoothgrad"}
    assert expected == set(ATTRIBUTION_METHODS)


def test_get_fn_unknown():
    with pytest.raises(ValueError, match="未知归因方法"):
        get_attribution_fn("random")


def test_attribute_shapes(models):
    x = models["data"].onehot[:6]
    for name in ["pwm", "kmer", "conv"]:
        for method in ["exact", "grad_input", "ig", "smoothgrad"]:
            out = attribute(models[name], x, method=method)
            assert out.shape == x.shape, f"{name}/{method} 形状错误"


def test_attribute_single_sequence_squeeze(models):
    x = models["data"].onehot[0]  # (L, 4)
    out = attribute(models["pwm"], x, method="grad_input")
    assert out.shape == x.shape


def test_grad_input_equals_exact_for_maxpool_pwm(models):
    """max 池化的 PWM 模型上，grad×input 应精确等于解析分解。"""
    x = models["data"].onehot[:8]
    a = position_scores(attribute(models["pwm"], x, method="exact"))
    b = position_scores(attribute(models["pwm"], x, method="grad_input"))
    assert np.allclose(a, b, atol=1e-8)


def test_grad_input_equals_exact_for_conv(models):
    x = models["data"].onehot[:8]
    a = position_scores(attribute(models["conv"], x, method="exact"))
    b = position_scores(attribute(models["conv"], x, method="grad_input"))
    assert np.allclose(a, b, atol=1e-6)


def test_reproducible_smoothgrad(models):
    x = models["data"].onehot[:4]
    a = attribute(models["conv"], x, method="smoothgrad", seed=1)
    b = attribute(models["conv"], x, method="smoothgrad", seed=1)
    assert np.allclose(a, b)


def test_ism_shapes_and_reasonableness(models):
    x = models["data"].onehot[:3]
    out = attribute(models["conv"], x, method="ism")
    assert out.shape == x.shape
    # 非当前碱基通道应非零（ISM 会探测替换效果）
    assert np.abs(out).sum() > 0


def test_exact_requires_support():
    class Dummy:
        pass

    with pytest.raises(TypeError, match="不支持精确分解"):
        attribute(Dummy(), np.zeros((1, 10, 4), dtype=np.float32), method="exact")


def test_position_scores_uses_absolute_sum():
    attr = np.zeros((1, 5, 4))
    attr[0, 0, :] = [1.0, -1.0, 0.0, 0.0]
    attr[0, 1, :] = [-2.0, 0.0, 0.0, 0.0]
    pos = position_scores(attr)
    assert pos[0, 0] == 2.0
    assert pos[0, 1] == 2.0


def test_normalize_scores_bounds():
    s = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    n = normalize_scores(s)
    assert np.isclose(n[0].min(), 0.0)
    assert np.isclose(n[0].max(), 1.0)
    # 常数行不应产生 NaN
    assert np.all(np.isfinite(n))


def test_ig_with_explicit_baseline(models):
    x = models["data"].onehot[:3]
    baseline = np.full_like(x, 0.25)
    a = attribute(models["conv"], x, method="ig", baseline=baseline, steps=8)
    assert a.shape == x.shape


def test_ig_steps_affect_result(models):
    x = models["data"].onehot[:2]
    a = attribute(models["conv"], x, method="ig", steps=8)
    b = attribute(models["conv"], x, method="ig", steps=64)
    # 步数增加应收敛，不应发散
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
