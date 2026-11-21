"""序列编码与 k-mer 工具测试。"""

from __future__ import annotations

import numpy as np
import pytest

from motifscope.seq import (
    BASES,
    decode,
    encode,
    encode_batch,
    kmer_index,
    kmer_onehot,
    kmer_vocabulary,
    reverse_complement,
)


def test_encode_shape_and_values():
    x = encode("ACGT")
    assert x.shape == (4, 4)
    assert np.allclose(x.sum(axis=1), 1.0)
    for i, b in enumerate("ACGT"):
        assert x[i, BASES.index(b)] == 1.0


def test_encode_handles_n():
    x = encode("ANGT")
    assert x.shape == (4, 4)
    # N 编码为全 0
    assert np.allclose(x[1], 0.0)


def test_decode_roundtrip():
    seq = "ACGTACGT"
    assert decode(encode(seq)) == seq


def test_encode_batch_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="长度必须一致"):
        encode_batch(["ACG", "ACGT"])


def test_encode_batch_empty():
    with pytest.raises(ValueError, match="不能为空"):
        encode_batch([])


def test_reverse_complement():
    assert reverse_complement("ACGT") == "ACGT"
    assert reverse_complement("AACC") == "GGTT"
    assert reverse_complement("ATN") == "NAT"


def test_reverse_complement_is_involution():
    seq = "ACGTACGTTGCA"
    assert reverse_complement(reverse_complement(seq)) == seq


def test_kmer_vocabulary():
    v = kmer_vocabulary(2)
    assert len(v) == 16
    assert v[0] == "AA"
    assert v[1] == "AC"
    assert v[5] == "CC"


def test_kmer_index_matches_vocabulary():
    idx = kmer_index(3)
    assert len(idx) == 64
    assert idx["AAA"] == 0


def test_kmer_onehot_matches_manual_count():
    x = encode("AAACCC")
    km = kmer_onehot(x[None, ...], k=3)
    assert km.shape == (1, 4, 64)
    # 窗口 0 = AAA -> code 0；窗口 1 = AAC -> code 1
    assert km[0, 0].argmax() == 0
    assert km[0, 1].argmax() == 1
    assert np.allclose(km.sum(axis=-1), 1.0)


def test_kmer_onehot_rejects_short_sequence():
    with pytest.raises(ValueError, match="小于"):
        kmer_onehot(encode("AC")[None, ...], k=5)
