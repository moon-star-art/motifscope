"""基序库、数据加载器与包结构的测试。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from motifscope.data.splice import SPLICE_CLASSES, SpliceDataset, parse_splice
from motifscope.motifs.database import MotifDatabase


# ------------------------------------------------------------ 基序库
def test_bundled_motifs_load_offline(tmp_dir):
    """无网络时必须能用内置基序集启动。"""
    db = MotifDatabase.load_default(cache_dir=tmp_dir, allow_network=False)
    assert len(db) > 0
    for m in db:
        assert m.length >= 2
        assert m.counts.shape[0] == 4


def test_motif_database_lookup(tmp_dir):
    db = MotifDatabase.load_default(cache_dir=tmp_dir, allow_network=False)
    first_id = db.ids[0]
    assert db[first_id].matrix_id == first_id
    assert db.get("NOPE") is None


def test_motif_database_select_is_deterministic(tmp_dir):
    db = MotifDatabase.load_default(cache_dir=tmp_dir, allow_network=False)
    a = [m.matrix_id for m in db.select(3)]
    b = [m.matrix_id for m in db.select(3)]
    assert a == b


def test_motif_database_select_more_than_available(tmp_dir):
    db = MotifDatabase.load_default(cache_dir=tmp_dir, allow_network=False)
    assert len(db.select(len(db) + 10)) == len(db)


def test_motif_database_filter_by_length(tmp_dir):
    db = MotifDatabase.load_default(cache_dir=tmp_dir, allow_network=False)
    filtered = db.filter_by_length(6, 10)
    for m in filtered:
        assert 6 <= m.length <= 10


def test_motif_database_filter_by_information(tmp_dir):
    db = MotifDatabase.load_default(cache_dir=tmp_dir, allow_network=False)
    filtered = db.filter_by_information(10.0)
    for m in filtered:
        assert m.information_content() >= 10.0


def test_motif_database_json_roundtrip(tmp_dir):
    db = MotifDatabase.load_default(cache_dir=tmp_dir, allow_network=False)
    out = tmp_dir / "m.json"
    db.to_json(out)
    db2 = MotifDatabase.from_json(out)
    assert len(db2) == len(db)
    assert set(db2.ids) == set(db.ids)


def test_motif_database_uses_cache(tmp_dir):
    db = MotifDatabase.load_default(cache_dir=tmp_dir, allow_network=False)
    db.to_json(tmp_dir / "jaspar_core.json")
    db2 = MotifDatabase.load_default(cache_dir=tmp_dir, allow_network=False)
    assert len(db2) == len(db)


def test_bundled_resource_is_valid_json():
    from importlib import resources

    ref = resources.files("motifscope.resources").joinpath("jaspar_core.json")
    data = json.loads(ref.read_text(encoding="utf-8"))
    assert "matrices" in data
    assert len(data["matrices"]) >= 10
    for item in data["matrices"]:
        assert "matrix_id" in item
        assert len(item["counts"]) == 4


# ------------------------------------------------------------ 剪接数据
SPLICE_SAMPLE = """EI,    TEST-DONOR-1,   CCAGCTGCATCACAGGAGGCCAGCGAG
IE,    TEST-ACCEPT-1,  AGACCCGCCGGGAGGCGGAGGACCTGC
N,     TEST-NONE-1,    GGGCTGCGTTGCTGGTCACATTCCTG
"""


def test_parse_splice_basic():
    d = parse_splice(SPLICE_SAMPLE)
    assert len(d) == 3
    assert set(d.labels.tolist()) == {0, 1, 2}
    assert d.seq_len == 27


def test_parse_splice_skips_bad_rows():
    raw = SPLICE_SAMPLE + "XX, BAD, ACGT\n"
    d = parse_splice(raw)
    assert len(d) == 3


def test_parse_splice_class_counts():
    d = parse_splice(SPLICE_SAMPLE)
    counts = d.class_counts()
    assert counts["EI"] == 1
    assert counts["IE"] == 1
    assert counts["N"] == 1


def test_parse_splice_empty_raises():
    with pytest.raises(ValueError, match="未解析出"):
        parse_splice("")


def test_splice_subset_is_stratified():
    d = parse_splice(SPLICE_SAMPLE * 20)
    sub = d.subset(9, seed=0)
    assert len(sub.labels) == len(sub.sequences)
    assert len(set(sub.labels.tolist())) == 3


def test_splice_dataset_summary():
    d = parse_splice(SPLICE_SAMPLE)
    s = d.summary()
    assert s["n_sequences"] == 3
    assert "class_counts" in s


def test_splice_label_names():
    d = parse_splice(SPLICE_SAMPLE)
    assert set(d.label_names) == {"EI", "IE", "N"}


def test_splice_classes_constant():
    assert SPLICE_CLASSES == {"EI": 0, "IE": 1, "N": 2}


# ------------------------------------------------------------ 包
def test_version_exposed():
    import motifscope

    assert motifscope.__version__


def test_public_api_importable():
    from motifscope import attribution, data, evaluate, models, motifs, report, seq  # noqa: F401

    assert callable(models.build_model)
    assert callable(attribution.attribute)
    assert callable(evaluate.recovery_report)
