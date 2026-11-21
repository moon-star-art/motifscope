"""CLI 与报告渲染的测试。"""

from __future__ import annotations

import numpy as np
import pytest

from motifscope.attribution import attribute, position_scores
from motifscope.cli import build_parser, main
from motifscope.data.synthetic import make_heldout_motifs, make_synthetic
from motifscope.evaluate import recovery_report
from motifscope.motifs.database import MotifDatabase
from motifscope.models.conv import ConvMaxPoolModel
from motifscope.report import (
    plot_attribution_heatmap,
    plot_curves,
    plot_sequence_logo,
    render_report,
)


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    # 显式给一个可写缓存目录，避免依赖用户主目录的写权限
    cache = tmp_path_factory.mktemp("motifcache")
    return MotifDatabase.load_default(cache_dir=cache, allow_network=False)


# ------------------------------------------------------------ 报告渲染
def test_plot_sequence_logo_returns_base64_png(db):
    m = next(iter(db))
    b64 = plot_sequence_logo(m)
    assert isinstance(b64, str) and len(b64) > 100
    # PNG 的 base64 应以 iVBOR 开头
    assert b64.startswith("iVBOR")


def test_plot_attribution_heatmap(db):
    m = next(iter(db))
    truth = np.zeros((3, 50), dtype=np.int8)
    truth[:, 10:20] = 1
    scores = {"exact": np.random.default_rng(0).random((3, 50))}
    b64 = plot_attribution_heatmap(["A" * 50] * 3, truth, scores)
    assert b64.startswith("iVBOR")


def test_plot_curves():
    curves = {
        "m1": {
            "deletion": {"fractions": [0, 0.5, 1], "probs": [0.9, 0.5, 0.1]},
            "insertion": {"fractions": [0, 0.5, 1], "probs": [0.1, 0.5, 0.9]},
        }
    }
    assert plot_curves(curves).startswith("iVBOR")


def test_render_report(tmp_dir):
    out = tmp_dir / "r.html"
    path = render_report(
        out,
        title="测试报告",
        summary={"基序数": 3, "序列数": 10},
        results=[
            {
                "model": "PWM",
                "method": "exact",
                "position_auc": 0.8,
                "top10pct_hit": 0.2,
                "peak_inside": 0.5,
                "aopc": 0.1,
                "sufficiency": 0.6,
                "seconds": 0.01,
            }
        ],
    )
    assert path.exists()
    html = out.read_text(encoding="utf-8")
    assert "测试报告" in html
    assert "PWM" in html
    assert "<table>" in html


def test_render_report_with_figures(tmp_dir, db):
    m = next(iter(db))
    out = tmp_dir / "r2.html"
    path = render_report(
        out,
        title="带图报告",
        summary={"a": 1},
        results=[],
        logos={"logo1": plot_sequence_logo(m)},
        figures={"fig1": plot_sequence_logo(m)},
    )
    html = path.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html


def test_render_report_creates_parent_dir(tmp_dir):
    out = tmp_dir / "nested" / "deep" / "r.html"
    render_report(out, title="T", summary={}, results=[])
    assert out.exists()


# ------------------------------------------------------------ CLI 解析
def test_parser_requires_subcommand():
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_parser_demo_defaults():
    p = build_parser()
    args = p.parse_args(["demo"])
    assert args.command == "demo"
    assert args.n_motifs == 6
    assert args.func.__name__ == "cmd_demo"


def test_parser_scan_args():
    p = build_parser()
    args = p.parse_args(["scan", "x.fa", "--pvalue", "1e-5", "--no-correct"])
    assert args.fasta == "x.fa"
    assert args.pvalue == 1e-5
    assert args.correct is False


def test_parser_bench():
    p = build_parser()
    args = p.parse_args(["bench", "--out", "res"])
    assert args.out == "res"


def test_parser_motifs():
    p = build_parser()
    args = p.parse_args(["motifs"])
    assert args.command == "motifs"


# ------------------------------------------------------------ CLI 行为
def test_cli_motifs_runs(capsys):
    rc = main(["motifs"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "基序" in out


def test_cli_motifs_lists_ids(capsys, db):
    main(["motifs"])
    out = capsys.readouterr().out
    assert db.ids[0] in out


def test_cli_scan(tmp_dir, capsys):
    fa = tmp_dir / "x.fa"
    fa.write_text(">s1\n" + "ACGT" * 30 + "\n", encoding="utf-8")
    rc = main(["scan", str(fa), "--pvalue", "0.05", "--max-show", "2"])
    assert rc == 0
    assert "读入" in capsys.readouterr().out


def test_cli_scan_writes_bed(tmp_dir):
    fa = tmp_dir / "x.fa"
    fa.write_text(">s1\n" + "ACGT" * 30 + "\n", encoding="utf-8")
    bed = tmp_dir / "o.bed"
    main(["scan", str(fa), "--pvalue", "0.5", "--bed", str(bed)])
    assert bed.exists()
    assert bed.read_text(encoding="utf-8").startswith("#chrom")


def test_cli_scan_writes_json(tmp_dir):
    fa = tmp_dir / "x.fa"
    fa.write_text(">s1\n" + "ACGT" * 30 + "\n", encoding="utf-8")
    js = tmp_dir / "o.json"
    main(["scan", str(fa), "--pvalue", "0.5", "--json", str(js)])
    assert js.exists()


def test_cli_demo_small(tmp_dir, capsys):
    out = tmp_dir / "d.html"
    rc = main(
        [
            "demo",
            "--out",
            str(out),
            "--n-motifs",
            "2",
            "--n-train",
            "60",
            "--n-test",
            "8",
            "--seq-len",
            "50",
            "--epochs",
            "2",
        ]
    )
    assert rc == 0
    assert out.exists()
    assert "motifscope" in capsys.readouterr().out
