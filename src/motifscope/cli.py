"""motifscope 命令行界面。

用法
----
    motifscope demo                       # 一键端到端演示（离线可用）
    motifscope scan seq.fa                # 注释序列上的 TF 结合位点
    motifscope bench --out results        # 复现论文实验
    motifscope motifs                     # 查看基序库
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _setup_matplotlib() -> None:
    """把 matplotlib 缓存指到可写目录，避免在受限环境下告警。"""
    if "MPLCONFIGDIR" not in os.environ:
        cache = Path.home() / ".cache" / "motifscope" / "mpl"
        try:
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cache)
        except OSError:
            pass


def cmd_demo(args: argparse.Namespace) -> int:
    """端到端演示：合成数据 -> 训练 -> 归因 -> 评测 -> 报告。"""
    import numpy as np

    from .attribution import attribute, position_scores
    from .data import make_heldout_motifs, make_synthetic
    from .evaluate import deletion_curve, recovery_report
    from .motifs.database import MotifDatabase
    from .models import ConvMaxPoolModel, KmerLogisticModel, PWMScoringModel
    from .report import plot_attribution_heatmap, plot_sequence_logo, render_report

    print("=" * 62)
    print(" motifscope 演示")
    print("=" * 62)

    cache = Path(args.cache) if args.cache else Path.cwd() / "data_cache"
    db = MotifDatabase.load_default(cache_dir=cache, allow_network=args.allow_network)
    motifs = db.select(min(args.n_motifs, len(db)))
    print(f"基序库: {len(db)} 条，本次使用 {len(motifs)} 条")
    for m in motifs:
        print(f"  - {m.matrix_id:10s} {m.name:12s} L={m.length:2d} 一致性序列={m.consensus}")

    train = make_synthetic(motifs, n_sequences=args.n_train, seq_len=args.seq_len, motifs_per_seq=(1, 3), seed=1)
    test = make_heldout_motifs(motifs, n_sequences=args.n_test, seq_len=args.seq_len, seed=2)
    print(f"训练集: {train.summary()}")
    print(f"测试集: {len(test)} 条，真值占比 {test.truth.mean():.3f}")

    models = {
        "PWM-max": PWMScoringModel(motifs, pooling="max"),
        "k-mer LR": KmerLogisticModel(k=4, C=0.05),
        "CNN-max": ConvMaxPoolModel(n_filters=16, width=10, n_epochs=args.epochs, lr=0.01, seed=0),
    }

    results: list[dict] = []
    score_store: dict[str, np.ndarray] = {}
    for name, model in models.items():
        print(f"\n[训练] {name} ...")
        model.fit(train.onehot, train.labels)
        acc = float((model.predict(train.onehot) == train.labels).mean())
        print(f"  训练集准确率 = {acc:.3f}")
        for method in ["exact", "grad_input", "ig", "ism"]:
            if method == "exact" and not hasattr(model, "exact_attribution"):
                continue
            attr = attribute(model, test.onehot, method=method)
            pos = position_scores(attr)
            rep = recovery_report(pos, test.truth)
            if name == "PWM-max":
                score_store[method] = pos
            results.append(
                {
                    "model": name,
                    "method": method,
                    "position_auc": rep["position_auc_mean"],
                    "top10pct_hit": rep["topk_hit_mean"],
                    "peak_inside": rep["peak_inside_rate"],
                    "aopc": 0.0,
                    "sufficiency": 0.0,
                    "seconds": 0.0,
                }
            )
            print(f"  {method:12s} 位置AUC={rep['position_auc_mean']:.3f} Top10%命中={rep['topk_hit_mean']:.3f}")

    logos = {}
    for m in motifs[:4]:
        logos[f"{m.name} ({m.matrix_id})"] = plot_sequence_logo(m)

    figures = {}
    if score_store:
        figures["归因 vs 真值热图"] = plot_attribution_heatmap(test.sequences, test.truth, score_store)

    out = Path(args.out)
    summary = {
        "使用基序": len(motifs),
        "训练序列": len(train),
        "测试序列": len(test),
        "平均位置AUC": f"{np.mean([r['position_auc'] for r in results]):.3f}" if results else "-",
    }
    path = render_report(
        out,
        title="motifscope：DNA 序列模型可解释性诊断报告",
        summary=summary,
        results=results,
        logos=logos,
        figures=figures,
    )
    print(f"\n报告已生成: {path}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    """在 FASTA 上注释转录因子结合位点。"""
    from .evaluate.scan import read_fasta, scan_many, write_bed
    from .motifs.database import MotifDatabase

    cache = Path(args.cache) if args.cache else Path.cwd() / "data_cache"
    db = MotifDatabase.load_default(cache_dir=cache, allow_network=args.allow_network)
    motifs = list(db)
    seqs = read_fasta(args.fasta)
    print(f"读入 {len(seqs)} 条序列，使用 {len(motifs)} 条基序，p 值阈值 {args.pvalue}")
    print(
        f"多重检验校正: {'开启(Bonferroni)' if args.correct else '关闭'}  "
        f"重叠命中抑制: {'开启' if not args.no_nms else '关闭'}"
    )

    hits = scan_many(
        seqs,
        motifs,
        pvalue_threshold=args.pvalue,
        both_strands=not args.single_strand,
        correct_multiple_testing=args.correct,
        apply_nms=not args.no_nms,
    )
    total = sum(len(v) for v in hits.values())
    print(f"共找到 {total} 个显著命中")
    for chrom, hs in hits.items():
        print(f"\n{chrom} (长度 {len(seqs[chrom])}): {len(hs)} 个命中")
        for h in hs[: args.max_show]:
            print(
                f"  {h.start:>5d}-{h.end:<5d} {h.strand}  "
                f"{h.motif_name:14s} ({h.motif_id:10s})  "
                f"score={h.score:6.2f}  p={h.pvalue:.2e}"
            )
        if len(hs) > args.max_show:
            print(f"  ... 其余 {len(hs) - args.max_show} 个已省略")

    if args.bed:
        write_bed(hits, args.bed)
        print(f"\nBED 已写入 {args.bed}")
    if args.json:
        payload = {chrom: [h.to_dict() for h in hs] for chrom, hs in hits.items()}
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 已写入 {args.json}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """调用基准实验脚本。"""
    script = Path(__file__).resolve().parents[2] / "examples" / "run_benchmark.py"
    if not script.exists():
        print(f"未找到基准脚本: {script}", file=sys.stderr)
        return 1
    import runpy

    sys.argv = [str(script), "--out", args.out]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def cmd_motifs(args: argparse.Namespace) -> int:
    """列出基序库内容。"""
    from .motifs.database import MotifDatabase

    cache = Path(args.cache) if args.cache else Path.cwd() / "data_cache"
    db = MotifDatabase.load_default(cache_dir=cache, allow_network=args.allow_network)
    print(f"共 {len(db)} 条基序\n")
    print(f"{'编号':<12}{'名称':<16}{'长度':>4}  {'信息量(bit)':>10}  一致性序列")
    print("-" * 78)
    for m in sorted(db, key=lambda x: x.matrix_id):
        print(f"{m.matrix_id:<12}{m.name:<16}{m.length:>4}  {m.information_content():>10.1f}  {m.consensus}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="motifscope",
        description="DNA 调控序列模型的可解释性验证工具包",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
        "  motifscope demo\n"
        "  motifscope scan my.fa --bed hits.bed\n"
        "  motifscope bench --out results\n"
        "  motifscope motifs\n",
    )
    p.add_argument("--cache", default=None, help="基序缓存目录（默认 ./data_cache）")
    p.add_argument("--allow-network", action="store_true", help="允许联网拉取 JASPAR 基序")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="一键端到端演示")
    d.add_argument("--out", default="motifscope_report.html", help="报告输出路径")
    d.add_argument("--n-motifs", type=int, default=6)
    d.add_argument("--n-train", type=int, default=1500)
    d.add_argument("--n-test", type=int, default=60)
    d.add_argument("--seq-len", type=int, default=200)
    d.add_argument("--epochs", type=int, default=20)
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("scan", help="注释序列上的 TF 结合位点")
    s.add_argument("fasta", help="输入 FASTA 文件")
    s.add_argument("--pvalue", type=float, default=1e-4, help="经验 p 值阈值")
    s.add_argument("--single-strand", action="store_true", help="只扫描正链")
    s.add_argument(
        "--no-correct",
        dest="correct",
        action="store_false",
        help="关闭多重检验校正（默认开启 Bonferroni）",
    )
    s.add_argument(
        "--no-nms",
        dest="no_nms",
        action="store_true",
        help="关闭重叠命中抑制（默认开启）",
    )
    s.add_argument("--bed", default=None, help="输出 BED6 文件")
    s.add_argument("--json", default=None, help="输出 JSON 文件")
    s.add_argument("--max-show", type=int, default=10, help="每条序列最多打印的命中数")
    s.set_defaults(func=cmd_scan, correct=True, no_nms=False)

    b = sub.add_parser("bench", help="复现论文基准实验")
    b.add_argument("--out", default="results", help="结果目录")
    b.set_defaults(func=cmd_bench)

    m = sub.add_parser("motifs", help="列出基序库")
    m.set_defaults(func=cmd_motifs)
    return p


def main(argv: list[str] | None = None) -> int:
    _setup_matplotlib()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
