"""motifscope 基准实验：生成论文所需的全部结果表与图形。

运行：
    python examples/run_benchmark.py --out results

产物（results/ 下）：
    benchmark.json      全部数值结果
    fig_*.png           论文用图
    results.csv         主结果表
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motifscope.attribution import attribute, position_scores
from motifscope.data import make_heldout_motifs, make_synthetic
from motifscope.evaluate import (
    aopc,
    deletion_curve,
    evaluate_attribution,
    insertion_curve,
    method_agreement,
    recovery_report,
    sufficiency,
)
from motifscope.motifs.database import MotifDatabase
from motifscope.models import ConvMaxPoolModel, KmerLogisticModel, PWMScoringModel

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130


def main() -> int:
    ap = argparse.ArgumentParser(description="motifscope 基准实验")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--seq-len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 数据
    cache = Path(__file__).resolve().parents[1] / "data_cache"
    db = MotifDatabase.load_default(cache_dir=cache, allow_network=False)
    motifs = db.select(6)
    print(f"[数据] 基序 {len(motifs)} 条: " + ", ".join(f"{m.name}({m.length})" for m in motifs))

    train = make_synthetic(
        motifs, n_sequences=args.n_train, seq_len=args.seq_len, motifs_per_seq=(1, 3), seed=args.seed
    )
    val = make_synthetic(
        motifs, n_sequences=max(200, args.n_train // 6), seq_len=args.seq_len,
        motifs_per_seq=(1, 3), seed=args.seed + 6,
    )
    test = make_heldout_motifs(motifs, n_sequences=args.n_test, seq_len=args.seq_len, seed=args.seed + 1)
    print(f"[数据] train={len(train)} val={len(val)} test={len(test)}")

    # ---------------------------------------------------------------- 模型
    models = {}
    print("[模型] 训练 PWM-max ...")
    models["PWM-max"] = PWMScoringModel(motifs, pooling="max").fit(train.onehot, train.labels)

    print("[模型] 训练 k-mer 逻辑回归 ...")
    models["k-mer LR"] = KmerLogisticModel(k=4, C=0.05).fit(train.onehot, train.labels)

    print("[模型] 训练 卷积+全局max池化 ...")
    conv = ConvMaxPoolModel(n_filters=16, width=10, n_epochs=args.epochs, lr=0.01, seed=0)
    conv.fit(train.onehot, train.labels)
    models["CNN-max"] = conv

    # 模型性能
    perf = {}
    for name, m in models.items():
        perf[name] = {
            "val_accuracy": float((m.predict(val.onehot) == val.labels).mean()),
            "train_accuracy": float((m.predict(train.onehot) == train.labels).mean()),
        }
        print(f"  {name:10s} train={perf[name]['train_accuracy']:.3f} val={perf[name]['val_accuracy']:.3f}")

    # -------------------------------------------------------------- 归因评测
    methods = ["exact", "ism", "grad_input", "ig", "smoothgrad"]
    rows: list[dict] = []
    curves: dict[str, dict] = {}
    scores_store: dict[str, dict[str, np.ndarray]] = {}

    for mname, model in models.items():
        scores_store[mname] = {}
        for method in methods:
            # 该模型不支持 exact 时跳过
            if method == "exact" and not hasattr(model, "exact_attribution"):
                continue
            t0 = time.time()
            try:
                attr = attribute(model, test.onehot, method=method)
            except Exception as exc:  # noqa: BLE001
                print(f"  [跳过] {mname}/{method}: {exc}")
                continue
            elapsed = time.time() - t0
            pos = position_scores(attr)
            scores_store[mname][method] = pos

            rep = recovery_report(pos, test.truth)
            dele = deletion_curve(model, test.onehot, pos, steps=20)
            ins = insertion_curve(model, test.onehot, pos, steps=20)
            row = {
                "model": mname,
                "method": method,
                "position_auc": rep["position_auc_mean"],
                "position_auc_std": rep["position_auc_std"],
                "top10pct_hit": rep["topk_hit_mean"],
                "peak_inside": rep["peak_inside_rate"],
                "aopc": aopc(dele),
                "sufficiency": sufficiency(ins),
                "seconds": elapsed,
                "val_accuracy": perf[mname]["val_accuracy"],
            }
            rows.append(row)
            curves[f"{mname}|{method}"] = {"deletion": dele, "insertion": ins}
            print(
                f"  {mname:10s} {method:12s} AUC={row['position_auc']:.3f} "
                f"top10%={row['top10pct_hit']:.3f} AOPC={row['aopc']:.3f} ({elapsed:.2f}s)"
            )

    # -------------------------------------------------------- 方法一致性
    agreement = {}
    for mname, store in scores_store.items():
        if "exact" in store:
            base = store["exact"]
            for method, sc in store.items():
                if method == "exact":
                    continue
                agreement[f"{mname}|exact_vs_{method}"] = method_agreement(base, sc)

    # ------------------------------------------------------ 完整性检验
    completeness = {}
    for mname, model in models.items():
        if not hasattr(model, "exact_attribution"):
            continue
        sub = test.onehot[:20]
        attr = attribute(model, sub, method="exact")
        s = attr.sum(axis=(1, 2))
        fx = model.decision_function(sub)
        f0 = model.decision_function(np.zeros_like(sub))
        completeness[mname] = float(np.abs(s - (fx - f0)).max())

    # -------------------------------------------------- 核宽 vs 基序长度
    width_sweep = []
    groups = {
        "短(6-8bp)": [m for m in db if m.length <= 8],
        "中(10-12bp)": [m for m in db if 10 <= m.length <= 12],
        "长(15-17bp)": [m for m in db if 15 <= m.length <= 17],
    }
    print("[实验] 核宽敏感性 ...")
    for gname, group in groups.items():
        if len(group) < 2:
            continue
        gtr = make_synthetic(group, n_sequences=1200, seq_len=args.seq_len, motifs_per_seq=(1, 2), seed=args.seed)
        gva = make_synthetic(group, n_sequences=250, seq_len=args.seq_len, motifs_per_seq=(1, 2), seed=args.seed + 6)
        gte = make_heldout_motifs(group, n_sequences=50, seq_len=args.seq_len, seed=args.seed + 1)
        for w in [6, 8, 10, 12, 16]:
            m = ConvMaxPoolModel(n_filters=16, width=w, n_epochs=20, lr=0.01, seed=0)
            m.fit(gtr.onehot, gtr.labels)
            pos = position_scores(attribute(m, gte.onehot, method="grad_input"))
            rep = recovery_report(pos, gte.truth)
            rec = {
                "group": gname,
                "mean_motif_len": float(np.mean([x.length for x in group])),
                "width": w,
                "val_accuracy": float((m.predict(gva.onehot) == gva.labels).mean()),
                "position_auc": rep["position_auc_mean"],
                "top10pct_hit": rep["topk_hit_mean"],
            }
            width_sweep.append(rec)
            print(f"  {gname} w={w:<3} val={rec['val_accuracy']:.3f} AUC={rec['position_auc']:.3f}")

    # ---------------------------------------------------------------- 汇总
    result = {
        "config": {
            "n_train": len(train),
            "n_test": len(test),
            "seq_len": args.seq_len,
            "seed": args.seed,
            "motifs": [{"id": m.matrix_id, "name": m.name, "length": m.length, "consensus": m.consensus} for m in motifs],
        },
        "model_performance": perf,
        "attribution_results": rows,
        "method_agreement": agreement,
        "completeness_max_error": completeness,
        "width_sweep": width_sweep,
    }
    (out / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV 主表
    if rows:
        keys = list(rows[0].keys())
        with (out / "results.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    # ---------------------------------------------------------------- 绘图
    # 图1：position AUC 对比条形图
    if rows:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        model_names = list(models)
        method_names = [m for m in methods if any(r["method"] == m for r in rows)]
        width = 0.8 / max(len(method_names), 1)
        xs = np.arange(len(model_names))
        for j, method in enumerate(method_names):
            vals = []
            for mn in model_names:
                hit = [r["position_auc"] for r in rows if r["model"] == mn and r["method"] == method]
                vals.append(hit[0] if hit else np.nan)
            ax.bar(xs + j * width, vals, width, label=method)
        ax.set_xticks(xs + width * (len(method_names) - 1) / 2)
        ax.set_xticklabels(model_names)
        ax.set_ylabel("位置级 ROC-AUC")
        # ax.set_title("图1  不同归因方法的位置级忠实度（越高越好）")
        ax.axhline(0.5, ls="--", c="gray", lw=0.8)
        ax.legend(fontsize=8, ncol=len(method_names))
        ax.set_ylim(0.4, max(0.95, max(r["position_auc"] for r in rows) + 0.05))
        fig.tight_layout()
        fig.savefig(out / "fig1_position_auc.png")
        plt.close(fig)

    # 图2：删除曲线
    if curves:
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), squeeze=False)
        for ax, mn in zip(axes[0], models):
            any_curve = False
            for method in methods:
                key = f"{mn}|{method}"
                if key in curves:
                    d = curves[key]["deletion"]
                    ax.plot(d["fractions"], d["probs"], marker="o", ms=2, lw=1.2, label=method)
                    any_curve = True
            ax.set_title(mn, fontsize=10)
            ax.set_xlabel("被删除位置比例")
            if any_curve:
                ax.legend(fontsize=7)
        axes[0][0].set_ylabel("平均预测概率")
        # fig.suptitle("图2  删除曲线：忠实方法的预测应快速下降", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "fig2_deletion.png")
        plt.close(fig)

    # 图3：核宽敏感性
    if width_sweep:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for gname in dict.fromkeys(r["group"] for r in width_sweep):
            sub = [r for r in width_sweep if r["group"] == gname]
            ax.plot([r["width"] for r in sub], [r["position_auc"] for r in sub], marker="o", label=gname)
        ax.set_xlabel("卷积核宽度 w")
        ax.set_ylabel("位置级 ROC-AUC")
        # ax.set_title("图3  核宽与基序长度的匹配效应")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "fig3_width_sweep.png")
        plt.close(fig)

    # 图4：归因热图示例
    if scores_store:
        mn = next(iter(scores_store))
        store = scores_store[mn]
        k = min(3, len(test.sequences))
        fig, axes = plt.subplots(len(store) + 1, 1, figsize=(9, 1.1 * (len(store) + 1)), squeeze=False)
        truth = test.truth[0]
        axes[0][0].imshow(truth[None, :], aspect="auto", cmap="Greys", vmin=0, vmax=1.2)
        axes[0][0].set_title("真值（深色=植入基序）", fontsize=8, loc="left")
        axes[0][0].set_yticks([])
        for r, (method, sc) in enumerate(store.items(), start=1):
            v = sc[0:1]
            axes[r][0].imshow(v / (v.max() + 1e-9), aspect="auto", cmap="Reds")
            axes[r][0].set_title(method, fontsize=8, loc="left")
            axes[r][0].set_yticks([])
        axes[-1][0].set_xlabel("序列位置")
        # fig.suptitle(f"图4  归因与真值对比（模型 {mn}）", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "fig4_heatmap.png")
        plt.close(fig)

    print(f"\n[完成] 结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
