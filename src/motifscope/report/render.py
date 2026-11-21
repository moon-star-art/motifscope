"""中文 HTML 报告渲染：sequence logo、热图、曲线与结论摘要。"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..motifs.pwm import PWM

# 中文字体回退链
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_sequence_logo(pwm: PWM, title: str | None = None) -> str:
    """绘制 sequence logo，返回 base64 PNG。"""
    heights = pwm.sequence_logo_heights()
    colors = {"A": "#2ecc71", "C": "#3498db", "G": "#f39c12", "T": "#e74c3c"}
    from ..seq import BASES

    fig, ax = plt.subplots(figsize=(max(3.5, pwm.length * 0.32), 1.9))
    for pos in range(pwm.length):
        y = 0.0
        for b in np.argsort(heights[:, pos]):  # 矮的在下
            h = heights[b, pos]
            if h <= 1e-6:
                continue
            ax.add_patch(
                plt.Rectangle((pos - 0.45, y), 0.9, h, color=colors[BASES[b]], alpha=0.9)
            )
            if h > 0.25:
                ax.text(
                    pos, y + h / 2, BASES[b], ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                )
            y += h
    ax.set_xlim(-0.6, pwm.length - 0.4)
    ax.set_ylim(0, 2.1)
    ax.set_xlabel("位置 (bp)", fontsize=8)
    ax.set_ylabel("信息量 (bit)", fontsize=8)
    ax.set_title(title or f"{pwm.name} ({pwm.matrix_id})", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_attribution_heatmap(
    sequences: list[str],
    truth: np.ndarray,
    score_dict: dict[str, np.ndarray],
    max_rows: int = 3,
) -> str:
    """绘制若干条序列的"真值 vs 各方法归因"对照热图。"""
    n_rows = min(max_rows, len(sequences))
    methods = list(score_dict)
    fig, axes = plt.subplots(
        len(methods) + 1, 1, figsize=(9, 0.85 * (len(methods) + 1)), squeeze=False
    )
    axes[0][0].imshow(truth[:n_rows], aspect="auto", cmap="Greys", vmin=0, vmax=1.3)
    axes[0][0].set_ylabel("真值", fontsize=8)
    axes[0][0].set_yticks([])
    for r, method in enumerate(methods, start=1):
        v = np.asarray(score_dict[method])[:n_rows]
        vmax = np.max(np.abs(v)) or 1.0
        axes[r][0].imshow(v, aspect="auto", cmap="Reds", vmin=0, vmax=vmax)
        axes[r][0].set_ylabel(method, fontsize=8)
        axes[r][0].set_yticks([])
    axes[-1][0].set_xlabel("序列位置 (bp)", fontsize=8)
    for ax in axes.ravel():
        ax.tick_params(labelsize=6)
    fig.suptitle("归因分数与植入基序真值的对照（深色=高重要性）", fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


def plot_curves(curves: dict[str, dict]) -> str:
    """删除/插入曲线。"""
    n = len(curves)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(3.4 * max(n, 1), 2.8), squeeze=False)
    for ax, (name, data) in zip(axes[0], curves.items()):
        d = data["deletion"]
        ax.plot(d["fractions"], d["probs"], marker="o", ms=2, lw=1.2)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("删除比例", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("平均预测概率", fontsize=7)
    fig.suptitle("删除曲线：忠实归因应使预测快速下降", fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


def render_report(
    out_path: str | Path,
    title: str,
    summary: dict,
    results: list[dict],
    logos: dict[str, str] | None = None,
    figures: dict[str, str] | None = None,
) -> Path:
    """渲染中文 HTML 报告。"""
    from jinja2 import Template

    template = Template(_TEMPLATE)
    html = template.render(
        title=title,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        summary=summary,
        results=results,
        logos=logos or {},
        figures=figures or {},
    )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
  :root { --fg:#1f2937; --muted:#6b7280; --line:#e5e7eb; --accent:#0f766e; --bg:#fbfbfa; }
  * { box-sizing: border-box; }
  body { font-family: "Microsoft YaHei","PingFang SC","Noto Sans SC",system-ui,sans-serif;
         color: var(--fg); background: var(--bg); margin: 0; padding: 32px 24px; line-height: 1.7; }
  .wrap { max-width: 960px; margin: 0 auto; }
  h1 { font-size: 22px; border-bottom: 2px solid var(--accent); padding-bottom: 10px; }
  h2 { font-size: 16px; margin-top: 32px; color: var(--accent); }
  .meta { color: var(--muted); font-size: 12px; margin-bottom: 20px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0; background: #fff; }
  th, td { border: 1px solid var(--line); padding: 7px 10px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  thead th { background: #f3f4f6; font-weight: 600; }
  tbody tr:nth-child(even) { background: #fafafa; }
  .card { background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px; margin:14px 0; }
  .kpis { display:flex; gap:14px; flex-wrap:wrap; margin:14px 0; }
  .kpi { flex:1 1 150px; background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
  .kpi .v { font-size:20px; font-weight:700; color:var(--accent); }
  .kpi .l { font-size:12px; color:var(--muted); }
  .figure { text-align:center; margin:16px 0; }
  .figure img { max-width:100%; border:1px solid var(--line); border-radius:6px; background:#fff; }
  .logos { display:flex; flex-wrap:wrap; gap:12px; }
  .logos figure { margin:0; background:#fff; border:1px solid var(--line); border-radius:6px; padding:8px; }
  .note { font-size:12px; color:var(--muted); }
  .good { color:#15803d; font-weight:600; }
  .bad { color:#b91c1c; font-weight:600; }
  footer { margin-top:40px; font-size:11px; color:var(--muted); border-top:1px solid var(--line); padding-top:12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>{{ title }}</h1>
  <div class="meta">生成时间：{{ generated }} ｜ motifscope</div>

  <h2>一、概览</h2>
  <div class="kpis">
  {% for k, v in summary.items() %}
    <div class="kpi"><div class="v">{{ v }}</div><div class="l">{{ k }}</div></div>
  {% endfor %}
  </div>

  {% if logos %}
  <h2>二、基序标识</h2>
  <div class="logos">
    {% for name, b64 in logos.items() %}
    <figure><img src="data:image/png;base64,{{ b64 }}" style="max-width:260px"><figcaption class="note">{{ name }}</figcaption></figure>
    {% endfor %}
  </div>
  {% endif %}

  <h2>三、归因忠实度结果</h2>
  <table>
    <thead><tr>
      <th>模型</th><th>归因方法</th><th>位置 AUC</th><th>Top10% 命中</th>
      <th>峰值命中</th><th>AOPC</th><th>充分性</th><th>耗时(s)</th>
    </tr></thead>
    <tbody>
    {% for r in results %}
      <tr>
        <td>{{ r.model }}</td><td>{{ r.method }}</td>
        <td>{{ "%.3f"|format(r.position_auc) }}</td>
        <td>{{ "%.3f"|format(r.top10pct_hit) }}</td>
        <td>{{ "%.3f"|format(r.peak_inside) }}</td>
        <td class="{{ 'bad' if r.aopc < 0 else 'good' }}">{{ "%.3f"|format(r.aopc) }}</td>
        <td>{{ "%.3f"|format(r.sufficiency) }}</td>
        <td>{{ "%.2f"|format(r.seconds) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  <p class="note">位置 AUC = 归因分数区分"真值基序内/外"的能力；AOPC 为删除曲线面积指标，负值说明按归因删除反而提升预测，即归因不可信。</p>

  {% for name, b64 in figures.items() %}
  <div class="figure"><img src="data:image/png;base64,{{ b64 }}"><div class="note">{{ name }}</div></div>
  {% endfor %}

  <footer>由 motifscope 自动生成。所有数值均来自本次运行，未做人工调整。</footer>
</div>
</body>
</html>
"""
