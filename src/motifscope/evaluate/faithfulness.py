"""忠实度（faithfulness）指标：归因分数是否真的反映了模型行为。

这里实现了三类互补的评估：

1. **扰动型**（deletion / insertion / AOPC）：按归因排序逐步删除重要位置，
   观察预测下降速度。忠实的方法应在少量删除后就使预测崩溃。
2. **真值回收型**（position AUC / motif recall / hit rate）：用合成数据的
   真值掩码直接检验归因是否指向真正的基序。
3. **稳定性**（跨方法 Spearman、跨种子一致性）：好方法不应剧烈抖动。

每个指标都返回可序列化的 dict，便于直接进入报告与论文表格。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ..attribution.base import normalize_scores, position_scores
from ..motifs.pwm import DEFAULT_BACKGROUND


# ---------------------------------------------------------------- 扰动型指标
def _replace_with_background(x: np.ndarray, positions: np.ndarray, background: np.ndarray) -> np.ndarray:
    """把指定位置替换为按背景分布采样（或用背景自身）的"中性"碱基。

    这里采用"置零 + 均匀化"的确定性做法（把该位置设为 4 个通道各等于
    背景频率），避免引入随机性，保证指标可复现。
    """
    out = x.copy()
    bg = background / background.sum()
    out[..., positions, :] = bg.reshape(1, 1, 4)
    return out


def deletion_curve(
    model,
    onehot: np.ndarray,
    scores: np.ndarray,
    steps: int = 20,
    background: np.ndarray | None = None,
    batch_size: int = 256,
) -> dict:
    """按归因分数从高到低逐步"置为背景"，记录预测概率的下降。

    Returns
    -------
    dict 含 ``fractions``（被删除比例）与 ``probs``（平均预测概率）。
    """
    bg = DEFAULT_BACKGROUND if background is None else np.asarray(background, dtype=np.float64)
    x = onehot.astype(np.float32)
    n, length, _ = x.shape
    order = np.argsort(-scores, axis=1)  # 高->低

    base_prob = model.predict_proba(x)[:, 1]
    fractions = np.linspace(0, 1, steps + 1)
    probs = [float(base_prob.mean())]

    for frac in fractions[1:]:
        k = max(1, int(round(frac * length)))
        # 批量构造被删除的输入
        variants = []
        for i in range(n):
            idx = order[i, :k]
            out = x[i].copy()
            out[idx, :] = bg.reshape(1, 4)
            variants.append(out)
        arr = np.stack(variants, axis=0)
        p = np.empty(n, dtype=np.float64)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            p[s:e] = model.predict_proba(arr[s:e])[:, 1]
        probs.append(float(p.mean()))

    return {"fractions": fractions.tolist(), "probs": probs}


def insertion_curve(
    model,
    onehot: np.ndarray,
    scores: np.ndarray,
    steps: int = 20,
    background: np.ndarray | None = None,
    batch_size: int = 256,
) -> dict:
    """从全背景出发，按归因从高到低逐步恢复真实碱基，记录预测上升。"""
    bg = DEFAULT_BACKGROUND if background is None else np.asarray(background, dtype=np.float64)
    x = onehot.astype(np.float32)
    n, length, _ = x.shape
    bgv = (bg / bg.sum()).reshape(4)
    order = np.argsort(-scores, axis=1)

    base_arr = np.broadcast_to(bgv.reshape(1, 1, 4), x.shape).astype(np.float32).copy()
    fractions = np.linspace(0, 1, steps + 1)
    probs = []

    for frac in fractions:
        k = int(round(frac * length))
        arr = base_arr.copy()
        for i in range(n):
            idx = order[i, :k]
            arr[i, idx, :] = x[i, idx, :]
        p = np.empty(n, dtype=np.float64)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            p[s:e] = model.predict_proba(arr[s:e])[:, 1]
        probs.append(float(p.mean()))

    return {"fractions": fractions.tolist(), "probs": probs}


def aopc(deletion: dict) -> float:
    """Area Over the Perturbation Curve（**越大越忠实**）。

    定义
    ----
    把删除曲线归一化为 p_norm(f)，其中 p_norm(0)=1。随机删除的期望曲线是
    从 1 线性降到 p_norm(1)，其面积为

        A_random = (1 + p_norm(1)) / 2

    忠实的方法删除高分区后预测下降得**更快**，曲线面积更小，因此

        AOPC = A_random - ∫_0^1 p_norm(f) df

    AOPC > 0 表示优于随机删除；< 0 表示比随机更差（归因方向错误或模型
    根本没学到信号）。本实现使用梯形积分。
    """
    f = np.asarray(deletion["fractions"], dtype=np.float64)
    p = np.asarray(deletion["probs"], dtype=np.float64)
    if p.size < 2 or p[0] <= 0:
        return 0.0
    p_norm = p / p[0]
    # numpy>=2.0 移除了 np.trapz
    trap = getattr(np, "trapezoid", None) or np.trapz
    area = float(trap(p_norm, f))
    area_random = float((1.0 + p_norm[-1]) / 2.0)
    return float(area_random - area)


def sufficiency(insertion: dict) -> float:
    """充分性：只保留少量高归因位置时已恢复的预测比例（越大越忠实）。"""
    p = np.asarray(insertion["probs"], dtype=np.float64)
    if p[-1] <= 0:
        return 0.0
    # 取恢复到 20% 位置时的相对预测值
    idx = int(round(0.2 * (len(p) - 1)))
    return float(p[idx] / p[-1])


# ------------------------------------------------------------ 真值回收型指标
def position_auc(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """逐序列计算位置级 ROC-AUC：归因分数能否区分"真值基序内/外"。

    使用 Mann-Whitney U 统计量的等价公式，无需 sklearn，速度快：
        AUC = (sum_of_ranks_pos - n_pos(n_pos+1)/2) / (n_pos * n_neg)
    """
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(truth).astype(bool)
    n = s.shape[0]
    out = np.full(n, np.nan)

    for i in range(n):
        pos = s[i][t[i]]
        neg = s[i][~t[i]]
        if pos.size == 0 or neg.size == 0:
            continue
        # 平均秩（处理并列）
        order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
        ranks = np.empty(pos.size + neg.size, dtype=np.float64)
        ranks[order] = np.arange(1, pos.size + neg.size + 1)

        # 处理并列：相同值取平均秩
        allv = np.concatenate([pos, neg])
        uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            sums = np.zeros(uniq.size)
            np.add.at(sums, inv, ranks)
            avg = sums / counts
            ranks = avg[inv]

        u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0
        out[i] = u / (pos.size * neg.size)
    return out


def motif_topk_hit(
    scores: np.ndarray,
    truth: np.ndarray,
    k_fraction: float = 0.1,
) -> np.ndarray:
    """Top-k% 归因位置落在真值基序内的比例。随机基线约等于真值占比。"""
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(truth).astype(bool)
    n, length = s.shape
    k = max(1, int(round(k_fraction * length)))
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        top = np.argpartition(-s[i], k - 1)[:k]
        out[i] = float(t[i][top].sum()) / k
    return out


def peak_inside_truth(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """归因峰值位置是否落在真值区域内（0/1）。"""
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(truth).astype(bool)
    peak = s.argmax(axis=1)
    return t[np.arange(s.shape[0]), peak].astype(np.float64)


def recovery_report(scores: np.ndarray, truth: np.ndarray) -> dict:
    """汇总真值回收相关指标。"""
    auc = position_auc(scores, truth)
    topk = motif_topk_hit(scores, truth, k_fraction=0.1)
    peak = peak_inside_truth(scores, truth)
    return {
        "position_auc_mean": float(np.nanmean(auc)),
        "position_auc_std": float(np.nanstd(auc)),
        "position_auc_values": auc.tolist(),
        "topk_hit_mean": float(np.mean(topk)),
        "peak_inside_rate": float(np.mean(peak)),
        "truth_fraction": float(np.mean(truth)),
    }


# ---------------------------------------------------------------- 稳定性指标
def method_agreement(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """两种方法在每条序列上的 Spearman 相关，返回平均值。"""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    vals = []
    for i in range(a.shape[0]):
        if np.std(a[i]) < 1e-12 or np.std(b[i]) < 1e-12:
            continue
        rho, _ = spearmanr(a[i], b[i])
        if np.isfinite(rho):
            vals.append(float(rho))
    return float(np.mean(vals)) if vals else float("nan")


def rank_stability(scores_a: np.ndarray, scores_b: np.ndarray, k_fraction: float = 0.1) -> float:
    """两种方法 Top-k% 位置的重合度（Jaccard），衡量排序稳定性。"""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    n, length = a.shape
    k = max(1, int(round(k_fraction * length)))
    vals = []
    for i in range(n):
        ta = set(np.argpartition(-a[i], k - 1)[:k].tolist())
        tb = set(np.argpartition(-b[i], k - 1)[:k].tolist())
        union = ta | tb
        vals.append(len(ta & tb) / len(union) if union else 0.0)
    return float(np.mean(vals))


def evaluate_attribution(
    model,
    onehot: np.ndarray,
    truth: np.ndarray | None,
    attr: np.ndarray,
    background: np.ndarray | None = None,
    curve_steps: int = 20,
) -> dict:
    """一次性算出某个归因方法的全部指标。

    返回 dict 可直接写入 CSV/JSON，或渲染进 HTML 报告。
    """
    pos = position_scores(attr)
    result: dict = {"n_sequences": int(onehot.shape[0])}

    dele = deletion_curve(model, onehot, pos, steps=curve_steps, background=background)
    ins = insertion_curve(model, onehot, pos, steps=curve_steps, background=background)
    result["aopc"] = aopc(dele)
    result["sufficiency"] = sufficiency(ins)
    result["deletion_curve"] = dele
    result["insertion_curve"] = ins

    if truth is not None:
        result.update(recovery_report(pos, truth))

    # 归一化后保存少量示例，供报告中画热图
    result["example_scores"] = normalize_scores(pos[:5]).tolist()
    return result
