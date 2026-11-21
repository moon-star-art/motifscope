"""基序扫描：在一条序列上注释所有显著命中（双向）。

这是 motifscope 的"实用功能"入口：研究人员可以拿任意 FASTA 序列，
用 JASPAR 基序库找出潜在的转录因子结合位点，并给出经验 p 值。

两个容易被忽略但至关重要的统计细节
----------------------------------
1. **多重检验校正**：若扫描 M 条基序、长度 L 的序列、双链，共做
   ``M * 2 * (L - w + 1)`` 次检验。以 p<1e-3 为阈值时，随机序列上也会出现
   若干"显著"命中。本模块默认对每条序列做 Bonferroni 校正，可与未校正阈值
   对照使用。
2. **重叠命中抑制（NMS）**：相邻窗口往往都超过阈值，但它们描述的是同一个
   结合位点。若不合并，一条 666bp 的序列可报出数百个冗余命中。本模块默认
   按得分做非极大值抑制，只保留局部最优。
"""

from __future__ import annotations

import numpy as np

from ..motifs.pwm import MotifHit, PWM, empirical_pvalue, null_score_distribution


def _nms(hits: list[MotifHit], max_overlap: float = 0.5, merge_gap: int = 2) -> list[MotifHit]:
    """按得分做非极大值抑制，去掉与更高分命中重叠过多的较低分命中。

    Parameters
    ----------
    max_overlap
        允许的最大重叠比例（相对于较短命中的长度）。0 表示完全不重叠，
        1 表示允许完全包含。
    merge_gap
        两个命中之间间隔不超过该值（bp）时也视为同一簇。串联重复的基序
        位点彼此紧邻，需要这一项才能正确合并。
    """
    if not hits:
        return []
    ordered = sorted(hits, key=lambda h: -h.score)
    kept: list[MotifHit] = []
    for h in ordered:
        h_len = h.end - h.start
        drop = False
        for k in kept:
            overlap = min(h.end, k.end) - max(h.start, k.start)
            k_len = k.end - k.start
            if overlap > 0:
                frac = overlap / max(min(h_len, k_len), 1)
                if frac > max_overlap:
                    drop = True
                    break
            # 考虑"紧邻"的情况（间隔 <= merge_gap）
            gap = max(h.start, k.start) - min(h.end, k.end)
            if -merge_gap <= gap <= merge_gap and h.motif_id == k.motif_id:
                drop = True
                break
        if not drop:
            kept.append(h)
    return kept


def scan_sequence(
    seq: str,
    motifs: list[PWM],
    pvalue_threshold: float = 1e-4,
    both_strands: bool = True,
    background: np.ndarray | None = None,
    seed: int = 0,
    correct_multiple_testing: bool = True,
    apply_nms: bool = True,
    nms_max_overlap: float = 0.5,
    use_exact_pvalue: bool = True,
) -> list[MotifHit]:
    """扫描单条序列，返回显著命中，按 p 值升序排列。

    Parameters
    ----------
    correct_multiple_testing
        是否做 Bonferroni 校正。校正后阈值 = ``pvalue_threshold / n_tests``，
        其中 n_tests 为实际执行的检验次数。默认开启，可显著降低随机序列的
        假阳性。
    apply_nms
        是否合并重叠命中。默认开启。
    use_exact_pvalue
        是否使用**精确**的 PWM 得分分布（卷积）计算 p 值。默认开启。
        此时 p 值分辨率可达 1e-15，从而支持 Bonferroni 这类严格校正；
        若关闭则退回经验零分布（分辨率约 1/n_seqs）。
    """
    from ..motifs.exact_pvalue import exact_pvalues as _exact_p
    from ..seq import encode

    x = encode(seq)[None, ...]
    hits: list[MotifHit] = []

    # 先统计检验次数（用于 Bonferroni）
    usable = [m for m in motifs if m.length <= len(seq)]
    n_tests = sum(
        (len(seq) - m.length + 1) * (2 if both_strands else 1) for m in usable
    )
    alpha = pvalue_threshold / max(n_tests, 1) if correct_multiple_testing else pvalue_threshold

    for motif in usable:
        targets = [("+", motif)]
        if both_strands:
            targets.append(("-", motif.reverse_complement()))

        for strand, m in targets:
            scores = m.score_windows(x, background=background)[0]
            if use_exact_pvalue:
                pvals = _exact_p(scores, m, background=background)
            else:
                from ..motifs.pwm import empirical_pvalue, null_score_distribution

                null = null_score_distribution(
                    m, seq_len=min(len(seq), 300), n_seqs=2000, background=background, seed=seed
                )
                pvals = empirical_pvalue(scores, null)
            sel = np.flatnonzero(pvals <= alpha)
            for i in sel:
                hits.append(
                    MotifHit(
                        motif_id=motif.matrix_id,
                        motif_name=motif.name,
                        start=int(i),
                        end=int(i + motif.length),
                        score=float(scores[i]),
                        strand=strand,
                        pvalue=float(pvals[i]),
                    )
                )

    if apply_nms:
        hits = _nms(hits, max_overlap=nms_max_overlap)

    hits.sort(key=lambda h: h.pvalue)
    return hits


def scan_many(
    sequences: dict[str, str],
    motifs: list[PWM],
    pvalue_threshold: float = 1e-4,
    both_strands: bool = True,
    background: np.ndarray | None = None,
    correct_multiple_testing: bool = True,
    apply_nms: bool = True,
    use_exact_pvalue: bool = True,
) -> dict[str, list[MotifHit]]:
    """批量扫描多条序列（例如一个 FASTA 文件）。"""
    return {
        name: scan_sequence(
            seq,
            motifs,
            pvalue_threshold=pvalue_threshold,
            both_strands=both_strands,
            background=background,
            correct_multiple_testing=correct_multiple_testing,
            apply_nms=apply_nms,
            use_exact_pvalue=use_exact_pvalue,
        )
        for name, seq in sequences.items()
    }


def read_fasta(path: str) -> dict[str, str]:
    """极简 FASTA 读取器（不依赖 Biopython）。"""
    seqs: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks).upper()
                name = line[1:].split()[0] if len(line) > 1 else f"seq{len(seqs) + 1}"
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        seqs[name] = "".join(chunks).upper()
    if not seqs:
        raise ValueError(f"未从 {path} 读到任何序列")
    return seqs


def write_bed(hits: dict[str, list[MotifHit]], path: str, seq_lengths: dict[str, int] | None = None) -> None:
    """把扫描结果写成 BED6 格式，便于在基因组浏览器中查看。"""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("#chrom\tstart\tend\tname\tscore\tstrand\n")
        for chrom, hs in hits.items():
            for h in hs:
                fh.write(
                    f"{chrom}\t{h.start}\t{h.end}\t{h.motif_name}({h.motif_id})\t"
                    f"{h.score:.2f}\t{h.strand}\n"
                )
