"""把 metrics.json / 时延预算 / 帕累托 汇总成 Markdown 报告。"""
from __future__ import annotations

import json
import os


def metrics_md(metrics, title="精度对比"):
    lines = [f"## {title}", ""]
    lines.append(f"- mAP50 = **{metrics['map50']:.4f}**")
    lines.append(f"- mAP50-95 = **{metrics['map50_95']:.4f}**")
    lines.append(f"- GT 数 = {metrics['num_gt']}，评估类别 = {metrics['classes_evaluated']}")
    lines.append("")
    names = metrics.get("class_names") or {}
    lines.append("| 类别 | AP50 | mAP50-95 |")
    lines.append("|---|---|---|")
    for c in sorted(metrics["per_class_ap50"]):
        nm = names.get(str(c), str(c)) if isinstance(names, dict) else c
        lines.append(f"| {nm} | {metrics['per_class_ap50'][c]:.4f} | "
                     f"{metrics['per_class_map'][c]:.4f} |")
    return "\n".join(lines)


def budget_md(summary, title="时延预算（六段拆分）"):
    lines = [f"## {title}", ""]
    e = summary["end_to_end"]
    lines.append(f"- 端到端 p50 = {e['p50']} ms · p90 = {e['p90']} ms · p99 = {e['p99']} ms")
    lines.append("")
    lines.append("| 阶段 | p50(ms) | p90(ms) | p99(ms) | 均值(ms) |")
    lines.append("|---|---|---|---|---|")
    for s, r in summary["stages"].items():
        lines.append(f"| {s} | {r['p50']} | {r['p90']} | {r['p99']} | {r['mean']} |")
    return "\n".join(lines)


def business_md(bm):
    lines = ["## 业务指标（可采摘果实，conf=0.3, IoU=0.5）", ""]
    lines.append(f"- 漏检率 = **{bm['miss_rate']:.3f}**（{bm['miss_count']}/{bm['gt_pickable']}）")
    lines.append(f"- 误检率 = **{bm['fp_rate']:.3f}**（{bm['fp_count']}/{bm['pred_count']}）")
    return "\n".join(lines)


def write_report(metrics, summary, business, out_path, pareto_csv=None):
    parts = [f"# 评测报告", "", "> 口径：见 results/metrics_*.json 的 meta 字段。", ""]
    parts.append(metrics_md(metrics))
    parts.append("")
    parts.append(business_md(business))
    parts.append("")
    if summary:
        parts.append(budget_md(summary))
    text = "\n".join(parts) + "\n"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
