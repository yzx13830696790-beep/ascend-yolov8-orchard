"""帕累托前沿：把 7 种方案的"时延-mAP"画成一张能讲故事的图。

规划书 4.4 节：图上要能指着任意两点说"从这里到这里省了多少 ms、
掉了多少 pt"。这里提供：
  * is_dominated / pareto_front 的严格判定
  * 用 matplotlib(Agg) 画图，`measured=True` 画实心点、False 画空心点
    —— 没有实测的数据点不允许画成实心，这是对"性能数字无口径"的纠正。
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 字体在 orch.fonts 里按运行环境探测：Windows 用雅黑，Linux 容器没有 CJK
# 字体时自动改用英文标签，避免"图能出、字全是方块"。
from .fonts import t, safe, HAS_CJK


def _ascii_label(p, idx):
    """无中文字体时给数据点选一个 ASCII 标签。

    优先用 name_en（显式声明的英文名）；没有就退回 scheme-N，
    而不是把中文硬塞给 DejaVu —— 那只会渲染成一串方块。
    """
    if HAS_CJK:
        return p["name"]
    en = (p.get("name_en") or "").strip()
    if en:
        return en
    try:
        p["name"].encode("ascii")
        return p["name"]
    except (UnicodeEncodeError, AttributeError):
        return "scheme-%d" % (idx + 1)


def load_points(path):
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pts.append({
                "name": row["name"],
                "name_en": (row.get("name_en") or ""),
                "latency": float(row["latency_ms"]),
                "map": float(row["map50_95"]),
                "measured": row.get("measured", "0") in ("1", "true", "True"),
            })
    return pts


def is_dominated(a, b):
    """a 是否被 b 支配（b 既更快又更准）。"""
    return b["latency"] <= a["latency"] and b["map"] >= a["map"] and \
        (b["latency"] < a["latency"] or b["map"] > a["map"])


def pareto_front(points):
    return [p for p in points if not any(is_dominated(p, q) for q in points)]


# 标签相对数据点的偏移（单位：point）。同一邻域内按档位依次取用，
# 垂直间距 22pt 大于 8pt 字号的行高，保证不会互相压字。
_LABEL_OFFSETS = ((6, 10), (6, -16), (6, 30), (6, -36), (6, 50), (6, -56))


def plot(points, out_png, title=None, latency_budget=None,
         map_budget=None):
    front = pareto_front(points)
    front_names = {p["name"] for p in front}
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=130)
    _placed = []
    _lats = [p["latency"] for p in points] or [0.0]
    _maps = [p["map"] for p in points] or [0.0]
    xlo, xspan = min(_lats), (max(_lats) - min(_lats))
    yspan = max(_maps) - min(_maps)
    # 顶部留 15% 余量：mAP 相同的方案会叠在最上面，标签需要空间
    # 上下都留余量：标签可能被推到点的上方或下方（最多 56pt），
    # 不留白就会顶到标题或被 x 轴裁掉。
    _pad = max(0.1, yspan)
    ax.set_ylim(min(_maps) - 0.20 * _pad, max(_maps) + 0.20 * _pad)
    for i, p in enumerate(points):
        m = "o" if p["measured"] else "^"
        c = "#d62728" if p["measured"] else "#9e9e9e"
        alpha = 1.0 if p["measured"] else 0.55
        ax.scatter(p["latency"], p["map"], marker=m, s=80, color=c,
                   alpha=alpha, zorder=3)
        # 多个方案常挤在一起（同一 mAP 水平，或时延相近），标签会互相压住。
        # 给每个点分配一个"邻域内未被占用"的最小档位（贪心着色），
        # 档位决定标签的垂直偏移 —— 相邻标签因此不会落在同一行。
        nx = 0.12 * xspan if xspan > 0 else 0.0
        ny = 0.06 * yspan if yspan > 0 else 0.0
        used = {lvl for (qx, qy, lvl) in _placed
                if abs(qx - p["latency"]) <= nx and abs(qy - p["map"]) <= ny}
        k = 0
        while k in used:
            k += 1
        _placed.append((p["latency"], p["map"], k))
        dx, dy = _LABEL_OFFSETS[k % len(_LABEL_OFFSETS)]
        # 靠右边的点改成左对齐，否则标签会顶出画布被裁掉
        ha = "left"
        if xspan > 0 and p["latency"] > xlo + 0.85 * xspan:
            ha, dx = "right", -abs(dx)
        ax.annotate(_ascii_label(p, i), (p["latency"], p["map"]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=8, color="#333333", ha=ha,
                    annotation_clip=False)
    if front:
        fs = sorted(front, key=lambda p: p["latency"])
        ax.plot([p["latency"] for p in fs], [p["map"] for p in fs],
                ls="--", lw=1.2, color="#7f7f7f", zorder=2,
                label=t("帕累托前沿", "Pareto front"))
    if latency_budget:
        ax.axvline(latency_budget, color="#d62728", ls=":", lw=1.2)
        ax.text(latency_budget, ax.get_ylim()[0],
                t(" 时延预算", " latency budget"),
                fontsize=8, color="#d62728")
    if map_budget:
        ax.axhline(map_budget, color="#d62728", ls=":", lw=1.2)
    ax.set_xlabel(t("端到端 p50 时延 (ms)", "end-to-end p50 latency (ms)"))
    ax.set_ylabel("mAP50-95")
    ax.set_title(safe(title or t("精度-时延帕累托前沿",
                                 "accuracy-latency Pareto front"),
                      "accuracy-latency Pareto front"),
                 fontsize=11)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return {"front": [p["name"] for p in front], "png": out_png}
