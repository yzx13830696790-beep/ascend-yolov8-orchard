"""任务执行器：跑评测、生成图、解析 msprof 预算。

所有耗时操作都可被 HTTP 层放到后台线程里执行；图用 matplotlib(Agg) 落盘，
前端直接 <img> 引用 —— 不依赖任何 CDN，离线可用。
"""
from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orch import metrics as M
from orch import latency as L
from orch import pareto as P
from orch import ledger as LEDGER
from orch.fonts import t as _t
from console import db

FIG_DIR = os.path.join(ROOT, "results", "figures")
RUN_DIR = os.path.join(ROOT, "results", "runs")
INBOX = os.path.join(ROOT, "console", "inbox")
_MPL_LOCK = threading.Lock()

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _load_cfg():
    with open(os.path.join(ROOT, "configs", "constraints.json"), encoding="utf-8") as f:
        return json.load(f)


def _load_val():
    with open(os.path.join(ROOT, "data", "meta", "gts.json"), encoding="utf-8") as f:
        gts = json.load(f)
    manifest = {}
    with open(os.path.join(ROOT, "data", "meta", "manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            manifest[d["image_id"]] = d
    return {iid: g for iid, g in gts.items() if g["split"] == "val"}, manifest


# --------------------------------------------------------------------------
# 评测任务
# --------------------------------------------------------------------------
def execute_eval(run_id, detector="toy", conf_thr=None):
    cfg = _load_cfg()
    conf_thr = cfg["conf_thr"] if conf_thr is None else conf_thr
    try:
        db.update_run(run_id, status="running")
        gts, manifest = _load_val()
        import cv2
        from toy_detector import detect_as_dict

        times = []
        preds = []
        t_start = time.perf_counter()
        for image_id in sorted(gts.keys()):
            bgr = cv2.imread(manifest[image_id]["image_path"])
            t0 = time.perf_counter()
            pred = detect_as_dict(image_id, bgr, conf_thr)
            times.append((time.perf_counter() - t0) * 1000.0)
            preds.append(pred)

        m = M.evaluate(preds, gts, class_names=cfg["class_names"])
        bm = M.business_metrics(preds, gts, conf=cfg["business_conf"],
                                iou=cfg["business_iou"], min_visible=cfg["min_visible"])
        lat = L.summarize({"model_infer": times, "postprocess_nms": []})
        wall = (time.perf_counter() - t_start) * 1000.0

        detail = {
            "run_id": run_id,
            "detector": detector,
            "conf_thr": conf_thr,
            "meta": {"n_val": len(gts), "note": "toy 检测器仅校验评测协议，mAP 无业务含义"},
            "metrics": m,
            "business": bm,
            "latency": lat,
            "wall_ms": wall,
        }
        os.makedirs(RUN_DIR, exist_ok=True)
        with open(os.path.join(RUN_DIR, "%d.json" % run_id), "w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)

        LEDGER.append(os.path.join(ROOT, "results", "ledger.csv"),
                      exp_id="run_%d" % run_id, change="平台触发评测",
                      model=detector, map50_95=round(m["map50_95"], 4),
                      map50=round(m["map50"], 4), p50_ms=lat["end_to_end"]["p50"],
                      conclusion="toy harness")

        db.update_run(run_id, status="done",
                      map50=round(m["map50"], 4), map50_95=round(m["map50_95"], 4),
                      p50_ms=round(lat["end_to_end"]["p50"] or 0, 3),
                      p99_ms=round(lat["end_to_end"]["p99"] or 0, 3),
                      n_val=len(gts),
                      metrics=json.dumps(detail, ensure_ascii=False),
                      finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:  # noqa: BLE001
        db.update_run(run_id, status="error", error=str(e),
                      finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))


def start_eval(name, detector="toy", conf_thr=None):
    run_id = db.create_run(name, detector, conf_thr if conf_thr is not None else 0.25)
    t = threading.Thread(target=execute_eval, args=(run_id, detector, conf_thr), daemon=True)
    t.start()
    return run_id


# --------------------------------------------------------------------------
# 图：帕累托 / 时延预算
# --------------------------------------------------------------------------
# name_en：容器里没有中文字体时用它，避免标签退化成 scheme-N
PENDING = [
    {"name": "GPU FP32（任务书口径）", "name_en": "GPU FP32 (doc)", "latency": 42, "map": 0.86},
    {"name": "OM FP16(AIPP)", "name_en": "OM FP16 (AIPP)", "latency": 22, "map": 0.86},
    {"name": "OM FP16+流水", "name_en": "OM FP16+pipeline", "latency": 16, "map": 0.86},
    {"name": "OM INT8 PTQ", "name_en": "OM INT8 PTQ", "latency": 13, "map": 0.85},
    {"name": "OM INT8 混合精度", "name_en": "INT8 mixed-prec.", "latency": 11, "map": 0.85},
]


def make_pareto_png():
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "platform_pareto.png")
    points = []
    for r in db.list_runs(200):
        if r["status"] == "done" and r["map50_95"] is not None and r["p50_ms"]:
            points.append({"name": "run#%d %s" % (r["id"], r["name"][:12]),
                           "latency": r["p50_ms"], "map": r["map50_95"],
                           "measured": True})
    for p in PENDING:
        points.append(dict(p, measured=False))
    if not points:
        return None
    with _MPL_LOCK:
        P.plot(points, out,
               title=_t("精度-时延帕累托前沿（实心=实测，空心=待实测）",
                        "accuracy-latency Pareto front (filled=measured)"),
               latency_budget=30)
    return out


def make_budget_png(stages):
    """stages: {stage: {p50,...}} -> 六段条形图。"""
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "platform_budget.png")
    names = [s for s in L.STAGES if stages.get(s, {}).get("p50") is not None]
    vals = [stages[s]["p50"] for s in names]
    with _MPL_LOCK:
        fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=130)
        if names:
            bars = ax.bar(names, vals, color="#2f6f4e")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, "%.2f" % v,
                        ha="center", va="bottom", fontsize=8)
            ax.set_ylabel("p50 (ms)")
        else:
            ax.text(0.5, 0.5, _t("暂无时延数据：把 msprof op_summary.csv 放进 console/inbox/",
                                 "no latency data: drop msprof op_summary.csv into console/inbox/"),
                    ha="center", va="center", fontsize=10, color="#888")
            ax.set_xticks([])
        ax.set_title(_t("端到端时延预算（六段拆分）",
                        "end-to-end latency budget (6 stages)"), fontsize=11)
        ax.grid(True, axis="y", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return out


# --------------------------------------------------------------------------
# msprof 入料
# --------------------------------------------------------------------------
def ingest_inbox():
    """扫描 inbox 下的 csv，解析成六段预算并入库（幂等：同名文件只入一次）。"""
    os.makedirs(INBOX, exist_ok=True)
    seen = db.budget_names()
    n = 0
    for fn in sorted(os.listdir(INBOX)):
        if not fn.lower().endswith(".csv") or fn in seen:
            continue
        path = os.path.join(INBOX, fn)
        data, _ = L.load_msprof_op_summary(path)
        if not data:
            data = L.load_stage_csv(path)
        if not data:
            continue
        s = L.summarize(data)
        e2e = s["end_to_end"]
        db.insert_budget(fn, e2e["p50"], e2e["p99"], s["stages"])
        n += 1
    return n


def parse_and_store_budget(name, content):
    """解析前端上传的 CSV 文本，落本地库并返回六段摘要。

    和 ingest_inbox 共用同一份解析器（orch.latency），
    区别只是输入来自 HTTP body 而不是磁盘文件。
    返回 (summary, error)：error 为 None 表示解析成功。
    """
    data, _ = L.parse_msprof_op_summary(content)
    if not data:
        data = L.parse_stage_csv(content)
    if not data:
        return None, "无法识别的 CSV：既不是 msprof op_summary，也不是阶段打点表"
    s = L.summarize(data)
    e2e = s["end_to_end"]
    db.insert_budget(name, e2e["p50"], e2e["p99"], s["stages"])
    return s, None
