"""时延剖析：把"端到端 N ms"拆成六段可归因的预算。

两个输入源：
  1. 应用层阶段打点 CSV（columns: stage,latency_ms），pipeline 每帧记录一段；
  2. msprof 的 op_summary.csv（columns 含 Op Name / Task Duration(us)）。

六段预算（规划书 4.3 节）：
  decode_dvpp / h2d_copy / model_infer / d2h_copy / postprocess_nms / ros_publish
"""
from __future__ import annotations

import csv
import io
import os

import numpy as np

STAGES = ["decode_dvpp", "h2d_copy", "model_infer", "d2h_copy", "postprocess_nms", "ros_publish"]

# msprof op_summary 的 Op Type 前缀 -> 六段归并
OP_TYPE_MAP = [
    (("Vpc", "Dvpp", "Vdec", "Jpeg", "Png"), "decode_dvpp"),
    (("MemcpyH2D", "Copy", "H2D"), "h2d_copy"),
    (("MemcpyD2H", "D2H"), "d2h_copy"),
    (("Nms", "NMS", "RoiAlign", "Yolo"), "postprocess_nms"),
    (("Send", "Publish", "Roi"), "ros_publish"),
]


def _map_stage(op_type, op_name=""):
    """先看方向性关键字（msprof 的 Memcpy 常靠 op name 区分方向），再按 OP Type 归并。"""
    s = (op_type + " " + op_name).lower()
    if any(k in s for k in ("h2d", "hosttodevice", "host_to_device")):
        return "h2d_copy"
    if any(k in s for k in ("d2h", "devicetohost", "device_to_host")):
        return "d2h_copy"
    for keys, stage in OP_TYPE_MAP:
        for k in keys:
            if op_type.lower().startswith(k.lower()):
                return stage
    return "model_infer"


def load_stage_csv(path):
    """应用层阶段打点 CSV -> dict(stage -> np.array ms)。"""
    with open(path, "r", encoding="utf-8") as f:
        return parse_stage_csv(f)


def parse_stage_csv(text_or_file):
    """同 load_stage_csv，但输入是文本或文件对象。

    云端上传的 CSV 走 HTTP body 进来，没有本地路径可传，
    所以解析逻辑只能有一份、入口允许内存文本。
    """
    f = (io.StringIO(text_or_file) if isinstance(text_or_file, str)
         else text_or_file)
    out = {s: [] for s in STAGES}
    for row in csv.DictReader(f):
        s = row.get("stage", "").strip()
        if s not in out:
            continue
        out[s].append(float(row["latency_ms"]))
    return {s: np.asarray(v, dtype=np.float64) for s, v in out.items() if v}


def load_msprof_op_summary(path):
    """msprof op_summary.csv -> (dict(stage -> np.array ms), 时延列名)。

    自动识别 'Task Duration(us)' 或 'Duration(us)' 列。
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return parse_msprof_op_summary(f)


def parse_msprof_op_summary(text_or_file):
    """同 load_msprof_op_summary，输入为文本或文件对象。"""
    f = (io.StringIO(text_or_file) if isinstance(text_or_file, str)
         else text_or_file)
    out = {s: [] for s in STAGES}
    reader = csv.DictReader(f)
    fields = reader.fieldnames or []
    op_col = "Op Type" if "Op Type" in fields else "OP Type"
    dur_col = None
    for cand in ("Task Duration(us)", "Duration(us)"):
        if cand in fields:
            dur_col = cand
            break
    if dur_col is None or op_col not in fields:
        return out, None
    op_name_col = "Op Name" if "Op Name" in fields else None
    for row in reader:
        try:
            us = float(row[dur_col])
        except (ValueError, TypeError):
            continue
        op_name = row.get(op_name_col, "") if op_name_col else ""
        out[_map_stage(row[op_col], op_name)].append(us / 1000.0)
    return {s: np.asarray(v, dtype=np.float64) for s, v in out.items() if v}, dur_col


def summarize(latency_dict):
    """latency_dict: stage -> array ms。返回每段 p50/p90/p99 与总量。"""
    rows = {}
    total = np.zeros(0)
    for s in STAGES:
        a = np.asarray(latency_dict.get(s, []), dtype=np.float64)
        if a.size:
            rows[s] = {"p50": float(np.percentile(a, 50)),
                       "p90": float(np.percentile(a, 90)),
                       "p99": float(np.percentile(a, 99)),
                       "mean": float(a.mean()), "n": int(a.size)}
            total = np.concatenate([total, a]) if total.size else a
        else:
            rows[s] = {"p50": None, "p90": None, "p99": None, "mean": None, "n": 0}
    end_to_end = {
        "p50": float(np.percentile(total, 50)) if total.size else None,
        "p90": float(np.percentile(total, 90)) if total.size else None,
        "p99": float(np.percentile(total, 99)) if total.size else None,
    }
    return {"stages": rows, "end_to_end": end_to_end}


def build_budget_table(summary):
    rows = []
    for s in STAGES:
        r = summary["stages"][s]
        rows.append({"stage": s, "p50_ms": r["p50"], "p90_ms": r["p90"],
                     "p99_ms": r["p99"], "mean_ms": r["mean"]})
    return rows
