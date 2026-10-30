"""实验台账：exp_id / 改动 / mAP / p50 / 结论 一行一条，append-only。

规划书证据链的第 5 条：随手翻一页实验台账，比任何口头解释都有说服力。
字段固定，避免"实验做完想不起来为什么这么做"。
"""
from __future__ import annotations

import csv
import os
import subprocess
from datetime import datetime

FIELDS = ["exp_id", "date", "git_commit", "change", "model", "map50_95",
          "map50", "p50_ms", "p99_ms", "om_size_mb", "conclusion"]


def git_commit(root=None):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def append(path, **kwargs):
    """追加一行实验。kwargs 里不认识的键被忽略。"""
    row = {k: "" for k in FIELDS}
    row["date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    row["exp_id"] = kwargs.get("exp_id", datetime.now().strftime("%Y%m%d%H%M%S"))
    row["git_commit"] = kwargs.get("git_commit", git_commit())
    for k in FIELDS:
        if k in kwargs and kwargs[k] is not None:
            row[k] = kwargs[k]
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row
