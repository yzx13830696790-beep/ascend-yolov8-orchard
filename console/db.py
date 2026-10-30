"""平台数据层：SQLite（标准库），存实验 run 与时延预算。

只依赖 sqlite3，保证在离线的昇腾机器上也能跑。
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "console", "console.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    detector    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'queued',
    map50       REAL,
    map50_95    REAL,
    p50_ms      REAL,
    p99_ms      REAL,
    n_val       INTEGER,
    metrics     TEXT,
    error       TEXT,
    created_at  TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS budgets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    p50_ms      REAL,
    p99_ms      REAL,
    stages      TEXT,
    created_at  TEXT
);
"""


def connect():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with connect() as c:
        c.executescript(SCHEMA)


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def create_run(name, detector="toy", conf_thr=0.25):
    with connect() as c:
        cur = c.execute(
            "INSERT INTO runs(name, detector, status, created_at) VALUES(?,?,?,?)",
            (name, detector, "queued", _now()))
        return cur.lastrowid


def update_run(run_id, **kw):
    if not kw:
        return
    cols = ", ".join(f"{k}=?" for k in kw)
    with connect() as c:
        c.execute(f"UPDATE runs SET {cols} WHERE id=?", list(kw.values()) + [run_id])


def get_run(run_id):
    with connect() as c:
        row = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_runs(limit=50):
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def latest_done_run():
    with connect() as c:
        row = c.execute(
            "SELECT * FROM runs WHERE status='done' ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def insert_budget(name, p50, p99, stages):
    with connect() as c:
        cur = c.execute(
            "INSERT INTO budgets(name, p50_ms, p99_ms, stages, created_at) VALUES(?,?,?,?,?)",
            (name, p50, p99, json.dumps(stages, ensure_ascii=False), _now()))
        return cur.lastrowid


def list_budgets(limit=20):
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM budgets ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["stages"] = json.loads(d["stages"]) if d.get("stages") else {}
        out.append(d)
    return out


def budget_names():
    """已入库的预算文件名集合，用于入料幂等。"""
    with connect() as c:
        return {r["name"] for r in c.execute("SELECT name FROM budgets").fetchall()}


def stats():
    with connect() as c:
        total = c.execute("SELECT COUNT(*) n FROM runs").fetchone()["n"]
        done = c.execute(
            "SELECT COUNT(*) n FROM runs WHERE status='done'").fetchone()["n"]
        running = c.execute(
            "SELECT COUNT(*) n FROM runs WHERE status IN ('queued','running')").fetchone()["n"]
    return {"total": total, "done": done, "running": running}
