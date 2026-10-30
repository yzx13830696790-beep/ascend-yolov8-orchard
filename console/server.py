"""零依赖 HTTP 平台（标准库 http.server + sqlite3），离线可跑。

启动：python platform/server.py --port 8765
接口：
  GET  /                       单页前端
  GET  /api/overview           目标约束 / 统计 / 最新 run
  GET  /api/runs               实验列表
  POST /api/runs               新建并异步执行评测  {"name": "...", "conf_thr": 0.25}
  GET  /api/runs/<id>          单次实验详情
  GET  /api/runs/<id>/report   该次实验的 Markdown 报告
  GET  /api/ledger.csv         实验台账导出
  GET  /api/budget             时延预算列表
  POST /api/budget/ingest      扫描 inbox/ 下的 csv 并入库
  GET  /figures/pareto.png     帕累托图
  GET  /figures/budget.png     六段预算条形图
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from console import db, jobs
from orch import report as R

STATIC = os.path.join(ROOT, "console", "static")


def _json(obj, status=200):
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return status, "application/json; charset=utf-8", body


def _text(s, status=200, ctype="text/plain; charset=utf-8"):
    return status, ctype, s.encode("utf-8")


def api_overview(_q):
    with open(os.path.join(ROOT, "configs", "constraints.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    latest = db.latest_done_run()
    return _json({
        "targets": cfg["budget"],
        "stats": db.stats(),
        "latest": latest,
        "budgets": db.list_budgets(5),
    })


def api_runs(_q):
    return _json({"runs": db.list_runs(50)})


def api_create_run(q, body):
    name = (body.get("name") or "").strip() or "run"
    conf = float(body.get("conf_thr") or 0.25)
    run_id = jobs.start_eval(name, "toy", conf)
    return _json({"id": run_id, "status": "queued"}, 201)


def api_run_detail(q, run_id):
    r = db.get_run(run_id)
    if not r:
        return _json({"error": "not found"}, 404)
    return _json(r)


def api_run_report(q, run_id):
    r = db.get_run(run_id)
    if not r or not r.get("metrics"):
        return _text("# 尚无结果\n", 404)
    detail = json.loads(r["metrics"])
    m = detail["metrics"]
    bm = detail["business"]
    lat = detail["latency"]
    parts = ["# 实验报告：%s（run #%s）" % (r["name"], r["id"]), ""]
    parts.append(R.metrics_md(m))
    parts.append("")
    parts.append(R.business_md(bm))
    parts.append("")
    parts.append(R.budget_md(lat))
    parts.append("")
    parts.append("> %s" % detail["meta"]["note"])
    return _text("\n".join(parts) + "\n", 200, "text/markdown; charset=utf-8")


def api_ledger(_q):
    path = os.path.join(ROOT, "results", "ledger.csv")
    if not os.path.exists(path):
        return _text("exp_id,date\n", 200, "text/csv; charset=utf-8")
    with open(path, "rb") as f:
        return 200, "text/csv; charset=utf-8", f.read()


def api_budget(_q):
    """只读：GET 不应有写副作用，入料统一走 POST /api/budget/ingest。"""
    return _json({"budgets": db.list_budgets(20)})


def api_budget_ingest(_q, body):
    n = jobs.ingest_inbox()
    return _json({"ingested": n})


def api_budget_parse(_q, body):
    """解析前端上传的 CSV 文本（云端上传的文件随请求带内容进来）。

    解析逻辑只有一份，在 orch.latency；这里只负责取名字、落库、回摘要。
    """
    name = (body.get("name") or "").strip() or "upload.csv"
    content = body.get("content") or ""
    if not content.strip():
        return _json({"error": "空文件"}, 400)
    summary, err = jobs.parse_and_store_budget(name, content)
    if err:
        return _json({"error": err}, 400)
    return _json({"name": name, "summary": summary})


def figure_pareto(_q):
    p = jobs.make_pareto_png()
    if not p or not os.path.exists(p):
        return _text("no data", 404)
    with open(p, "rb") as f:
        return 200, "image/png", f.read()


def figure_budget(_q):
    budgets = db.list_budgets(1)
    stages = budgets[0]["stages"] if budgets else {}
    p = jobs.make_budget_png(stages)
    with open(p, "rb") as f:
        return 200, "image/png", f.read()


ROUTES = [
    ("GET", r"^/$", None),                       # 静态首页
    ("GET", r"^/api/overview$", api_overview),
    ("GET", r"^/api/runs$", api_runs),
    ("POST", r"^/api/runs$", api_create_run),
    ("GET", r"^/api/runs/(\d+)$", api_run_detail),
    ("GET", r"^/api/runs/(\d+)/report$", api_run_report),
    ("GET", r"^/api/ledger\.csv$", api_ledger),
    ("GET", r"^/api/budget$", api_budget),
    ("POST", r"^/api/budget/ingest$", api_budget_ingest),
    ("POST", r"^/api/budget/parse$", api_budget_parse),
    ("GET", r"^/figures/pareto\.png$", figure_pareto),
    ("GET", r"^/figures/budget\.png$", figure_budget),
]


class Handler(BaseHTTPRequestHandler):
    server_version = "orchard310b/0.1"

    def log_message(self, fmt, *args):  # 静默访问日志
        pass

    def _send(self, status, ctype, body):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _static(self, path):
        rel = path.lstrip("/")
        full = os.path.normpath(os.path.join(STATIC, rel))
        if not full.startswith(STATIC) or not os.path.isfile(full):
            return self._send(*_text("404", 404))
        ctype = "text/html; charset=utf-8"
        if full.endswith(".js"):
            ctype = "application/javascript; charset=utf-8"
        elif full.endswith(".css"):
            ctype = "text/css; charset=utf-8"
        elif full.endswith(".png"):
            ctype = "image/png"
        with open(full, "rb") as f:
            self._send(200, ctype, f.read())

    def _dispatch(self, method):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        if path in ("/", "/index.html"):
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path[len("/static/"):])

        body = {}
        if method == "POST":
            n = int(self.headers.get("Content-Length") or 0)
            if n:
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    body = {}

        for m, pattern, fn in ROUTES:
            if m != method:
                continue
            mt = re.match(pattern, path)
            if not mt:
                continue
            if fn is None:
                return self._static("index.html")
            groups = [int(g) for g in mt.groups()] if mt.groups() else []
            try:
                if groups:
                    result = fn(query, *groups)
                elif method == "POST":
                    result = fn(query, body)
                else:
                    result = fn(query)
            except Exception as e:  # noqa: BLE001
                result = _json({"error": str(e)}, 500)
            return self._send(*result)
        return self._send(*_text("404 " + path, 404))

    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")


def main():
    ap = argparse.ArgumentParser()
    # 云托管/容器会注入 PORT；本地默认 8765。
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("PORT", "8765")))
    # 默认监听全部网卡，便于局域网与其他机器访问；
    # 只想本机访问时显式传 --host 127.0.0.1。
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    args = ap.parse_args()
    db.init()
    jobs.ingest_inbox()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print("orchard310b platform -> http://%s:%d" % (args.host, args.port), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
