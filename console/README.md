# console —— 模型评测与量化实验平台

零第三方依赖的控制台：`http.server` + `sqlite3` + `matplotlib`，标准库级别。
**能在离线的昇腾机器上直接跑**（不需要 pip install flask/fastapi，也不需要 npm）。

```bash
python console/server.py --port 8765     # 打开 http://127.0.0.1:8765
```

## 为什么不用 Flask / FastAPI

本机（以及多数昇腾离线环境）没有外网。平台的定位是"实验管理与度量看板"，
不是高并发 Web 服务，标准库足够；换来的收益是**拷过去就能跑**，不用为核心
工具链再维护一套 Python 依赖。这是工程取舍，不是能力妥协：
- 并发：`ThreadingHTTPServer`，评测任务丢后台线程，前端轮询
- 持久化：SQLite 单文件，随仓库走，不依赖数据库服务
- 图表：matplotlib(Agg) 出 PNG，前端 `<img>` 引用，不需要 CDN 或 JS 图表库

## 功能（全部可点、可跑）

| 页 | 能力 | 接口 |
|---|---|---|
| 概览 | 目标约束 vs 最新实测；达标/超标判定 | `GET /api/overview` |
| 实验 | 发起评测（异步）、列表轮询、查看报告 | `POST /api/runs` · `GET /api/runs` · `GET /api/runs/<id>/report` |
| 帕累托 | 实测点（实心）+ 待实测占位（空心） | `GET /figures/pareto.png` |
| 时延预算 | msprof 入料 → 六段条形图 | `POST /api/budget/ingest` · `GET /figures/budget.png` |
| 台账 | 每次评测自动追加、可下载 CSV | `GET /api/ledger.csv` |

## msprof 入料约定

把 `msprof` 导出的 `op_summary.csv`（或应用层阶段打点 csv，列 `stage,latency_ms`）
放进 `console/inbox/`，点「扫描 inbox 入库」。解析规则在 `orch/latency.py`：
按 OP Type + Op Name 归并到六段（decode_dvpp / h2d_copy / model_infer /
d2h_copy / postprocess_nms / ros_publish），一次入料**幂等**，重复点不会重复入库。

## 数据落盘

- `console/console.db`：runs / budgets
- `results/runs/<id>.json`：每次实验的完整明细（含每类 AP、业务指标、六段时延）
- `results/ledger.csv`：append-only 实验台账
- `results/figures/platform_pareto.png`、`platform_budget.png`：平台渲染图

## 从"能跑"到"真落地"还差什么

平台已是可运行的最小闭环，若要放到团队里长期用，按顺序补：

1. **鉴权**：单用户本机 → 加 token（`Authorization` 头）或反代到企业 SSO
2. **任务队列**：后台线程 → 落盘队列 + 重启恢复（当前 run 状态在 SQLite，进程重启后排队任务不会自动续跑）
3. **真实模型接入**：把 `jobs.execute_eval` 的 `toy_detector` 换成 OM/ONNX 推理结果（读 predictions.jsonl 即可，评测协议不用改）
4. **多机**：`results/` 挂共享存储，各昇腾机器只跑任务、平台只做看板
5. **部署**：`docker run -p 8765:8765`，镜像复用根目录 Dockerfile（CANN 已固定）

## 已知边界（对外陈述不要越线）

- mAP 来自经典 CV 占位检测器，**无业务含义**，仅证明评测协议与后处理正确
- 帕累托图上的空心点是任务书口径占位，不是实测；实测后应删除占位并补实心点
- 排队中的任务在服务重启后不会自动续跑（无持久化队列）

## 云服务接入（WorkBuddy Cloud Service）

平台原本用本地 SQLite 存实验与预算，换台机器就丢。现在接入了云服务的三个模块：

| 模块 | 用途 | 位置 |
|---|---|---|
| Auth | 手机验证码 / 微信扫码登录，拿到 end-user 身份 | `index.html` 的 `restoreAuth` / `sendCode` / `wechatLogin` |
| Database | 实验记录 `orchard_runs`、时延预算 `orchard_budgets` | `loadCloudRuns` / `insertRun` / `loadCloudBudgets` |
| Storage | msprof CSV 归档到 `users/<uid>/msprof/...` | `#budgetFile` 的 onchange |

### 为什么计算还在 Python 侧

评测、六段时延解析、出图仍然是服务端在跑（`orch/*`），云服务只承担
**身份 + 持久化 + 文件归档**。这样解析逻辑只有一份（`orch/latency.py`），
不会因为"前端也要解析一遍"而分叉。前端拿到结果后写云端。

### 数据模型与安全边界

两张表都是 **公共可读、仅本人可写**：

- `SELECT` 策略 `USING (true)` —— 实验看板不需要登录就能看，这是刻意的团队共享设计。
- `INSERT` 策略 `WITH CHECK (owner_id = auth.uid())`，`owner_id` 默认由 `DEFAULT auth.uid()` 填，前端从不发送它。
- **授权只给 `authenticated`，不给 `anon`**（见下面的坑）。

### 坑：匿名请求的 auth.uid() 是字符串 'anon'

一开始按 `management.md` 的通用写法把 `INSERT, DELETE` 同时授权给 `authenticated, anon`，
再用真实 SDK 一测：**未登录也能插入**，落库的 `owner_id` 是字符串 `"anon"`。
原因是 `WITH CHECK (owner_id = auth.uid())` 在匿名请求下退化成 `owner_id = 'anon'`，
而 `DEFAULT auth.uid()` 恰好也填 `'anon'` —— 两个条件同时成立，策略形同虚设。
修法是 `REVOKE INSERT, DELETE ... FROM anon`，只给 `authenticated`。
修完同样的探测返回 `DATABASE_42501 permission denied`。

教训：**"策略写对了"不等于"权限配对了"**。GRANT 和 POLICY 是两道独立的门，
只有拿真实凭据打一次才能确认关没关。

### 验证脚本

```bash
# 1) 用官方 SDK（Node 里加载同一份 IIFE 产物）打数据面：确认匿名读通、匿名写被拒
WB_PK=<publishableKey> node scripts/probe_cloud.js

# 2) 真浏览器跑页面，点登录闸门 / 校验手机号格式 / 看云端表格是否渲染
node scripts/verify_cloud_ui.mjs     # 需先起 headless chrome --remote-debugging-port=9222
```

### SDK 为什么是自托管的

`console/static/vendor/workbuddy-cloud-sdk.js` 是官方 `@tencent-ai/workbuddy-cloud-sdk@dev`
的 IIFE 构建产物（版本见同目录 `*.package.json`），没有走 jsdelivr：

- 这台机器和容器访问不到 jsdelivr（SSL 直接失败），而本项目本来就要求离线可跑；
- 依赖外部 CDN 意味着断网时页面白屏，而白屏是最难排查的故障形态。

仍然是 Form B（IIFE 全局 `WorkBuddyCloud`），只是把同一份产物放到自己的静态目录。
升级时重新从 npm `dev` 标签取 `lib/index.global.js` 覆盖即可。
