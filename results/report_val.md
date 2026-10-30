# 评测报告

> 口径：见 results/metrics_*.json 的 meta 字段。

## 精度对比

- mAP50 = **0.1889**
- mAP50-95 = **0.1408**
- GT 数 = 400，评估类别 = [0, 1, 2, 3]

| 类别 | AP50 | mAP50-95 |
|---|---|---|
| apple | 0.6409 | 0.5225 |
| apple_occluded | 0.0450 | 0.0145 |
| branch | 0.0148 | 0.0068 |
| leaf_cluster | 0.0550 | 0.0195 |

## 业务指标（可采摘果实，conf=0.3, IoU=0.5）

- 漏检率 = **0.533**（169/317）
- 误检率 = **0.513**（156/304）

## 时延预算（六段拆分）

- 端到端 p50 = 9.809700000000031 ms · p90 = 16.313699999999987 ms · p99 = 27.716945999999954 ms

| 阶段 | p50(ms) | p90(ms) | p99(ms) | 均值(ms) |
|---|---|---|---|---|
| decode_dvpp | None | None | None | None |
| h2d_copy | None | None | None | None |
| model_infer | 9.809700000000031 | 16.313699999999987 | 27.716945999999954 | 11.402862962962956 |
| d2h_copy | None | None | None | None |
| postprocess_nms | None | None | None | None |
| ros_publish | None | None | None | None |
