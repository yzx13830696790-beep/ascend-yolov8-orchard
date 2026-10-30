# 项目状态：什么已实测、什么待硬件

> 本文件是"诚实边界"的唯一真源。面试前先对照它，确保你不会把占位数字说成实测。

## 已在本机实测（Windows + Python 3.8 + numpy/cv2，无昇腾）

| 组件 | 验证方式 | 结论 |
|---|---|---|
| letterbox / 坐标还原 | `tests.test_core.TestLetterbox` | 往返误差 < 1e-4 |
| AIPP 仿射 = 训练归一化 | `TestAipp.test_affine_matches_normalize` | 全精度 mean/var 下误差 < 1e-3 |
| AIPP 整数 mean 截断误差 | `TestAipp.test_affine_integer_mean_bounded` | 有界 < 0.02（真实量级 0.008） |
| 向量化 NMS | `TestNms` + `scripts/exp_nms_bench.py` | 与 naive 单类等价；8400 框 numpy 约 100ms |
| mAP 协议 | `TestMetrics`（完美/漏检/业务指标） | 边界正确 |
| ATC 校验器 | `TestAtcValidator` | 三处硬伤全部拦截，正确配置 0 错误 |
| 量化命令生成 | `TestQuant` | PTQ/敏感度命令含 skip_layers |
| 帕累托判定 | `TestPareto` | 支配关系正确 |
| 全链路 | `scripts/run_demo.py` | 数据→清单→评测→实验→图 跑通 |
| 实验平台 console | 启动服务 + curl 全接口 | 首页/概览/发起评测/报告/台账/两图/入料 全部 200；入料幂等；JS 语法通过 node --check |

## 待昇腾硬件 / 训练机

| 组件 | 依赖 | 状态 |
|---|---|---|
| YOLOv8 训练 | torch + ultralytics | `export/strip_nms.py` 已就绪 |
| ONNX 结构级校验 | onnx（可选装） | `export/check_contract.py` 支持 |
| AMCT PTQ/敏感度/QAT | CANN + amct | `orch.quant` 产出命令 |
| ATC 转 OM | CANN atc | `deploy/atc/build_om.sh` |
| ACL 三线程流水 | CANN runtime | `deploy/acl` 工程，目标机编译 |
| C++ NMS | g++ | `deploy/postprocess/nms.cpp`，目标机编译 |

## 关键数字口径

- 本机 mAP（toy 占位检测器，4 类合成数据）：**无业务含义**，仅证明评测协议正确。
- `results/pareto_points.csv` 中 `measured=0` 的行是**占位**（任务书口径），
  只有在 310B 上 `msprof` 实测后回填为 `measured=1`。
- 任务书"GPU 42ms / mAP 0.86"缺硬件型号、batch、评测口径，本仓库**不采信**。
