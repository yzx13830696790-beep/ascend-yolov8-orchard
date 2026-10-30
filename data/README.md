# data/ —— 合成数据集（本地生成，不入库）

这个目录由脚本本地生成，**不进 Git**（见根目录 `.gitignore`）。

```bash
make data
# 等价于：
python scripts/make_demo_data.py   # 生成 images/ + labels/（固定随机种子）
python scripts/prepare_data.py     # 生成 manifest.jsonl / 分集划分 / 校准集抽样
```

生成后会是这样：

```
data/
  images/          6 个场景 × 若干张 640×480 合成果园图（sunny/backlit/overcast/night/rain/blur）
  labels/          YOLO 格式标注，4 类：apple / apple_occluded / branch / leaf_cluster
  meta/
    manifest.jsonl   全量清单（分层划分 train/val，含可见度标签）
    gts.json         val 集真值，评测直接读它
    calibration.txt  PTQ 校准集抽样（每个场景等比例）
    summary.json     生成概要（张数、类别分布）
```

## 为什么用合成数据

真机数据在目标机上才有，而**评测协议的正确性不该依赖数据的真实性**。
这里的合成图只用来证明一件事：从"标注 → 清单 → 检测 → mAP → 业务指标 → 报告 → 台账"
这条链路是自洽的，边界情况（无预测、无 GT、单类缺失）都能算对。

## 换成真数据

`prepare_data.py` 只认标准 YOLO 格式（`images/*.jpg` + `labels/*.txt`，类别编号对齐
`configs/constraints.json` 的 `classes`）。把 `scripts/make_demo_data.py` 换成你的真实
数据集拷贝即可，后面全链路不用改一行。

`toy_detector`（`scripts/toy_detector.py`，经典 CV 占位检测器）**必须**在这一步换成真实模型，
否则评测出来的 mAP 没有任何业务含义——这一点在根目录 README 的"诚实边界"里也说了一遍。
