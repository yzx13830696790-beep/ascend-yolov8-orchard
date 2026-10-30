# 踩坑机制：三处硬伤的根因，而不是"照抄结论"

任务书里的三处错误，每一处都对应一个**可以讲出机制的根因**。本仓库用代码
把这些机制固化下来，`scripts/gen_atc.py` 会现场演示校验器能否抓到。

## 1. CANN 版本与芯片不匹配

- **现象**：`atc --soc_version=Ascend310B4` 直接报"不支持的 soc_version"。
- **机制**：soc_version 是 ATC 里把算子映射到具体芯片实现的关键字，芯片的
  算子库随 CANN 版本发布。310B 系列在 7.0.RC1 才首次纳入支持，6.3.RC2 的
  算子库根本没有 `Ascend310B4` 这个注册项。
- **验证**：`orch.atc.validate(soc_version="Ascend310B4", cann_version="6.3.RC2")`
  返回 error。

## 2. AIPP 输入格式与 DVPP 输出不匹配（全项目最大精度杀手）

- **现象**：OM 上 mAP 掉 5~12pt，且越靠后的层误差越大。
- **机制**：DVPP 硬解码输出的是 YUV420SP（NV12），而 AIPP 写 `input_format: RGB`
  意味着 AIPP 会按 RGB 的字节布局去解释 YUV 数据 —— 通道顺序和色度采样完全对不上，
  第一层就引入了系统性颜色错误，后续逐层放大。
  - 正确路径：`input_format: YUV420SP_U8` + `csc_switch: true` + 与训练一致的 csc 矩阵。
  - 只有在 Host 侧已经 `cv::cvtColor` 成 RGB 再送 Device 时，`RGB` 才成立 ——
    而那条路径让预处理退回 CPU，DVPP 的硬件解码/缩放就白用了。
- **实测量化**（`scripts/exp_color_matrix.py`，合成数据）：
  - 图像层面：YUV 用错 BT709 转回 RGB 的 MAE 是正确 BT601 的 **约 5 倍**。
  - 张量层面：AIPP 归一化写错的误差见下表。

| AIPP 配置 | 最大绝对误差（vs 训练归一化） |
|---|---|
| 正确 mean/var（= 255×mean, 1/(255×std)） | ~0.008 |
| 忘乘 255（直接填 0~1 的 mean/std） | **~33.7** |
| 把 std 当 var 填 | ~0.034 |

## 3. force_fp16 与 op_select_implmode=high_precision 冲突

- **现象**：本想用 FP16 提性能，结果敏感算子全回落到 FP32，时延没降下来。
- **机制**：`precision_mode=force_fp16` 是"全局强制 FP16"；而
  `op_select_implmode=high_precision` 是"精度优先的算子实现选择"——它会让对精度
  敏感的算子选择 FP32 实现。两个开关在语义上互相抵消。
- **取舍**：要性能就 `high_performance`，再用任务级 mAP 验证精度；要精度就先
  `high_precision` 拿到精度上限，再逐算子放开到 FP16。

## 坑：容器里图能出，中文全是方块

**现象**：本机（Windows）帕累托图中文正常；同一份代码上了 Linux 容器，
标题、坐标轴标题、图例、数据点标签全部变成方块，但程序退出码 0、图也照样写盘。

**根因**：`matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]` 写死了
Windows 字体。Linux 容器/昇腾训练机通常只装 DejaVu 这类西文字体，matplotlib 找不到
字形时不报错，直接画 .notdef 方块。

**修法**：`orch/fonts.py` 运行时探测候选 CJK 字体（雅黑/黑体/苹方/Noto CJK/文泉驿），
命中就用中文，全都没有就整张图切英文文案；数据点额外带 `name_en` 兜底。
环境变量 `ORCH_FORCE_NO_CJK=1` 可强制走英文分支，用来在 Windows 上验证容器效果
——否则这个问题在开发机永远复现不出来。

**教训**：渲染类产物必须"在目标环境里看一眼"。退出码 0 不等于结果可读。
