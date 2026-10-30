# 版本矩阵：CANN / AMCT / ATC / ONNX opset

> 高频追问："你这套环境是怎么固定的？" 版本矩阵就是答案，工程上也给出了安装前的自检依据。

## 芯片 -> 最低 CANN

| soc_version | 最低 CANN | 说明 |
|---|---|---|
| Ascend310B4 | 7.0.RC1 | 任务书写 6.3.RC2 是错的，ATC 不认识该 soc_version |
| Ascend310P3 | 7.0.RC1 | 与 310B4 同代 |
| Ascend910B4 | 8.0.RC1 | 训练卡 |

## 推荐组合（本项目 Dockerfile 固定）

| 组件 | 版本 | 用途 |
|---|---|---|
| CANN | 8.0.RC1 | 工具链 + ACL runtime |
| AMCT | 与 CANN 配套（见官方对应表） | ONNX/PyTorch 量化 |
| ATC | 随 CANN | ONNX -> OM |
| ONNX opset | 11 | 导出契约固定 |
| 输入 shape | 1×3×640×640 | 固定，无动态轴 |

## 踩坑记录模板

| 日期 | CANN | AMCT | 现象 | 解法 |
|---|---|---|---|---|
| —— | —— | —— | —— | —— |

> 维护这张表本身，就是"我在认真管理环境可复现性"的最强证明。
