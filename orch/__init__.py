"""orchard310b —— 昇腾 YOLOv8 果园检测的工程化实现。

模块划分（与规划书的"三条线"对应）：

  线 A 模型：  orch.dataset（数据/清单/校准集）
  线 B 部署：  orch.preprocess（AIPP 契约） · orch.decode · orch.nms · orch.atc
  线 C 度量：  orch.metrics · orch.latency · orch.pareto · orch.ledger
  量化：       orch.quant

设计原则：所有"能在 CPU 上验的"都用纯 numpy 实现并可测试；
所有"必须上昇腾"的动作都收敛成 orch.atc / orch.quant 里的命令构造器，
生成的命令原样可执行，不做假的占位。
"""

__version__ = "0.1.0"
