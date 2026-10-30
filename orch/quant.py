"""量化流水线：PTQ -> 逐层敏感度 -> 混合精度 -> QAT 的命令构造器。

规划书 4.1「量化是核心考点」。AMCT 的 CLI 参数按官方口径生成，不臆造；
本机没有 AMCT，所以这里产出的是"可直接执行的命令/配置"，并在
--dry-run 下打出完整执行计划。真正跑时把 dry-run 关掉即可。

四个阶段：
  S1 PTQ：      全局 INT8 基线（amct_onnx calibration + quantization）
  S2 敏感度：   对每个 Conv 只量化它自己（skip 其余层），测 ΔmAP 排序
  S3 混合精度： 敏感层保 FP16（skip_layers），其余 INT8
  S4 QAT：      amct_pytorch 兜底 10 epoch
"""
from __future__ import annotations

import csv
import json
import os


def _onnx_shape(name="images", n=1, c=3, h=640, w=640):
    return f'{name}:{n},{c},{h},{w}'


def ptq_commands(model_onnx, calib_dir, workdir, input_shape=None,
                 calib_types=None, soc_version="Ascend310B4"):
    shape = input_shape or _onnx_shape()
    calib_types = calib_types or "float32"
    cal = [
        "amct_onnx calibration",
        f"    --model {model_onnx}",
        f"    --save_path {workdir}/calib_result",
        f"    --input_shape {shape}",
        f"    --data_dir {calib_dir}",
        f"    --data_types {calib_types}",
    ]
    quant = [
        "amct_onnx quantization",
        f"    --model {model_onnx}",
        f"    --save_path {workdir}/quant_result",
        f"    --input_shape {shape}",
        f"    --data_dir {calib_dir}",
        f"    --data_types {calib_types}",
    ]
    return {"stage": "PTQ", "calibration": "\n".join(cal),
            "quantization": "\n".join(quant)}


def sensitivity_commands(model_onnx, layers, calib_dir, workdir, input_shape=None):
    """对每个 Conv 只量化它自己：skip 其余所有层。"""
    shape = input_shape or _onnx_shape()
    cmds = []
    for layer in layers:
        skip = ",".join(l for l in layers if l != layer)
        cmds.append({
            "layer": layer,
            "cmd": (
                "amct_onnx quantization\n"
                f"    --model {model_onnx}\n"
                f"    --save_path {workdir}/sens_{layer}\n"
                f"    --input_shape {shape}\n"
                f"    --data_dir {calib_dir}\n"
                f"    --data_types float32\n"
                f"    --skip_layers {skip}"
            ),
        })
    return {"stage": "SENSITIVITY", "layers": layers, "commands": cmds}


def mixed_precision_commands(model_onnx, sensitive_layers, calib_dir, workdir,
                             input_shape=None):
    """敏感层保 FP16（进 skip_layers），其余 INT8。"""
    shape = input_shape or _onnx_shape()
    skip = ",".join(sensitive_layers)
    return {
        "stage": "MIXED_PRECISION",
        "keep_fp16_layers": sensitive_layers,
        "cmd": (
            "amct_onnx quantization\n"
            f"    --model {model_onnx}\n"
            f"    --save_path {workdir}/quant_mixed\n"
            f"    --input_shape {shape}\n"
            f"    --data_dir {calib_dir}\n"
            f"    --data_types float32\n"
            f"    --skip_layers {skip}"
        ),
    }


def qat_commands(model_pt, train_loader_module, workdir, epochs=10, lr=1e-4):
    return {
        "stage": "QAT",
        "cmd": (
            f"# 用 amct_pytorch 对 {model_pt} 做 {epochs} epoch QAT 兜底\n"
            f"# 学习率 {lr}，训练脚本 {train_loader_module}\n"
            "python -m quant.qat_train "
            f"--model {model_pt} --epochs {epochs} --lr {lr} "
            f"--out {workdir}/qat_model.onnx"
        ),
    }


def sensitivity_ranking(csv_path, thr=1.0):
    """读 S2 结果（layer,map,delta），返回按 ΔmAP 降序 + 是否敏感。"""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"layer": r["layer"],
                         "map": float(r["map"]),
                         "delta": float(r["delta"])})
    rows.sort(key=lambda r: -r["delta"])
    for r in rows:
        r["sensitive"] = r["delta"] >= thr
    return rows


def dump_plan(plan, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    return out_path
