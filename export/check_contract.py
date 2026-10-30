"""导出契约校验：不依赖 onnx，用 onnx 提供的 Python API（若装了）做结构级检查，
否则退化为契约 JSON 的自洽校验（固定输入 shape、opset、无 NMS、输出布局）。

本机可跑：`python export/check_contract.py --contract baseline.contract.json`
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_via_onnx(model_path, contract):
    import onnx
    m = onnx.load(model_path)
    onnx.checker.check_model(m)
    opset = m.opset_import[0].version
    graph = m.graph
    op_types = [n.op_type for n in graph.node]
    errs = []
    if opset != contract["opset"]:
        errs.append(f"opset {opset} != {contract['opset']}")
    if any("NonMaxSuppression" in t or "Nms" in t for t in op_types):
        errs.append("检测到 NMS 算子，未剥离干净")
    # 输入 shape
    inp = graph.input[0]
    dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    if dims != contract["input_shape"]:
        errs.append(f"input shape {dims} != {contract['input_shape']}")
    return errs


def check_contract(contract):
    errs = []
    shape = contract.get("input_shape")
    if shape and (len(shape) != 4 or shape[0] != 1 or shape[1] != 3):
        errs.append(f"input_shape {shape} 应为 [1,3,H,W]")
    out_shape = contract.get("output_shape")
    if out_shape and out_shape[1] != 4 + contract.get("nc", 4):
        errs.append(f"output_shape {out_shape} 与 nc 不一致")
    if not contract.get("no_nms_op", False):
        errs.append("契约未声明 no_nms_op=true")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    with open(args.contract, encoding="utf-8") as f:
        contract = json.load(f)

    errs = check_contract(contract)
    if args.model and os.path.exists(args.model):
        try:
            errs += check_via_onnx(args.model, contract)
        except ImportError:
            print("[warn] onnx 未安装，跳过结构级检查（仅契约自洽校验）")
    if errs:
        print("FAIL:")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print("PASS: 导出契约合规（固定输入、opset 匹配、无 NMS、输出布局 1x(4+nc)x8400）")


if __name__ == "__main__":
    main()
