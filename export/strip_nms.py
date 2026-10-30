"""ONNX 导出：剥离 NMS 与后处理，只导主干，固定 1x3x640x640。

运行环境：训练机（有 ultralytics + torch）。本机无 torch，此脚本在训练机
执行后，配合 export/check_contract.py 做契约校验。

为什么剥离 NMS（规划书"三层追问"第 1 条）：
  ATC 对 NonMaxSuppression 支持差、动态 shape 不友好；后处理放 Host 侧
  更灵活，且能对 8400 候选框做向量化 NMS 与坐标还原的独立调优。
"""
import argparse
import json
import os


def export(model_weights, out_onnx, imgsz=640, opset=11):
    from ultralytics import YOLO
    model = YOLO(model_weights)
    # YOLOv8 默认导出已把 NMS 留在外面？—— 不，ultralytics 默认带后处理。
    # 这里显式关闭 NMS：end2end=False 表示不导出 NMS 头。
    model.export(format="onnx", imgsz=imgsz, opset=opset,
                 dynamic=False, simplify=True, nms=False)
    # ultralytics 会输出到模型同目录；复制到目标路径
    src = os.path.splitext(model_weights)[0] + ".onnx"
    if not os.path.exists(src):
        src = model_weights.replace(".pt", ".onnx")
    os.replace(src, out_onnx)
    print(f"exported -> {out_onnx}")


def write_contract(out_onnx, contract_path, nc=4, imgsz=640):
    """写导出契约 JSON，供 check_contract.py 校验。"""
    contract = {
        "model": out_onnx,
        "input_shape": [1, 3, imgsz, imgsz],
        "opset": 11,
        "nc": nc,
        "num_anchors": 8400,
        "output_name": "output0",          # YOLOv8 头名
        "output_shape": [1, 4 + nc, 8400],
        "no_nms_op": True,                  # 已剥离 NMS
        "dynamic_axes": [],
    }
    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, ensure_ascii=False, indent=2)
    print(f"contract -> {contract_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="baseline.onnx")
    ap.add_argument("--contract", default="baseline.contract.json")
    ap.add_argument("--nc", type=int, default=4)
    args = ap.parse_args()
    export(args.weights, args.out)
    write_contract(args.out, args.contract, nc=args.nc)
