"""实验：AIPP 色域错配 / 归一化系数错配的精度代价 —— 实测量化复现。

把规划书第九节"全项目最大精度杀手"做成一次可复现的小实验，三个层面：
  1) 图像层面：YUV420SP 用错 csc 矩阵(BT709 vs BT601)转回 RGB 的 MAE
  2) 检测层面：错配矩阵后 toy 检测器的 mAP 下降
  3) 张量层面：AIPP mean/var 填错（忘乘255 / 把std当var）后的最大绝对误差

所有数字都在本机实测，并明确标注是"合成数据上的复现"，不是 310B 实测。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from orch import metrics
from orch.preprocess import (rgb_to_yuv420sp, yuv420sp_to_rgb, normalize_nchw,
                             apply_aipp, aipp_affine, MEAN, STD)
from toy_detector import detect_as_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_val():
    with open(os.path.join(ROOT, "data", "meta", "gts.json"), encoding="utf-8") as f:
        gts = json.load(f)
    manifest = {}
    with open(os.path.join(ROOT, "data", "meta", "manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            manifest[d["image_id"]] = d
    return {iid: g for iid, g in gts.items() if g["split"] == "val"}, manifest


def main():
    gts, manifest = load_val()
    conf_thr = 0.25

    maes = {"correct": [], "wrong_matrix": []}
    m = {"correct": [], "wrong_matrix": []}  # 各图像 mAP 增量的容器，这里按整集算
    preds_base, preds_correct, preds_wrong = [], [], []
    for image_id in sorted(gts.keys()):
        bgr = cv2.imread(manifest[image_id]["image_path"])
        rgb = bgr[..., ::-1]
        nv12 = rgb_to_yuv420sp(rgb, "BT601")
        rgb_correct = yuv420sp_to_rgb(nv12, "BT601")
        rgb_wrong = yuv420sp_to_rgb(nv12, "BT709")

        maes["correct"].append(np.mean(np.abs(rgb.astype(np.int16)
                                              - rgb_correct.astype(np.int16))))
        maes["wrong_matrix"].append(np.mean(np.abs(rgb.astype(np.int16)
                                                   - rgb_wrong.astype(np.int16))))

        preds_base.append(detect_as_dict(image_id, bgr, conf_thr))
        preds_correct.append(detect_as_dict(image_id, rgb_correct[..., ::-1], conf_thr))
        preds_wrong.append(detect_as_dict(image_id, rgb_wrong[..., ::-1], conf_thr))

    map_base = metrics.evaluate(preds_base, gts)["map50_95"]
    map_correct = metrics.evaluate(preds_correct, gts)["map50_95"]
    map_wrong = metrics.evaluate(preds_wrong, gts)["map50_95"]

    # 张量层：AIPP 归一化错误
    img = cv2.imread(manifest[sorted(gts.keys())[0]]["image_path"])[..., ::-1]
    ref = normalize_nchw(img, MEAN, STD)
    cases = {}
    cases["correct"] = apply_aipp(img, mean_aipp=aipp_affine(MEAN, STD)[0],
                                  var_aipp=aipp_affine(MEAN, STD)[1])
    cases["forgot_scale255"] = apply_aipp(img, mean_aipp=np.array(MEAN),
                                          var_aipp=np.array(STD))  # 直接填 mean/std
    cases["std_as_var"] = apply_aipp(img, mean_aipp=aipp_affine(MEAN, STD)[0],
                                     var_aipp=aipp_affine(MEAN, STD)[1][::-1])
    tensor_err = {k: float(np.max(np.abs(ref - v))) for k, v in cases.items()}

    result = {
        "image_mae": {k: float(np.mean(v)) for k, v in maes.items()},
        "detector_map50_95": {
            "base_rgb": float(map_base),
            "yuv_bt601_correct": float(map_correct),
            "yuv_bt601_to_bt709_wrong": float(map_wrong),
            "map_drop_wrong_matrix_pt": float((map_correct - map_wrong) * 100),
        },
        "tensor_max_abs_err_aipp": tensor_err,
        "note": "合成数据上的复现实验，量化 AIPP 配置错误的量级；非 310B 实测",
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", "exp_aipp_color_mismatch.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
