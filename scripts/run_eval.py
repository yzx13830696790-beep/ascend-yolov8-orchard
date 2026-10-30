"""端到端评测：detector -> predictions -> mAP / 业务指标 / 时延打点 -> 报告。

detector 可插拔：
  --detector toy         使用 scripts/toy_detector.py（本机自测）
  --predictions x.jsonl  直接读预测（真模型/OM 推理后）
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from orch import metrics, report, ledger, latency

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_gts():
    with open(os.path.join(ROOT, "data", "meta", "gts.json"), encoding="utf-8") as f:
        gts = json.load(f)
    return {iid: g for iid, g in gts.items() if g["split"] == "val"}


def load_cfg():
    with open(os.path.join(ROOT, "configs", "constraints.json"), encoding="utf-8") as f:
        return json.load(f)


def run_detector_toy(image_id, image_path, conf_thr):
    import cv2
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from toy_detector import detect_as_dict
    bgr = cv2.imread(image_path)
    t0 = time.perf_counter()
    pred = detect_as_dict(image_id, bgr, conf_thr)
    return pred, (time.perf_counter() - t0) * 1000.0


def load_predictions(path):
    preds = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["boxes"] = np.array(d["boxes"], dtype=np.float32)
            d["scores"] = np.array(d["scores"], dtype=np.float32)
            d["class_ids"] = np.array(d["class_ids"], dtype=np.int64)
            preds.append(d)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="toy", choices=["toy"])
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    cfg = load_cfg()
    gts = load_gts()
    manifest = {}
    with open(os.path.join(ROOT, "data", "meta", "manifest.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            manifest[d["image_id"]] = d

    stage_lat = {"model_infer": [], "postprocess_nms": []}
    preds = []
    if args.predictions:
        preds = load_predictions(args.predictions)
    else:
        import cv2
        val_ids = sorted(gts.keys())
        for image_id in val_ids:
            pred, dt = run_detector_toy(image_id, manifest[image_id]["image_path"],
                                        cfg["conf_thr"])
            preds.append(pred)
            stage_lat["model_infer"].append(dt)

    m = metrics.evaluate(preds, gts, class_names=cfg["class_names"])
    bm = metrics.business_metrics(preds, gts, conf=cfg["business_conf"],
                                  iou=cfg["business_iou"],
                                  min_visible=cfg["min_visible"])
    summary = latency.summarize(stage_lat)

    out = {
        "meta": {
            "n_val": len(gts),
            "detector": "toy" if args.predictions is None else args.predictions,
            "note": "toy 检测器仅用于校验评测协议，mAP 无业务含义" if args.predictions is None
                    else "真模型预测",
            "conf_thr": cfg["conf_thr"],
        },
        "metrics": m,
        "business": bm,
        "latency_ms": summary["end_to_end"],
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics_val.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    report.write_report(m, summary, bm,
                        os.path.join(args.out, "report_val.md"))
    ledger.append(os.path.join(args.out, "ledger.csv"),
                  exp_id="eval_val_toy" if args.predictions is None else "eval_val_model",
                  change="val 集评测（toy 占位）" if args.predictions is None else "val 集评测",
                  model="toy" if args.predictions is None else "model",
                  map50_95=round(m["map50_95"], 4), map50=round(m["map50"], 4),
                  p50_ms=summary["end_to_end"]["p50"],
                  conclusion="harness 自测" if args.predictions is None else "")
    print(json.dumps({"mAP50-95": round(m["map50_95"], 4),
                      "mAP50": round(m["map50"], 4),
                      "business": bm}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
