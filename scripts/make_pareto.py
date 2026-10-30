"""生成帕累托前沿图：实心点=实测，空心点=待 310B 实测（占位，来自任务书口径）。

严格遵循"性能数字必须有口径"：没有实测的数据点一律画成空心三角形，
并在 caption 里标注来源，绝不与实测点混淆。
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orch import pareto

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    # 实测点：CPU toy 基线（真数字）
    with open(os.path.join(ROOT, "results", "metrics_val.json"), encoding="utf-8") as f:
        mv = json.load(f)
    measured = [{
        "name": "CPU toy 基线（实测）",
        "name_en": "CPU toy baseline",
        "latency": round(mv["latency_ms"]["p50"], 2),
        "map": round(mv["metrics"]["map50_95"], 4),
        "measured": True,
    }]

    # 待实测占位点（任务书口径，明确标注）
    # name_en：容器/昇腾机上通常没有中文字体，图里要用它兜底
    pending = [
        {"name": "GPU FP32（任务书口径，待复核）", "name_en": "GPU FP32 (doc, TBC)",
         "latency": 42, "map": 0.86},
        {"name": "OM FP16(AIPP)", "name_en": "OM FP16 (AIPP)", "latency": 22, "map": 0.86},
        {"name": "OM FP16+三线程流水", "name_en": "OM FP16+3thread", "latency": 16, "map": 0.86},
        {"name": "OM INT8 PTQ", "name_en": "OM INT8 PTQ", "latency": 13, "map": 0.85},
        {"name": "OM INT8 混合精度", "name_en": "INT8 mixed-prec.", "latency": 11, "map": 0.85},
        {"name": "OM INT8+DVPP+流水（目标）", "name_en": "INT8+DVPP+pipeline", "latency": 11, "map": 0.85},
    ]
    for p in pending:
        p["measured"] = False
    points = measured + pending

    csv_path = os.path.join(ROOT, "results", "pareto_points.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "name_en", "latency_ms",
                                          "map50_95", "measured"])
        w.writeheader()
        for p in points:
            w.writerow({"name": p["name"], "name_en": p.get("name_en", ""),
                        "latency_ms": p["latency"], "map50_95": p["map"],
                        "measured": int(p["measured"])})

    png = os.path.join(ROOT, "results", "figures", "pareto.png")
    res = pareto.plot(points, png, latency_budget=30)
    print(json.dumps({"csv": csv_path, "png": png,
                      "measured_points": [p["name"] for p in measured],
                      "front": res["front"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
