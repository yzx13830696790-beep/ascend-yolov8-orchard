"""数据准备：扫描 images+labels -> manifest/gts -> 分层划分 -> 校准集抽样。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orch import dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    img_dir = os.path.join(ROOT, "data", "images")
    lab_dir = os.path.join(ROOT, "data", "labels")
    out_dir = os.path.join(ROOT, "data", "meta")
    os.makedirs(out_dir, exist_ok=True)

    items = dataset.scan_images(img_dir, lab_dir, scene_tags={
        "sunny": "sunny", "backlit": "backlit", "overcast": "overcast",
        "night": "night", "rain": "rain", "blur": "blur"})
    splits = {"train": 0.5, "val": 0.3, "calib": 0.2}
    _, gts = dataset.build_manifest(items, os.path.join(out_dir, "manifest.jsonl"),
                                    splits=splits, seed=0)

    # 为每个 image 记录 split
    split_of = {it["image_id"]: gts[it["image_id"]]["_split"] for it in items}
    val_ids = [iid for iid, s in split_of.items() if s == "val"]

    # 校准集：从 train+calib 池子里分层抽样
    calib_ids = dataset.sample_calibration(gts, n=30, seed=0)

    # 输出 gts.json（含 split 与 visible）
    gts_out = {iid: {"boxes": g["boxes"], "classes": g["classes"],
                     "visible": g["visible"], "scene": g["scene"],
                     "split": g["_split"]} for iid, g in gts.items()}
    with open(os.path.join(out_dir, "gts.json"), "w", encoding="utf-8") as f:
        json.dump(gts_out, f, ensure_ascii=False, indent=1)

    with open(os.path.join(out_dir, "calibration.txt"), "w") as f:
        f.write("\n".join(calib_ids) + "\n")

    # 分布报告
    dist = dataset.distribution_report({iid: gts[iid] for iid in val_ids})
    summary = {
        "n_total": len(items),
        "n_val": len(val_ids),
        "n_calib": len(calib_ids),
        "val_scene_distribution": dist,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
