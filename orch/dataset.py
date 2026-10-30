"""数据集管理：YOLO 格式扫描 -> 清单 -> 分层划分 -> 校准集抽样。

关键设计（对应规划书 4.1 数据线）：
  * 标签第 6 列可带 visible_ratio（0~1），用于"高可见度子集 AP"与
    量化掉点归因；缺失时默认 1.0。
  * 校准集必须从真实分布里按 scene 分层抽，不能网上随便下 ——
    "分布偏移导致校准 scale 偏移、进而整层截断"的机制在这里落地。
"""
from __future__ import annotations

import json
import os
import random

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def load_yolo_label(path):
    boxes, classes, vis = [], [], []
    if not os.path.exists(path):
        return boxes, classes, vis
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            cx, cy, w, h = (float(x) for x in p[1:5])
            v = float(p[5]) if len(p) >= 6 else 1.0
            classes.append(c)
            boxes.append((cx, cy, w, h))
            vis.append(v)
    return boxes, classes, vis


def yolo_to_xyxy(cx, cy, w, h, iw, ih):
    x1 = (cx - w / 2) * iw
    y1 = (cy - h / 2) * ih
    x2 = (cx + w / 2) * iw
    y2 = (cy + h / 2) * ih
    return [x1, y1, x2, y2]


def scan_images(images_dir, labels_dir=None, label_suffix=".txt", scene_tags=None):
    """扫描图像，返回 list[dict(image_id, image_path, label_path, scene)]。

    scene_tags: 若给 dict(filename_prefix -> scene)，用于按场景标注；
    否则按目录名推断。
    """
    items = []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for fn in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in exts:
            continue
        img_path = os.path.join(images_dir, fn)
        lab_path = os.path.join(labels_dir or images_dir, stem + label_suffix)
        scene = "default"
        if scene_tags:
            for prefix, s in scene_tags.items():
                if stem.startswith(prefix):
                    scene = s
                    break
        items.append({"image_id": stem, "image_path": img_path,
                      "label_path": lab_path, "scene": scene})
    return items


def build_manifest(items, out_path, splits=None, seed=0):
    """写 manifest.jsonl，并把图片尺寸/标签一并展开，方便后续评测。"""
    rng = random.Random(seed)
    gts = {}
    for it in items:
        boxes, classes, vis = load_yolo_label(it["label_path"])
        iw = ih = 0
        if cv2 is not None and os.path.exists(it["image_path"]):
            im = cv2.imread(it["image_path"])
            if im is not None:
                ih, iw = im.shape[:2]
        xyxy = [yolo_to_xyxy(*boxes[k], iw, ih) for k in range(len(boxes))]
        gts[it["image_id"]] = {"boxes": xyxy, "classes": classes,
                               "visible": vis, "scene": it["scene"]}
        it["shape"] = [ih, iw]
        it["split"] = "val"
    # 划分 split
    if splits:
        ids = [it["image_id"] for it in items]
        rng.shuffle(ids)
        ratios = [splits[k] for k in ("train", "val", "calib") if k in splits]
        total = sum(ratios)
        cur = 0
        for key in ("train", "val", "calib"):
            if key not in splits:
                continue
            n = int(round(len(ids) * splits[key] / total))
            for iid in ids[cur:cur + n]:
                gts[iid]["_split"] = key
            cur += n
    else:
        for it in items:
            gts[it["image_id"]]["_split"] = "val"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return items, gts


def sample_calibration(gts, n, seed=0, per_scene=True):
    """从 val 之外的池子里按 scene 分层抽 n 张作校准集。"""
    rng = random.Random(seed)
    pool = [iid for iid, g in gts.items() if g.get("_split") != "val"]
    if not pool:
        pool = list(gts.keys())
    if per_scene:
        by_scene = {}
        for iid in pool:
            by_scene.setdefault(gts[iid].get("scene", "default"), []).append(iid)
        chosen = []
        while len(chosen) < n and by_scene:
            for scene in sorted(by_scene):
                if by_scene[scene]:
                    chosen.append(by_scene[scene].pop())
                if len(chosen) >= n:
                    break
        return chosen
    return rng.sample(pool, min(n, len(pool)))


def distribution_report(gts):
    """校准集/验证集的分布统计（用于落盘与人工复核讲解）。"""
    scenes = {}
    for iid, g in gts.items():
        scenes.setdefault(g.get("scene", "default"), 0)
        scenes[g.get("scene", "default")] += 1
    return scenes
