"""经典 CV 基线检测器 —— 只用于校验评测协议/后处理正确性，不是 YOLO 基线。

用一个纯 OpenCV 的颜色/形状检测器站在 CNN 的位置上，让 run_eval 全链路
在本机（无 torch/无昇腾）也能实跑出真实数字。README 里明确标注：
* 它是 harness 自测用的占位模型，其 mAP 无业务含义；
* 检测 class 0/1/2/3 各用独立颜色/形态启发式，可自由替换为真实模型输出。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


def _components(mask, min_area=40):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append((x, y, w, h, area))
    return out


def detect_apples(bgr, occluded=False):
    """红/暗红果实。occluded=True 用更暗、更低饱和度的掩码。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    if occluded:
        mask = cv2.inRange(hsv, (0, 40, 40), (10, 200, 160)) \
             | cv2.inRange(hsv, (160, 40, 40), (180, 200, 160))
    else:
        mask = cv2.inRange(hsv, (0, 90, 90), (10, 255, 255)) \
             | cv2.inRange(hsv, (160, 90, 90), (180, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def detect_branches(bgr):
    """棕色长条：brown mask + 长宽比过滤。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 40, 30), (25, 200, 160))
    return mask


def detect_leaves(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (30, 60, 40), (80, 255, 200))
    return mask


def _boxes_from_mask(mask, h, w, norm_area, min_area=40, elong_ratio=None):
    preds = []
    for (x, y, bw, bh, area) in _components(mask, min_area):
        if elong_ratio is not None:
            ar = max(bw, bh) / max(1, min(bw, bh))
            if ar < elong_ratio:
                continue
        score = float(np.clip(area / norm_area, 0.05, 0.98))
        preds.append((x, y, x + bw, y + bh, score))
    return preds


def detect(bgr):
    """返回 list[(x1,y1,x2,y2,score,class_id)]。"""
    h, w = bgr.shape[:2]
    preds = []
    for (x1, y1, x2, y2, s) in _boxes_from_mask(detect_apples(bgr), h, w, 2000, 60):
        preds.append((x1, y1, x2, y2, s, 0))
    for (x1, y1, x2, y2, s) in _boxes_from_mask(detect_apples(bgr, occluded=True),
                                               h, w, 1600, 40):
        preds.append((x1, y1, x2, y2, min(s * 0.9, 0.9), 1))
    for (x1, y1, x2, y2, s) in _boxes_from_mask(detect_branches(bgr), h, w, 3500, 120,
                                               elong_ratio=2.5):
        preds.append((x1, y1, x2, y2, s, 2))
    for (x1, y1, x2, y2, s) in _boxes_from_mask(detect_leaves(bgr), h, w, 3000, 200):
        preds.append((x1, y1, x2, y2, s, 3))
    return preds


def detect_as_dict(image_id, bgr, conf_thr=0.25):
    preds = detect(bgr)
    boxes = np.array([[p[0], p[1], p[2], p[3]] for p in preds], dtype=np.float32) \
        if preds else np.zeros((0, 4), np.float32)
    scores = np.array([p[4] for p in preds], dtype=np.float32) \
        if preds else np.zeros((0,), np.float32)
    class_ids = np.array([p[5] for p in preds], dtype=np.int64) \
        if preds else np.zeros((0,), np.int64)
    keep = scores >= conf_thr
    return {"image_id": image_id, "boxes": boxes[keep],
            "scores": scores[keep], "class_ids": class_ids[keep]}
