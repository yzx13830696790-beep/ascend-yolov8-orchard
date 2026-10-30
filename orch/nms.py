"""后处理：YOLOv8 头解码 + 向量化 NMS。

为什么要自己写：
  ATC 对 NonMaxSuppression 的支持很差（动态 shape、TopK 组合多），
  把它留在 ONNX 里通常会被拆成一堆小算子甚至转不过去。所以 OM 只输出
  原始头 (1, 4+nc, 8400)，NMS 放到 Host 侧。既然如此，Host 侧 NMS 的
  耗时就直接进端到端时延预算表，必须做到 8400 个候选框 1ms 量级。
"""
from __future__ import annotations

import numpy as np


def iou_matrix(a, b):
    """a:(N,4) b:(M,4) xyxy -> (N,M) IoU。"""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = a[:, None, :]
    b = b[None, :, :]
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])
    w = np.clip(xx2 - xx1, 0, None)
    h = np.clip(yy2 - yy1, 0, None)
    inter = w * h
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area_a + area_b - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def nms_naive(boxes, scores, iou_thr=0.45, max_det=300):
    """O(n^2) 参考实现，只用于和向量化版本做等价性测试。"""
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
        if len(keep) >= max_det:
            break
    return np.asarray(keep, dtype=np.int64)


def nms(boxes, scores, iou_thr=0.45, max_det=300, class_ids=None):
    """单类/多类统一的向量化 NMS（class-offset 技巧）。

    多类时给每类的框加上 class_id * offset 的平移量，使得"不同类之间
    的 IoU 必然为 0"，从而一次贪心扫完所有类，避免 Python 层的按类循环。
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.int64)

    if class_ids is not None:
        class_ids = np.asarray(class_ids, dtype=np.float32)
        offset = class_ids * (boxes.max() + 1.0)
    else:
        offset = np.zeros(len(boxes), dtype=np.float32)

    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[i:i + 1] + offset[i],
                          boxes[rest] + offset[rest][:, None])[0]
        order = rest[ious <= iou_thr]
        if len(keep) >= max_det:
            break
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes, scores, class_ids, iou_thr=0.45, max_det=300):
    """包装：返回按分数降序的 (boxes, scores, class_ids)。"""
    keep = nms(boxes, scores, iou_thr, max_det, class_ids)
    return boxes[keep], scores[keep], np.asarray(class_ids)[keep]
