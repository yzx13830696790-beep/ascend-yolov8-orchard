"""YOLOv8 输出头解码。

OM 的输出约定（剥离了 NMS 与后处理）：
  * FP16 OM：output  = (1, 4 + nc, 8400)  float16，8400 = 80*80 + 40*40 + 20*20
  * 若做了 NMS 后处理： (1, 300, 6)       —— 本项目不采用

解码顺序：transpose -> (8400, 4+nc) -> 取前 4 列为 cxcywh -> 置信度 argmax
-> 阈值过滤 -> xyxy（仍在 640 输入坐标系）-> scale_boxes_back。
"""
from __future__ import annotations

import numpy as np

from .nms import nms
from .preprocess import scale_boxes_back


def cxcywh_to_xyxy(x):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2.0
    y[:, 1] = x[:, 1] - x[:, 3] / 2.0
    y[:, 2] = x[:, 0] + x[:, 2] / 2.0
    y[:, 3] = x[:, 1] + x[:, 3] / 2.0
    return y


def decode_yolov8(output, conf_thr=0.25, iou_thr=0.45, max_det=300,
                  nc=None, r=1.0, pad=(0.0, 0.0), to_original=True):
    """output: (1, 4+nc, N) 或 (1, N, 4+nc) 或 (4+nc, N)。

    返回 dict(boxes(N,4) xyxy, scores(N,), class_ids(N,))，坐标默认还原到原图。
    """
    o = np.asarray(output)
    if o.dtype == np.float16:
        o = o.astype(np.float32)
    while o.ndim > 2:
        o = o[0]
    if o.shape[0] <= 256 and (o.shape[1] > o.shape[0]):   # (4+nc, N)
        o = o.T
    if nc is None:
        nc = o.shape[1] - 4
    boxes_c = o[:, :4]
    cls_scores = o[:, 4:4 + nc]
    class_ids = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

    m = scores >= conf_thr
    boxes_c = boxes_c[m]
    scores = scores[m]
    class_ids = class_ids[m]
    if boxes_c.shape[0] == 0:
        empty = np.zeros((0, 4), np.float32)
        return {"boxes": empty, "scores": np.zeros((0,), np.float32),
                "class_ids": np.zeros((0,), np.int64)}

    boxes = cxcywh_to_xyxy(boxes_c)
    keep = nms(boxes, scores, iou_thr, max_det, class_ids)
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    if to_original:
        boxes = scale_boxes_back(boxes, r, pad[0], pad[1])
    return {"boxes": boxes, "scores": scores, "class_ids": class_ids}
