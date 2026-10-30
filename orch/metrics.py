"""评测协议：mAP50-95 / 每类 AP / 小目标 AP / 可见度切片 / 业务指标。

这是整份项目"精度对齐"的度量准绳，规则全部固定并落进 results/metrics_*.json：
  * 固定 val 集（不参与训练/校准）
  * conf=0.001 时才做 101 点插值召回；业务指标用 conf=0.3
  * NMS IoU=0.7 作为任务级兜底（模型输出已过 NMS，这里是保险）
  * 所有数字都带口径（哪张 val、多少张、哪些类参与）
"""
from __future__ import annotations

import numpy as np

from .nms import iou_matrix


def _unique(img_ids):
    return sorted(set(img_ids))


def compute_ap(rec, prec):
    """101-point interpolation (COCO/VOC 风格)。"""
    rec = np.asarray(rec, dtype=np.float64)
    prec = np.asarray(prec, dtype=np.float64)
    if rec.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _sorted_preds(preds):
    """preds: list[dict(boxes(N,4), scores(N,), class_ids(N,), image_id)]
    -> 按分数降序的全局数组。"""
    all_scores, all_cids, all_img, all_boxes = [], [], [], []
    for p in preds:
        n = len(p["scores"])
        all_scores.append(p["scores"])
        all_cids.append(p["class_ids"])
        all_img.append(np.full(n, p["image_id"], dtype=object))
        all_boxes.append(p["boxes"])
    if not all_scores:
        return (np.empty((0,), np.float32), np.empty((0,), np.int64),
                np.empty((0,), object), [])
    s = np.concatenate(all_scores)
    c = np.concatenate(all_cids).astype(np.int64)
    im = np.concatenate(all_img)
    b = np.concatenate(all_boxes)
    order = np.argsort(-s, kind="stable")
    return s[order], c[order], im[order], b[order]


def _gt_index(gts):
    """gts: {image_id: {boxes(N,4), classes(N,), visible(N,)}}
    -> 扁平数组，用于按 class/image 快速取用。"""
    idx = {"img": [], "cls": [], "vis": [], "box": [], "id": []}
    for im_id, g in gts.items():
        n = len(g["classes"])
        idx["img"].extend([im_id] * n)
        idx["cls"].extend(np.asarray(g["classes"], dtype=np.int64).tolist())
        idx["vis"].extend(np.asarray(g.get("visible", np.ones(n)), dtype=np.float64).tolist())
        idx["box"].append(np.asarray(g["boxes"], dtype=np.float32))
        idx["id"].extend([im_id] * n)
    if not idx["img"]:
        return {"img": np.empty((0,), object), "cls": np.empty((0,), np.int64),
                "vis": np.empty((0,), np.float64),
                "box": np.zeros((0, 4), np.float32)}
    return {"img": np.asarray(idx["img"], dtype=object),
            "cls": np.asarray(idx["cls"], dtype=np.int64),
            "vis": np.asarray(idx["vis"], dtype=np.float64),
            "box": np.vstack(idx["box"]) if idx["box"] else np.zeros((0, 4), np.float32)}


def evaluate(preds, gts, class_names=None, iou_thrs=None, gt_filter=None,
             area_filter=None):
    """计算标准检测指标。

    参数
      gt_filter: 可选 dict，用于切片（例如 {"min_visible": 0.5}）
      area_filter: ("small", area<=32^2) / ("medium", ...) / ("large", ...)，
                   按 640 坐标系面积。
    返回 dict，含 mAP50、mAP50-95、per_class（每类 AP50 与 AP50-95）、
    num_gt / num_pred 等。
    """
    if iou_thrs is None:
        iou_thrs = np.round(np.linspace(0.5, 0.95, 10), 2)
    gidx = _gt_index(gts)

    # GT 过滤（可见度切片 / 面积切片）
    mask = np.ones(len(gidx["img"]), dtype=bool)
    if gt_filter and "min_visible" in gt_filter:
        mask &= gidx["vis"] >= gt_filter["min_visible"]
    if area_filter is not None:
        areas = np.clip(gidx["box"][:, 2] - gidx["box"][:, 0], 0, None) \
            * np.clip(gidx["box"][:, 3] - gidx["box"][:, 1], 0, None)
        if area_filter == "small":
            mask &= areas <= 32 ** 2
        elif area_filter == "medium":
            mask &= (areas > 32 ** 2) & (areas <= 96 ** 2)
        elif area_filter == "large":
            mask &= areas > 96 ** 2
    gimg = gidx["img"][mask]
    gcls = gidx["cls"][mask]
    gbox = gidx["box"][mask]

    # 匹配：对每个 IoU 阈值算一次
    aps = {float(t): {} for t in iou_thrs}
    classes = sorted(set(gidx["cls"].tolist())) if len(gidx["cls"]) else []
    s, cls, im, boxes = _sorted_preds(preds)
    for c in classes:
        gt_c = gcls == c
        for t in iou_thrs:
            aps[float(t)][c] = _ap_class_c(s, cls, im, boxes, gimg[gt_c], gbox[gt_c], c, t)
    # 若某类无 GT，不计入均值
    per_class_ap50 = {c: aps[0.5][c] for c in classes}
    per_class_map = {c: float(np.mean([aps[float(t)][c] for t in iou_thrs])) for c in classes}
    map50 = float(np.mean(list(per_class_ap50.values()))) if per_class_ap50 else 0.0
    map5095 = float(np.mean(list(per_class_map.values()))) if per_class_map else 0.0
    return {
        "map50": map50,
        "map50_95": map5095,
        "per_class_ap50": per_class_ap50,
        "per_class_map": per_class_map,
        "num_gt": int(mask.sum()),
        "classes_evaluated": classes,
        "class_names": class_names,
    }


def _ap_class_c(s, cls, im, boxes, gimg_c, gbox_c, c, t):
    """针对类别 c、IoU 阈值 t，返回 AP。"""
    sel = cls == c
    ps, pim, pbox = s[sel], im[sel], boxes[sel]
    if ps.size == 0:
        return 0.0 if len(gimg_c) else 1.0
    # 按图像分组 GT
    gt_by_img = {}
    for i, img in enumerate(gimg_c):
        gt_by_img.setdefault(img, []).append(gbox_c[i])
    # 贪心匹配
    tp = np.zeros(ps.size, dtype=bool)
    used = {img: np.zeros(len(v), dtype=bool) for img, v in gt_by_img.items()}
    for i in range(ps.size):
        img = pim[i]
        if img not in gt_by_img:
            continue
        gb = np.asarray(gt_by_img[img], dtype=np.float32)
        ious = iou_matrix(pbox[i:i + 1], gb)[0]
        j = int(np.argmax(ious)) if ious.size else -1
        if j >= 0 and ious[j] >= t and not used[img][j]:
            tp[i] = True
            used[img][j] = True
    fp = np.cumsum(~tp)
    tp_c = np.cumsum(tp)
    npos = len(gimg_c)
    rec = tp_c / npos if npos else np.zeros_like(tp_c, dtype=np.float64)
    prec = tp_c / np.maximum(tp_c + fp, 1)
    return compute_ap(rec, prec)


def business_metrics(preds, gts, conf=0.3, iou=0.5, min_visible=0.5):
    """业务口径：可采摘果实(visible>=0.5)的漏检率、误检率。

    漏检：GT 可采摘果实没有任何 conf>=conf 的预测框与其 IoU>=iou。
    误检：conf>=conf 的预测框没有命中任何可采摘 GT。
    """
    gt_pick = []
    for im_id, g in gts.items():
        vis = np.asarray(g.get("visible", np.ones(len(g["classes"]))), dtype=np.float64)
        for i in range(len(g["classes"])):
            if vis[i] >= min_visible:
                gt_pick.append((im_id, np.asarray(g["boxes"], dtype=np.float32)[i]))
    matched = [False] * len(gt_pick)
    n_pred = 0
    n_fp = 0
    for p in preds:
        keep = p["scores"] >= conf
        if not np.any(keep):
            continue
        pb = p["boxes"][keep]
        for b in pb:
            n_pred += 1
            hit = False
            best = -1
            best_iou = 0.0
            for k, (im_id, gb) in enumerate(gt_pick):
                if im_id != p["image_id"] or matched[k]:
                    continue
                iou_v = float(iou_matrix(b[None, :], gb[None, :])[0, 0])
                if iou_v > best_iou:
                    best_iou, best = iou_v, k
            if best >= 0 and best_iou >= iou:
                matched[best] = True
                hit = True
            if not hit:
                n_fp += 1
    miss = sum(1 for m in matched if not m)
    miss_rate = miss / len(gt_pick) if gt_pick else 0.0
    fp_rate = n_fp / n_pred if n_pred else 0.0
    return {"miss_count": miss, "gt_pickable": len(gt_pick),
            "miss_rate": float(miss_rate), "fp_count": n_fp,
            "pred_count": n_pred, "fp_rate": float(fp_rate)}
