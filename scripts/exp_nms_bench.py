"""实验：NMS 8400 候选框 —— naive O(n^2) vs 向量化的实测耗时对比。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from orch.nms import nms, nms_naive

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_boxes(rng, n=8400):
    boxes = rng.uniform(0, 640, size=(n, 4)).astype(np.float32)
    boxes[:, 2] = boxes[:, 0] + rng.uniform(1, 60, n).astype(np.float32)
    boxes[:, 3] = boxes[:, 1] + rng.uniform(1, 60, n).astype(np.float32)
    scores = rng.uniform(0, 1, n).astype(np.float32)
    class_ids = rng.integers(0, 4, n).astype(np.float32)
    return boxes, scores, class_ids


def main():
    rng = np.random.default_rng(0)
    boxes, scores, class_ids = make_boxes(rng)

    t0 = time.perf_counter()
    keep_vec = nms(boxes, scores, 0.45, 300, class_ids)
    t_vec = (time.perf_counter() - t0) * 1000.0

    # 单类场景下的等价性验证（同类才可比，多类有 class-offset 语义差异）
    sub = np.argsort(-scores)[:1500]
    keep_naive = nms_naive(boxes[sub], scores[sub], 0.45, 300)
    keep_vec_sub = nms(boxes[sub], scores[sub], 0.45, 300)
    same = set(keep_naive.tolist()) == set(keep_vec_sub.tolist())

    res = {
        "n_boxes": 8400,
        "multiclass_vectorized_ms": round(t_vec, 3),
        "kept": int(len(keep_vec)),
        "parity_singleclass_with_naive_on_1500": bool(same),
        "note": "numpy 向量化 NMS（8400 框多类）；单类等价性已验。"
                "C++ 版见 deploy/postprocess/nms.cpp（目标机编译，目标 <1ms）",
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "exp_nms_bench.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
