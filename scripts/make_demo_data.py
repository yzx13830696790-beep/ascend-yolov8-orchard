"""合成果园数据生成器 —— 让整条评测链路在没有真实数据/硬件时也能实跑。

用途：把 data/ 下的图像+标签生成出来，供 prepare_data / run_eval /
exp_color_matrix 走通。生成的数据**只用于校验评测协议与后处理正确性**，
不是训练/量化用的真实数据（README 里已声明）。

类别：0 apple, 1 apple_occluded, 2 branch, 3 leaf_cluster
标签第 6 列为 visible_ratio（0~1）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

CLASSES = ["apple", "apple_occluded", "branch", "leaf_cluster"]
SCENES = ["sunny", "backlit", "overcast", "night", "rain", "blur"]


def _rng(seed):
    return np.random.default_rng(seed)


def _draw_background(img, scene, rng):
    h, w = img.shape[:2]
    # 天空->地面 渐变
    for y in range(h):
        t = y / h
        if scene == "night":
            c = (int(8 + 20 * t), int(8 + 22 * t), int(24 + 30 * t))
        elif scene == "backlit":
            c = (int(60 + 120 * t), int(70 + 130 * t), int(80 + 120 * t))
        elif scene == "overcast":
            g = int(90 + 60 * t)
            c = (g, g + 4, g + 6)
        else:
            c = (int(90 + 60 * t), int(120 + 80 * t), int(160 + 60 * t))
        img[y, :] = c
    # 地面绿色区域
    gh = int(h * 0.72)
    img[gh:, :] = np.array([(40, 80, 30)], dtype=np.uint8)


def _draw_branches(img, rng):
    h, w = img.shape[:2]
    n = int(rng.integers(1, 4))
    boxes = []
    for _ in range(n):
        y0 = int(rng.integers(h * 0.1, h * 0.6))
        x0 = int(rng.integers(0, w))
        ang = rng.uniform(-0.7, 0.7)
        length = rng.integers(120, 280)
        x1 = int(x0 + length * np.cos(ang))
        y1 = int(y0 + length * np.sin(ang))
        cv2.line(img, (x0, y0), (x1, y1), (40, 30, 24), 8)
        cv2.line(img, (x0, y0), (x1, y1), (70, 50, 38), 4)
        boxes.append((x0, y0, x1, y1))
    return boxes


def _draw_leaf_clusters(img, rng):
    h, w = img.shape[:2]
    n = int(rng.integers(2, 6))
    boxes = []
    for _ in range(n):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(int(h * 0.5), h))
        for _ in range(int(rng.integers(4, 12))):
            rx = int(rng.integers(8, 30))
            ry = int(rng.integers(6, 20))
            dx = int(rng.integers(-40, 40))
            dy = int(rng.integers(-30, 30))
            c = (int(rng.integers(20, 60)), int(rng.integers(80, 150)),
                 int(rng.integers(20, 60)))
            cv2.ellipse(img, (cx + dx, cy + dy), (rx, ry),
                        float(rng.uniform(0, 180)), 0, 360, c, -1)
        x1 = max(0, cx - 55); y1 = max(0, cy - 45)
        x2 = min(w, cx + 55); y2 = min(h, cy + 45)
        boxes.append((x1, y1, x2, y2))
    return boxes


def _draw_apple(img, cx, cy, r, occluded, rng, scene):
    # 苹果主体：径向渐变（亮心 + 暗边）
    base = np.array([(40, 60, 200)], dtype=np.float32)  # BGR 红
    for rr in range(r, 0, -1):
        t = rr / max(r, 1)
        shade = 1.0 - 0.35 * (1 - t)
        if occluded:
            shade *= 0.7
        c = tuple(np.clip(base[0] * shade, 0, 255).astype(np.uint8))
        cv2.circle(img, (cx, cy), int(rr), (int(c[0]), int(c[1]), int(c[2])), -1)
    # 高光
    cv2.circle(img, (cx - int(r * 0.35), cy - int(r * 0.4)),
               max(1, int(r * 0.25)), (230, 230, 235), -1)
    # 果柄
    cv2.line(img, (cx, cy - r), (cx + int(r * 0.2), cy - r - int(r * 0.5)),
             (30, 60, 30), 2)


def _occlude_with_leaf(img, cx, cy, r, rng):
    ox = int(cx + rng.uniform(-r, r))
    oy = int(cy + rng.uniform(-r, r))
    rx = int(rng.integers(r, r * 2))
    ry = int(rng.integers(int(r * 0.5), r))
    c = (int(rng.integers(20, 60)), int(rng.integers(90, 160)), int(rng.integers(20, 60)))
    cv2.ellipse(img, (ox, oy), (rx, ry), float(rng.uniform(0, 180)), 0, 360, c, -1)


def _apply_scene(img, scene, rng):
    h, w = img.shape[:2]
    if scene == "backlit":
        img = cv2.addWeighted(img, 0.55, np.full_like(img, (60, 60, 80)), 0.45, 0)
        # 逆光轮廓：压暗中心
        mask = np.zeros((h, w), np.float32)
        cv2.circle(mask, (w // 2, int(h * 0.35)), int(w * 0.7), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 80)
        img = (img.astype(np.float32) * (1 - 0.3 * mask[..., None])).astype(np.uint8)
    elif scene == "overcast":
        img = cv2.addWeighted(img, 0.7, np.full_like(img, (120, 120, 122)), 0.3, 0)
    elif scene == "night":
        img = (img.astype(np.float32) * 0.25).astype(np.uint8)
        img[..., 0] = np.clip(img[..., 0] + 20, 0, 255)
    elif scene == "rain":
        for _ in range(60):
            x = int(rng.integers(0, w)); y = int(rng.integers(0, h))
            cv2.line(img, (x, y), (x - int(rng.integers(3, 8)), y + int(rng.integers(10, 20))),
                     (180, 190, 200), 1)
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif scene == "blur":
        img = cv2.GaussianBlur(img, (15, 15), 0)
    # 统一噪声
    if scene not in ("blur",):
        noise = rng.integers(0, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def generate_one(idx, scene, rng, size=(640, 480)):
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _draw_background(img, scene, rng)
    branch_boxes = _draw_branches(img, rng)
    leaf_boxes = _draw_leaf_clusters(img, rng)

    labels = []
    n_apple = int(rng.integers(6, 13))
    for _ in range(n_apple):
        r = int(rng.integers(14, 30))
        cx = int(rng.integers(r, w - r))
        cy = int(rng.integers(int(h * 0.35), h - r))
        occluded = rng.random() < 0.3
        _draw_apple(img, cx, cy, r, occluded, rng, scene)
        if occluded:
            _occlude_with_leaf(img, cx, cy, r, rng)
            vis = float(rng.uniform(0.15, 0.45))
            cls = 1
        else:
            vis = float(rng.uniform(0.9, 1.0))
            cls = 0
        labels.append((cls, cx / w, cy / h, (2 * r) / w, (2 * r) / h, vis))

    for (x0, y0, x1, y1) in branch_boxes:
        bx = (min(x0, x1) + max(x0, x1)) / 2 / w
        by = (min(y0, y1) + max(y0, y1)) / 2 / h
        bw = abs(x1 - x0) / w + 0.05
        bh = abs(y1 - y0) / h + 0.05
        labels.append((2, bx, by, bw, bh, 1.0))
    for (x1, y1, x2, y2) in leaf_boxes:
        bx = (x1 + x2) / 2 / w
        by = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        labels.append((3, bx, by, bw, bh, 1.0))

    img = _apply_scene(img, scene, rng)
    return img, labels


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(root, "data", "images")
    lab_dir = os.path.join(root, "data", "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    per_scene = int(os.environ.get("N_PER_SCENE", "15"))
    rng = _rng(0)
    total = 0
    for scene in SCENES:
        for i in range(per_scene):
            img, labels = generate_one(total, scene, rng)
            fn = f"{scene}_{i:03d}"
            cv2.imwrite(os.path.join(img_dir, fn + ".jpg"), img)
            with open(os.path.join(lab_dir, fn + ".txt"), "w") as f:
                for lab in labels:
                    f.write(" ".join(f"{x:.6g}" for x in lab) + "\n")
            total += 1
    print(f"generated {total} synthetic images -> {img_dir}")


if __name__ == "__main__":
    main()
