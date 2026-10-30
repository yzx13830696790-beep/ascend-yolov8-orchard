"""预处理契约：训练侧、ONNX 侧、AIPP 侧必须逐位对齐。

这个模块是整个项目"精度对齐"的唯一真源（single source of truth）：

  * letterbox 的缩放比与 padding —— 决定检测框坐标还原公式
  * 归一化的 (mean, std) —— 决定 AIPP 的 mean/var 取值
  * 色域矩阵 BT.601 / BT.709 —— 决定 DVPP 输出 YUV 如何转回 RGB

任务书里的三处硬伤（AIPP 写 RGB、force_fp16 配 high_precision、
mean/var 拍脑袋）全部落在本文件定义的契约上，所以这里每个函数都带
可测试的数学定义，而不是"调个库就完事"。
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - 环境相关
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# ImageNet 归一化常数（YOLOv8 训练的默认值）
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PAD_VALUE = 114


# --------------------------------------------------------------------------
# letterbox
# --------------------------------------------------------------------------
def letterbox_params(h, w, new_shape=(640, 640), scaleup=True):
    """返回 (r, new_unpad_wh, dw, dh)。

    与 ultralytics 的 `LetterBox` 完全一致：
        r = min(new_h / h, new_w / w)
        new_unpad = (round(w * r), round(h * r))
        dw = (new_w - new_unpad_w) / 2, dh 同理
    注意 dw/dh 是浮点的一半 padding，不是取整后的像素偏移 —— 量化的
    部署里最常见的 0.5~1px 系统性偏移就是这里写成 int 造成的。
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))
    dw = (new_shape[1] - new_unpad_w) / 2.0
    dh = (new_shape[0] - new_unpad_h) / 2.0
    return r, (new_unpad_w, new_unpad_h), dw, dh


def _resize(im, out_w, out_h):
    if cv2 is not None:
        if im.shape[1] == out_w and im.shape[0] == out_h:
            return im.copy()
        return cv2.resize(im, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return _resize_bilinear_np(im, out_h, out_w)


def _resize_bilinear_np(im, out_h, out_w):
    """无 cv2 时的可复现 bilinear（align_corners=False）。"""
    h, w = im.shape[:2]
    if (h, w) == (out_h, out_w):
        return im.copy()
    sy = (np.arange(out_h) + 0.5) * (h / out_h) - 0.5
    sx = (np.arange(out_w) + 0.5) * (w / out_w) - 0.5
    sy = np.clip(sy, 0, h - 1)
    sx = np.clip(sx, 0, w - 1)
    y0 = np.floor(sy).astype(np.int32)
    x0 = np.floor(sx).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (sy - y0)[:, None, None].astype(np.float32)
    wx = (sx - x0)[None, :, None].astype(np.float32)
    f = im.astype(np.float32)
    top = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
    bot = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
    out = top * (1 - wy) + bot * wy
    if im.dtype == np.uint8:
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return out.astype(im.dtype)


def letterbox(im, new_shape=(640, 640), color=PAD_VALUE, scaleup=True):
    """返回 (padded_image, r, (dw, dh))。"""
    h, w = im.shape[:2]
    r, (nw, nh), dw, dh = letterbox_params(h, w, new_shape, scaleup)
    if (w, h) != (nw, nh):
        im = _resize(im, nw, nh)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if im.ndim == 3:
        pad = ((top, bottom), (left, right), (0, 0))
        fill = (color, color, color) if isinstance(color, int) else tuple(color)
    else:
        pad = ((top, bottom), (left, right))
        fill = color
    out = np.full((new_shape[0], new_shape[1]) + im.shape[2:], fill, dtype=im.dtype)
    out[top:top + im.shape[0], left:left + im.shape[1]] = im
    return out, r, (dw, dh)


def scale_boxes_back(boxes, r, dw, dh):
    """把 640 输入坐标系下的 xyxy 还原回原图坐标系。"""
    b = np.asarray(boxes, dtype=np.float32).copy()
    if b.size == 0:
        return b.reshape(0, 4)
    b[:, [0, 2]] = (b[:, [0, 2]] - dw) / r
    b[:, [1, 3]] = (b[:, [1, 3]] - dh) / r
    return b


def scale_boxes_forward(boxes, r, dw, dh):
    b = np.asarray(boxes, dtype=np.float32).copy()
    b[:, [0, 2]] = b[:, [0, 2]] * r + dw
    b[:, [1, 3]] = b[:, [1, 3]] * r + dh
    return b


# --------------------------------------------------------------------------
# 色域：DVPP 出 YUV420SP，AIPP 里 csc 矩阵必须与"训练时 RGB 的来源"一致
# --------------------------------------------------------------------------
# BT.601（SD/传统摄像头默认）与 BT.709（HD）的 YUV->RGB 系数
CSC = {
    "BT601": np.array([[1.0, 0.0, 1.402],
                       [1.0, -0.344136, -0.714136],
                       [1.0, 1.772, 0.0]], dtype=np.float32),
    "BT709": np.array([[1.0, 0.0, 1.5748],
                       [1.0, -0.1873, -0.4681],
                       [1.0, 1.8556, 0.0]], dtype=np.float32),
}
# 对应的 RGB->YUV 矩阵（用于把合成数据模拟成 DVPP 的输出）
RGB2YUV = {
    "BT601": np.array([[0.299, 0.587, 0.114],
                       [-0.168736, -0.331264, 0.5],
                       [0.5, -0.418688, -0.081312]], dtype=np.float32),
    "BT709": np.array([[0.2126, 0.7152, 0.0722],
                       [-0.1146, -0.3854, 0.5],
                       [0.5, -0.4542, -0.0458]], dtype=np.float32),
}


def rgb_to_yuv420sp(rgb, matrix="BT601", full_range=False):
    """RGB(uint8 HxWx3) -> YUV420SP(NV12 = Y plane + 交错 UV)。

    这里模拟 DVPP 硬解码的输出，用于量化"色域/范围错配"的精度代价。
    """
    m = RGB2YUV[matrix]
    f = rgb.astype(np.float32)
    yuv = f.reshape(-1, 3) @ m.T
    if not full_range:  # video range: Y 16-235, UV 16-240
        yuv[:, 0] = yuv[:, 0] * (219.0 / 255.0) + 16.0
        yuv[:, 1:] = yuv[:, 1:] * (224.0 / 255.0) + 128.0
    yuv = np.clip(np.round(yuv), 0, 255).astype(np.uint8).reshape(rgb.shape)
    return yuv


def yuv420sp_to_rgb(nv12, matrix="BT601", full_range=False):
    """YUV420SP -> RGB，`matrix` 参数就是 AIPP 里的 csc 矩阵。"""
    m = CSC[matrix]
    f = nv12.astype(np.float32)
    if not full_range:
        f = f.copy()
        f[..., 0] = (f[..., 0] - 16.0) * (255.0 / 219.0)
        f[..., 1:] = (f[..., 1:] - 128.0) * (255.0 / 224.0)
    rgb = f.reshape(-1, 3) @ m.T
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8).reshape(nv12.shape)


# --------------------------------------------------------------------------
# AIPP 归一化：uint8 输入上的 (x - mean) * var 仿射
# --------------------------------------------------------------------------
def aipp_affine(mean=MEAN, std=STD):
    """把训练侧的 (x/255 - mean) / std 折算成 AIPP 的 (x - mean') * var'。

    AIPP 直接吃 uint8，所以：
        x/255/s - mean/s = (x - 255*mean) * (1/(255*s))
    于是 mean_aipp = 255*mean, var_aipp = 1/(255*std)。

    这一步是"归一化系数写反/忘除 255"事故的根因所在：
    直接把 mean/std 填进 cfg，等于把输入放大了 255 倍。
    """
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    return 255.0 * mean, 1.0 / (255.0 * std)


def apply_aipp(rgb_uint8, mean=MEAN, std=STD, mean_aipp=None, var_aipp=None,
               round_affine=True):
    """模拟 AIPP：NCHW float32 输出。round_affine 复现 cfg 只支持整数的截断。"""
    if mean_aipp is None or var_aipp is None:
        m, v = aipp_affine(mean, std)
    else:
        m, v = np.asarray(mean_aipp, np.float64), np.asarray(var_aipp, np.float64)
    if round_affine:
        # aipp.cfg 的 mean_chn_i 是整数、var_chn_i 保留 8 位小数
        m = np.round(m)
        v = np.round(v, 8)
    x = rgb_uint8.astype(np.float32)
    out = (x - m.astype(np.float32)) * v.astype(np.float32)
    return np.ascontiguousarray(out.transpose(2, 0, 1)[None, ...])


def normalize_nchw(rgb_uint8, mean=MEAN, std=STD):
    """训练侧标准归一化（float64 精度），作为 AIPP 的对齐基准。"""
    x = rgb_uint8.astype(np.float64) / 255.0
    out = (x - np.asarray(mean)) / np.asarray(std)
    return np.ascontiguousarray(out.transpose(2, 0, 1)[None, ...].astype(np.float32))


def preprocess(im_bgr, new_shape=(640, 640), input_format="RGB"):
    """端到端：BGR(HxWx3 uint8) -> letterbox -> RGB -> NCHW float32。

    返回 (tensor, r, (dw, dh))，r/dw/dh 用于把框还原。
    """
    rgb = im_bgr[..., ::-1] if input_format == "RGB" else im_bgr
    padded, r, pad = letterbox(np.ascontiguousarray(rgb), new_shape)
    tensor = normalize_nchw(padded)
    return tensor, r, pad
