"""ATC / AIPP：命令与配置的"正确"构造器 + 三条硬伤的校验器。

规划书第九节的三处技术细节，全部固化为可执行的校验规则：

  1. Atlas 310B4 需要 CANN >= 7.0.RC1，6.3.RC2 不认识这个 soc_version
  2. 走 DVPP 硬解码路径，AIPP 必须 input_format=YUV420SP_U8 + csc 矩阵，
     写 RGB 会让预处理退回 Host，DVPP 白用
  3. force_fp16 与 op_select_implmode=high_precision 语义冲突

不臆造 ATC 之外的参数：这里只生成官方文档里真实存在的字段。
"""
from __future__ import annotations

import re

from .preprocess import MEAN, STD, aipp_affine


# --------------------------------------------------------------------------
# 版本比较
# --------------------------------------------------------------------------
def parse_cann_version(v):
    m = re.match(r"(\d+)\.(\d+)(?:\.(\w+))?(?:\.(\w+))?", str(v).strip())
    if not m:
        return (0, 0, 0)
    major, minor = int(m.group(1)), int(m.group(2))
    patch = m.group(3) or "0"
    if patch == "RC1":
        patch_num = 1
    elif patch.startswith("RC"):
        patch_num = int(patch[2:]) if patch[2:].isdigit() else 99
    elif patch.isdigit():
        patch_num = int(patch)
    else:
        patch_num = 0
    return (major, minor, patch_num)


def cann_ge(a, b):
    return parse_cann_version(a) >= parse_cann_version(b)


# 芯片 -> 最低 CANN 版本
MIN_CANN = {
    "Ascend310B4": "7.0.RC1",
    "Ascend310P3": "7.0.RC1",
    "Ascend910B4": "8.0.RC1",
}


# --------------------------------------------------------------------------
# AIPP 配置生成
# --------------------------------------------------------------------------
def build_aipp_cfg(soc_version="Ascend310B4", pipeline="dvpp",
                   matrix="BT601", mean=MEAN, std=STD, input_size=(640, 640)):
    """生成 aipp.cfg 文本。

    matrix 必须是 BT601 或 BT709 —— 与训练侧 RGB 来源一致；
    默认 BT601（DVPP 传统视频管线的默认值）。
    """
    m, v = aipp_affine(mean, std)
    m = [round(x) for x in m]
    v = [round(x, 8) for x in v]
    if matrix == "BT601":
        r2y = "0.299 0.587 0.114 -0.168736 -0.331264 0.5 0.5 -0.418688 -0.081312"
    else:
        r2y = "0.2126 0.7152 0.0722 -0.1146 -0.3854 0.5 0.5 -0.4542 -0.0458"
    inp = "YUV420SP_U8" if pipeline == "dvpp" else "RGB888_U8"
    csc = "true" if pipeline == "dvpp" else "false"
    lines = [
        "aipp_op {",
        "    aipp_mode : static",
        f"    input_format : {inp}",
        f"    csc_switch : {csc}",
        "    rbuv_swap_switch : false",
        f"    matrix_r2y : {r2y}",
        f"    src_image_size_w : {input_size[0]}",
        f"    src_image_size_h : {input_size[1]}",
        f"    mean_chn_0 : {m[0]}",
        f"    mean_chn_1 : {m[1]}",
        f"    mean_chn_2 : {m[2]}",
        f"    var_reci_chn_0 : {v[0]:.8f}",
        f"    var_reci_chn_1 : {v[1]:.8f}",
        f"    var_reci_chn_2 : {v[2]:.8f}",
        "}",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# ATC 命令
# --------------------------------------------------------------------------
def build_atc_cmd(model_onnx, output_om, soc_version, precision_mode="force_fp16",
                  aipp_cfg=None, out_nodes=None, log="error", extra=None):
    parts = [
        "atc",
        f"--model={model_onnx}",
        f"--output={output_om}",
        "--framework=5",
        f"--soc_version={soc_version}",
        f"--precision_mode={precision_mode}",
    ]
    if aipp_cfg:
        parts.append(f"--insert_op_conf={aipp_cfg}")
    if out_nodes:
        parts.append(f"--out_nodes={out_nodes}")
    parts.append(f"--log={log}")
    if extra:
        parts.extend(extra)
    return " \\\n  ".join(parts)


# --------------------------------------------------------------------------
# 校验器：返回 list[dict(level, msg)]
# --------------------------------------------------------------------------
def validate(soc_version="Ascend310B4", cann_version=None, pipeline="dvpp",
             input_format=None, precision_mode=None, op_select_implmode=None,
             matrix="BT601", train_matrix=None, mean=MEAN, std=STD,
             cfg_mean=None, cfg_std=None):
    errs = []
    if cann_version and soc_version in MIN_CANN:
        need = MIN_CANN[soc_version]
        if not cann_ge(cann_version, need):
            errs.append({"level": "error",
                         "msg": f"{soc_version} 需要 CANN >= {need}，"
                                f"当前 {cann_version} 的 ATC 不认识该 soc_version"})
    if pipeline == "dvpp" and input_format and input_format.upper() in ("RGB", "RGB888_U8"):
        errs.append({"level": "error",
                     "msg": "DVPP 硬解码输出 YUV420SP，AIPP 应设 "
                            "YUV420SP_U8 + csc_switch=true；写 RGB 会让预处理退回 Host"})
    if precision_mode == "force_fp16" and op_select_implmode == "high_precision":
        errs.append({"level": "error",
                     "msg": "force_fp16 与 op_select_implmode=high_precision 语义冲突："
                            "敏感算子会回落 FP32，抵消 FP16 的性能收益"})
    if train_matrix and matrix != train_matrix:
        errs.append({"level": "warning",
                     "msg": f"csc 矩阵 {matrix} 与训练侧 {train_matrix} 不一致，"
                            "mAP 可能掉 5~12pt"})
    # 归一化系数校验
    if cfg_mean is not None or cfg_std is not None:
        m_ref, v_ref = aipp_affine(mean, std)
        if cfg_mean is not None:
            cfg_mean = [float(x) for x in cfg_mean]
            if max(abs(cfg_mean[i] - m_ref[i]) for i in range(3)) > 1.0:
                errs.append({"level": "error",
                             "msg": "AIPP mean 与训练归一化不一致（疑似直接填了 0~1 的 mean，"
                                    "或忘乘 255），会导致整层系统性偏移"})
        if cfg_std is not None:
            cfg_std = [float(x) for x in cfg_std]
            if max(abs(cfg_std[i] - v_ref[i]) for i in range(3)) > 1e-3:
                errs.append({"level": "error",
                             "msg": "AIPP var 与训练归一化不一致（疑似把 std 当 var 填）"})
    return errs
