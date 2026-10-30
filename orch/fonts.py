"""中文字体探测：有中文字体就用中文标签，没有就自动退回英文。

为什么需要这一层：把 ``font.sans-serif`` 写死成 "Microsoft YaHei" 时，
Windows 本机看着一切正常，但一旦跑在 Linux 容器/昇腾训练机上（那里通常只装
DejaVu 这类西文字体），所有中文标签会静默渲染成方块 —— 图照样生成、退出码
照样是 0，只是没人能读懂。这类问题不会报错，只能靠"在目标环境里看一眼"发现。

所以这里改成运行时探测：命中候选字体就用中文，否则整体切英文文案。
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

# 按平台常见程度排序：Windows -> macOS -> Linux 发行版常见 CJK 字体
CJK_CANDIDATES = (
    "Microsoft YaHei",      # Windows
    "SimHei",               # Windows
    "SimSun",               # Windows
    "PingFang SC",          # macOS
    "Hiragino Sans GB",     # macOS
    "STHeiti",              # macOS
    "Noto Sans CJK SC",     # Debian/Ubuntu 常见
    "Noto Sans SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",    # 国产发行版
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
)


def _detect_cjk_font():
    """返回第一个可用的 CJK 字体名；一个都没有则返回 None。"""
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return None
    for name in CJK_CANDIDATES:
        if name in available:
            return name
    return None


CJK_FONT = (None if os.environ.get("ORCH_FORCE_NO_CJK", "")
            .lower() in ("1", "true", "yes")
            else _detect_cjk_font())
HAS_CJK = CJK_FONT is not None

if HAS_CJK:
    matplotlib.rcParams["font.sans-serif"] = [CJK_FONT, "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def t(zh, en):
    """按字体可用性二选一：有中文字体用中文，否则用英文。"""
    return zh if HAS_CJK else en


def safe(text, fallback_ascii):
    """外部传入的文案（可能是中文）在无中文字体时换成 ASCII 兜底。"""
    if HAS_CJK:
        return text
    try:
        text.encode("ascii")
        return text
    except (UnicodeEncodeError, AttributeError):
        return fallback_ascii
