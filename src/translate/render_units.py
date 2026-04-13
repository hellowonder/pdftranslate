#!/usr/bin/env python3
"""
渲染相关的单位归一化与转换工具。
"""
from __future__ import annotations

import re
from math import sqrt
from typing import Optional, Union


_LENGTH_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*([A-Za-z]+)?\s*$")
_UNIT_TO_BP = {
    "bp": 1.0,
    "pt": 72.0 / 72.27,
    "in": 72.0,
    "cm": 72.0 / 2.54,
    "mm": 72.0 / 25.4,
    "px": 72.0 / 96.0,
}

_PX_TO_PT = 72.0 / 96.0
_SMALL_PAGE_REFERENCE_PT = (431.0, 649.0)
_LARGE_PAGE_REFERENCE_PT = (612.0, 792.0)
_SMALL_PAGE_MARGIN_IN = 0.3
_LARGE_PAGE_MARGIN_IN = 0.8
_MAX_DYNAMIC_MARGIN_IN = 1.0


def normalize_margin_value(margin: Union[str, int, float]) -> str:
    """
    将边距参数归一化为带单位的长度字符串。

    纯数字输入按像素处理，以兼容旧命令行参数。
    """
    if isinstance(margin, (int, float)):
        return f"{margin}px"

    text = str(margin).strip()
    if not text:
        raise ValueError("margin cannot be empty")

    match = _LENGTH_RE.fullmatch(text)
    if not match:
        raise ValueError(f"Invalid margin value: {margin!r}")

    number, unit = match.groups()
    if unit is None:
        return f"{number}px"

    normalized_unit = unit.lower()
    if normalized_unit not in _UNIT_TO_BP:
        raise ValueError(
            f"Unsupported margin unit: {unit!r}. Supported units: {', '.join(sorted(_UNIT_TO_BP))}."
        )
    return f"{number}{normalized_unit}"


def page_size_to_margin(width: Union[int, float], height: Union[int, float]) -> str:
    """
    根据页面尺寸动态推导边距。

    输入宽高按渲染阶段实际使用的像素尺寸处理，内部换算为 pt 后，
    使用页面几何平均边长做线性插值，并限制在一个温和的范围内。
    """
    width_px = max(float(width), 0.0)
    height_px = max(float(height), 0.0)
    if width_px <= 0 or height_px <= 0:
        return f"{_SMALL_PAGE_MARGIN_IN:.1f}in"

    page_scale_pt = sqrt(width_px * height_px) * _PX_TO_PT
    small_scale_pt = sqrt(_SMALL_PAGE_REFERENCE_PT[0] * _SMALL_PAGE_REFERENCE_PT[1])
    large_scale_pt = sqrt(_LARGE_PAGE_REFERENCE_PT[0] * _LARGE_PAGE_REFERENCE_PT[1])

    if large_scale_pt <= small_scale_pt:
        margin_in = _SMALL_PAGE_MARGIN_IN
    else:
        ratio = (page_scale_pt - small_scale_pt) / (large_scale_pt - small_scale_pt)
        margin_in = _SMALL_PAGE_MARGIN_IN + ratio * (_LARGE_PAGE_MARGIN_IN - _SMALL_PAGE_MARGIN_IN)

    margin_in = min(max(margin_in, _SMALL_PAGE_MARGIN_IN), _MAX_DYNAMIC_MARGIN_IN)
    formatted = f"{margin_in:.2f}".rstrip("0").rstrip(".")
    return f"{formatted}in"


def resolve_page_margin_value(
    margin: Optional[Union[str, int, float]],
    page_width: Union[int, float],
    page_height: Union[int, float],
) -> str:
    """
    返回最终可用的边距值。

    显式指定的 margin 原样归一化；未指定时根据页面尺寸动态计算。
    """
    if margin is None:
        return page_size_to_margin(page_width, page_height)
    return normalize_margin_value(margin)


def margin_to_css_value(margin: Union[str, int, float]) -> str:
    """
    返回可直接用于 CSS 的边距长度。
    """
    return normalize_margin_value(margin)


def margin_to_latex_value(margin: Union[str, int, float]) -> str:
    """
    将边距转换为 Pandoc/LaTeX geometry 可接受的长度字符串。

    统一输出为 `bp`，以避免 `px` 在不同 TeX 引擎中的兼容性问题。
    """
    normalized = normalize_margin_value(margin)
    match = _LENGTH_RE.fullmatch(normalized)
    if not match:
        raise ValueError(f"Invalid margin value: {margin!r}")

    number_text, unit = match.groups()
    assert unit is not None
    value = float(number_text)
    unit_factor = _UNIT_TO_BP[unit.lower()]
    return f"{value * unit_factor:.3f}bp"


def normalize_font_size_value(font_size: Union[str, int, float]) -> str:
    """
    将字号参数归一化为带单位的长度字符串。

    纯数字输入按 pt 处理，以便 PDF 输出使用更稳定的物理字号。
    """
    if isinstance(font_size, (int, float)):
        return f"{font_size}pt"

    text = str(font_size).strip()
    if not text:
        raise ValueError("font_size cannot be empty")

    match = _LENGTH_RE.fullmatch(text)
    if not match:
        raise ValueError(f"Invalid font size value: {font_size!r}")

    number, unit = match.groups()
    if unit is None:
        return f"{number}pt"

    normalized_unit = unit.lower()
    if normalized_unit not in _UNIT_TO_BP:
        raise ValueError(
            f"Unsupported font size unit: {unit!r}. Supported units: {', '.join(sorted(_UNIT_TO_BP))}."
        )
    return f"{number}{normalized_unit}"


def font_size_to_css_value(font_size: Union[str, int, float]) -> str:
    """
    返回可直接用于 CSS 的字号长度。
    """
    return normalize_font_size_value(font_size)


def font_size_to_latex_value(font_size: Union[str, int, float]) -> str:
    """
    将字号转换为 Pandoc/LaTeX 可接受的长度字符串。

    统一输出为 `bp`，避免不同单位在 TeX 中的兼容差异。
    """
    normalized = normalize_font_size_value(font_size)
    match = _LENGTH_RE.fullmatch(normalized)
    if not match:
        raise ValueError(f"Invalid font size value: {font_size!r}")

    number_text, unit = match.groups()
    assert unit is not None
    value = float(number_text)
    unit_factor = _UNIT_TO_BP[unit.lower()]
    return f"{value * unit_factor:.3f}bp"
