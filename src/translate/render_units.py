#!/usr/bin/env python3
"""
渲染相关的单位归一化与转换工具。
"""
from __future__ import annotations

import re
from typing import Union


_LENGTH_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*([A-Za-z]+)?\s*$")
_UNIT_TO_BP = {
    "bp": 1.0,
    "pt": 72.0 / 72.27,
    "in": 72.0,
    "cm": 72.0 / 2.54,
    "mm": 72.0 / 25.4,
    "px": 72.0 / 96.0,
}


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
