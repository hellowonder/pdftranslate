#!/usr/bin/env python3
"""
PDF crop helpers shared by the standalone crop tool and tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import fitz


MIN_SELECTION_SIZE = 4.0


@dataclass(frozen=True)
class CropMargins:
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True)
class NormalizedMargins:
    left_ratio: float
    top_ratio: float
    right_ratio: float
    bottom_ratio: float


def normalize_selection(selection: fitz.Rect, base_rect: fitz.Rect) -> fitz.Rect:
    rect = fitz.Rect(selection)
    rect = fitz.Rect(
        max(base_rect.x0, min(rect.x0, base_rect.x1)),
        max(base_rect.y0, min(rect.y0, base_rect.y1)),
        max(base_rect.x0, min(rect.x1, base_rect.x1)),
        max(base_rect.y0, min(rect.y1, base_rect.y1)),
    ).normalize()
    if rect.width < MIN_SELECTION_SIZE or rect.height < MIN_SELECTION_SIZE:
        raise ValueError("Crop selection is too small.")
    return rect


def selection_to_margins(selection: fitz.Rect, base_rect: fitz.Rect) -> CropMargins:
    rect = normalize_selection(selection, base_rect)
    return CropMargins(
        left=rect.x0 - base_rect.x0,
        top=rect.y0 - base_rect.y0,
        right=base_rect.x1 - rect.x1,
        bottom=base_rect.y1 - rect.y1,
    )


def margins_to_selection(margins: CropMargins, base_rect: fitz.Rect) -> fitz.Rect:
    selection = fitz.Rect(
        base_rect.x0 + margins.left,
        base_rect.y0 + margins.top,
        base_rect.x1 - margins.right,
        base_rect.y1 - margins.bottom,
    )
    return normalize_selection(selection, base_rect)


def margins_to_normalized(margins: CropMargins, base_rect: fitz.Rect) -> NormalizedMargins:
    width = base_rect.width
    height = base_rect.height
    if width <= 0 or height <= 0:
        raise ValueError("Base page rect must have positive size.")
    return NormalizedMargins(
        left_ratio=margins.left / width,
        top_ratio=margins.top / height,
        right_ratio=margins.right / width,
        bottom_ratio=margins.bottom / height,
    )


def normalized_to_margins(margins: NormalizedMargins, base_rect: fitz.Rect) -> CropMargins:
    return CropMargins(
        left=base_rect.width * margins.left_ratio,
        top=base_rect.height * margins.top_ratio,
        right=base_rect.width * margins.right_ratio,
        bottom=base_rect.height * margins.bottom_ratio,
    )


def selection_to_normalized(selection: fitz.Rect, base_rect: fitz.Rect) -> NormalizedMargins:
    return margins_to_normalized(selection_to_margins(selection, base_rect), base_rect)


def selection_from_normalized(margins: NormalizedMargins, base_rect: fitz.Rect) -> fitz.Rect:
    return margins_to_selection(normalized_to_margins(margins, base_rect), base_rect)


def remap_selection(selection: fitz.Rect, source_rect: fitz.Rect, target_rect: fitz.Rect) -> fitz.Rect:
    normalized = selection_to_normalized(selection, source_rect)
    return selection_from_normalized(normalized, target_rect)


def canvas_rect_to_pdf_rect(canvas_rect: tuple[float, float, float, float], zoom: float) -> fitz.Rect:
    if zoom <= 0:
        raise ValueError("Zoom must be positive.")
    x0, y0, x1, y1 = canvas_rect
    return fitz.Rect(x0 / zoom, y0 / zoom, x1 / zoom, y1 / zoom).normalize()


def pdf_rect_to_canvas_rect(pdf_rect: fitz.Rect, zoom: float) -> tuple[float, float, float, float]:
    if zoom <= 0:
        raise ValueError("Zoom must be positive.")
    rect = fitz.Rect(pdf_rect)
    return (rect.x0 * zoom, rect.y0 * zoom, rect.x1 * zoom, rect.y1 * zoom)


def save_cropped_pdf(input_path: str, output_path: str, page_selections: Mapping[int, fitz.Rect]) -> None:
    doc = fitz.open(input_path)
    try:
        for page_index, selection in page_selections.items():
            if not 0 <= page_index < doc.page_count:
                raise IndexError(f"Page index out of range: {page_index}")
            page = doc[page_index]
            base_rect = fitz.Rect(page.rect)
            normalized = normalize_selection(selection, base_rect)
            page.set_cropbox(normalized)
        doc.save(output_path)
    finally:
        doc.close()
