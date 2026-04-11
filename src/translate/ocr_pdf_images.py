#!/usr/bin/env python3
"""
分页文档页选择与光栅化相关的辅助函数。
"""
from __future__ import annotations

import io
from typing import List, Optional, Set

import fitz
from PIL import Image


def pdf_to_images_high_quality(
    pdf_path: str,
    dpi: int = 144,
    page_numbers: Optional[List[int]] = None,
) -> List[Image.Image]:
    """
    以指定 DPI 将分页文档指定页面渲染为 PIL 图像列表。

    参数:
        pdf_path: 输入文档路径，例如 PDF 或 DjVu。
        dpi: 渲染分辨率，默认 144。
        page_numbers: 可选的 1 基页码列表；为空时渲染全部页面。
    返回:
        List[Image.Image]: 与 page_numbers 顺序一致的页面图像列表；为空时与原始页顺序一致。
    """
    images: List[Image.Image] = []
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    if page_numbers is None:
        selected_page_numbers = list(range(1, pdf_document.page_count + 1))
    else:
        selected_page_numbers = page_numbers

    for page_number in selected_page_numbers:
        page = pdf_document[page_number - 1]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        images.append(img)
    pdf_document.close()
    return images


def resolve_page_numbers(pages_arg: Optional[str], total_pages: int) -> List[int]:
    if not pages_arg:
        return list(range(1, total_pages + 1))
    selected: Set[int] = set()
    for part in pages_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                raise ValueError(f"Invalid page range: '{part}'")
            if start > end:
                start, end = end, start
            for num in range(start, end + 1):
                selected.add(num)
        else:
            try:
                selected.add(int(part))
            except ValueError:
                raise ValueError(f"Invalid page number: '{part}'")
    filtered = sorted(num for num in selected if 1 <= num <= total_pages)
    if not filtered:
        raise ValueError("No valid page numbers selected after filtering.")
    return filtered
