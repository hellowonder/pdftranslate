#!/usr/bin/env python3
from __future__ import annotations

from PIL import Image

from ocr_deepseek_postprocess import build_deepseek_page_markdown


def process_ocr_page_content(
    content: str,
    img: Image.Image,
    image_output_dir: str,
    page_number: int,
) -> str:
    return build_deepseek_page_markdown(
        raw_text=content,
        image=img,
        image_output_dir=image_output_dir,
        page_number=page_number,
    )
