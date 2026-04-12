#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from typing import List

from PIL import Image

IMAGE_PLACEHOLDER_PATTERN = re.compile(r"!\[\]\(image_(\d+)_(\d+)_(\d+)_(\d+)\.png\)")
PLACEHOLDER_SCALE = 1000


def _to_pixel(coord: int, span: int) -> int:
    clamped = min(PLACEHOLDER_SCALE, max(0, coord))
    return int(clamped / PLACEHOLDER_SCALE * span)


def build_gemma_page_markdown(
    raw_text: str,
    image: Image.Image,
    image_output_dir: str,
    page_number: int,
) -> str:
    content = (raw_text or "").replace("<｜end▁of▁sentence｜>", "").strip()
    if not content:
        return ""

    os.makedirs(image_output_dir, exist_ok=True)
    image_width, image_height = image.size
    replacements: List[tuple[str, str]] = []

    for image_idx, match in enumerate(IMAGE_PLACEHOLDER_PATTERN.finditer(content)):
        x1, y1, x2, y2 = [int(group) for group in match.groups()]
        px1 = _to_pixel(x1, image_width)
        py1 = _to_pixel(y1, image_height)
        px2 = _to_pixel(x2, image_width)
        py2 = _to_pixel(y2, image_height)
        if px2 <= px1 or py2 <= py1:
            continue
        filename = f"{page_number}_{image_idx}.jpg"
        image.crop((px1, py1, px2, py2)).convert("RGB").save(os.path.join(image_output_dir, filename))
        replacements.append((match.group(0), f"![](images/{filename})"))

    for source, target in replacements:
        content = content.replace(source, target, 1)

    content = (
        content.replace("\\coloneqq", ":=")
        .replace("\\eqqcolon", "=:")
        .replace("\n\n\n\n", "\n\n")
        .replace("\n\n\n", "\n\n")
    )
    return content.strip()
