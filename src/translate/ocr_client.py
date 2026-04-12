#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image

from llm_util import has_low_diversity_or_repetition
from translate_service import configure_openai

DEFAULT_OCR_IMAGE_MAX_SIDE = 640
REPEATED_NUMBERING_PATTERN = re.compile(r"(?:^|\s)(?:\d+\.\s*){40,}")


def looks_invalid_ocr_output(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return True
    if len(stripped) <= 2:
        return True
    alpha_numeric_count = sum(ch.isalnum() for ch in stripped)
    if alpha_numeric_count == 0:
        return True
    if REPEATED_NUMBERING_PATTERN.search(stripped):
        return True
    if has_low_diversity_or_repetition(stripped):
        return True
    return False


def resize_image_for_ocr(image: Image.Image, max_side: int = DEFAULT_OCR_IMAGE_MAX_SIDE) -> Image.Image:
    if max_side <= 0:
        return image.copy()
    resized = image.convert("RGB")
    if max(resized.size) <= max_side:
        return resized
    resized.thumbnail((max_side, max_side))
    return resized


def encode_image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def encode_image_data_url_for_ocr(image: Image.Image) -> str:
    return encode_image_to_data_url(image)


@dataclass
class OCRPageRequest:
    page_number: int
    image: Image.Image
    image_output_dir: str
    pdf_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRPageResult:
    raw_text: str
    markdown: str


class OCRClient(Protocol):
    def recognize_page(self, request: OCRPageRequest) -> OCRPageResult:
        ...

    def build_markdown_from_raw(self, request: OCRPageRequest, raw_text: str) -> str:
        ...


def init_ocr_client(args: argparse.Namespace) -> OCRClient:
    client = configure_openai(args.ocr_base_url, args.ocr_api_key)
    from ocr_gemma_client import GemmaOCRClient

    return GemmaOCRClient(
        client=client,
        model=args.ocr_model,
    )
