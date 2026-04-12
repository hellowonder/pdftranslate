#!/usr/bin/env python3
from __future__ import annotations

import re

from openai import OpenAI
from PIL import Image

from ocr_client import OCRPageRequest, OCRPageResult, encode_image_to_data_url, looks_invalid_ocr_output
from ocr_deepseek_postprocess import build_deepseek_page_markdown

DEFAULT_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
DEFAULT_OCR_PROMPT2 = "<image>\n<|grounding|>Convert the document to markdown"
DEFAULT_OCR_MAX_RETRIES = 3
COORDINATE_PATTERN = re.compile(r"\[\[(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*)\]\]")


class DeepseekOCRClient:
    def __init__(
        self,
        client: OpenAI,
        model: str,
    ) -> None:
        self.client = client
        self.model = model

    def _infer_with_retry(self, prompt: str, data_url: str) -> tuple[str, bool]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        for _ in range(DEFAULT_OCR_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
                content = response.choices[0].message.content or ""
                return content, True
            except Exception as exc:
                print(f"OCR inference error: {exc}")
                continue
        return "", False

    def infer(self, data_url: str) -> str:
        content, suc = self._infer_with_retry(DEFAULT_OCR_PROMPT, data_url)
        if not suc:
            return content

        is_valid = not looks_invalid_ocr_output(content)
        if not is_valid:
            print("Initial OCR output looks invalid, retrying with alternative prompt...")
            content, suc = self._infer_with_retry(DEFAULT_OCR_PROMPT2, data_url)
            if not suc:
                return content
            is_valid = not looks_invalid_ocr_output(content)

        return content if is_valid else ""

    def infer_image(self, image: Image.Image) -> str:
        data_url = encode_image_to_data_url(image)
        content = self.infer(data_url=data_url)
        if content:
            return content

        print("OCR output still looks invalid, attempting to crop and re-map coordinates...")
        crop_result = crop_main_text_region(image)
        if crop_result is None:
            return ""

        crop, crop_box = crop_result
        print(f"Original image size: {image.size}, Cropped image size: {crop.size}")
        cropped_content = self.infer(data_url=encode_image_to_data_url(crop))
        if not cropped_content:
            return ""
        return remap_ocr_coordinates_to_original(
            cropped_content,
            original_size=image.size,
            crop_box=crop_box,
        )

    def recognize_page(self, request: OCRPageRequest) -> OCRPageResult:
        raw_text = self.infer_image(request.image)
        markdown = self.build_markdown_from_raw(request, raw_text)
        return OCRPageResult(
            raw_text=raw_text,
            markdown=markdown,
        )

    def build_markdown_from_raw(self, request: OCRPageRequest, raw_text: str) -> str:
        return build_deepseek_page_markdown(
            raw_text=raw_text,
            image=request.image,
            image_output_dir=request.image_output_dir,
            page_number=request.page_number,
        )


def crop_main_text_region(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
    grayscale = image.convert("L")
    width, height = grayscale.size
    pixels = grayscale.load()
    threshold = 245
    min_dark_per_row = max(2, width // 200)
    min_dark_per_col = max(2, height // 200)

    top = None
    bottom = None
    for y in range(height):
        dark = 0
        for x in range(width):
            if pixels[x, y] < threshold:  # type: ignore[index]
                dark += 1
        if dark >= min_dark_per_row:
            top = y
            break
    for y in range(height - 1, -1, -1):
        dark = 0
        for x in range(width):
            if pixels[x, y] < threshold:  # type: ignore[index]
                dark += 1
        if dark >= min_dark_per_row:
            bottom = y
            break

    left = None
    right = None
    for x in range(width):
        dark = 0
        for y in range(height):
            if pixels[x, y] < threshold:  # type: ignore[index]
                dark += 1
        if dark >= min_dark_per_col:
            left = x
            break
    for x in range(width - 1, -1, -1):
        dark = 0
        for y in range(height):
            if pixels[x, y] < threshold:  # type: ignore[index]
                dark += 1
        if dark >= min_dark_per_col:
            right = x
            break

    if None in (top, bottom, left, right):
        return None
    if bottom < top or right < left:
        return None

    pad_x = max(24, width // 40)
    pad_y = max(24, height // 40)
    crop_box = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )
    cropped = image.crop(crop_box)
    if cropped.size == image.size:
        return None
    return cropped, crop_box


def remap_ocr_coordinates_to_original(
    content: str,
    original_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> str:
    if not content or "[[" not in content:
        return content

    original_width, original_height = original_size
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    crop_width = max(1, crop_right - crop_left)
    crop_height = max(1, crop_bottom - crop_top)

    def _replace(match: re.Match[str]) -> str:
        try:
            x1, y1, x2, y2 = [int(part.strip()) for part in match.group(1).split(",")]
        except ValueError:
            return match.group(0)

        mapped = (
            _map_coord(x1, crop_left, crop_width, original_width),
            _map_coord(y1, crop_top, crop_height, original_height),
            _map_coord(x2, crop_left, crop_width, original_width),
            _map_coord(y2, crop_top, crop_height, original_height),
        )
        return f"[[{mapped[0]}, {mapped[1]}, {mapped[2]}, {mapped[3]}]]"

    return COORDINATE_PATTERN.sub(_replace, content)


def _map_coord(coord: int, crop_start: int, crop_span: int, original_span: int) -> int:
    clamped = min(999, max(0, coord))
    absolute = crop_start + (clamped / 999.0) * crop_span
    mapped = round((absolute / max(1, original_span)) * 999)
    return min(999, max(0, mapped))
