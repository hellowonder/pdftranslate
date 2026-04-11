#!/usr/bin/env python3
import base64
import io
import re

from openai import OpenAI
from PIL import Image
from llm_util import has_low_diversity_or_repetition

DEFAULT_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
DEFAULT_LAYOUT_PROMPT = "<image>\n<|grounding|>Given the layout of the image."
DEFAULT_OCR_IMAGE_MAX_SIDE = 640
DEFAULT_OCR_MAX_RETRIES = 3
OCR_FAILURE_PREFIX = "【OCR "
COORDINATE_PATTERN = re.compile(r"\[\[(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*)\]\]")


def looks_invalid_ocr_output(text: str) -> bool:
    """
    Detect obviously broken OCR outputs returned by incompatible prompts or
    partially supported multimodal endpoints.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    if len(stripped) <= 2:
        return True
    alpha_numeric_count = sum(ch.isalnum() for ch in stripped)
    if alpha_numeric_count == 0:
        return True
    if has_low_diversity_or_repetition(stripped):
        return True
    return False


def resize_image_for_ocr(image: Image.Image, max_side: int = DEFAULT_OCR_IMAGE_MAX_SIDE) -> Image.Image:
    """
    Resize an image to the OCR model's preferred working size while preserving aspect ratio.
    """
    if max_side <= 0:
        return image.copy()
    resized = image.convert("RGB")
    if max(resized.size) <= max_side:
        return resized
    resized.thumbnail((max_side, max_side))
    return resized


def encode_image_to_data_url(image: Image.Image) -> str:
    """
    将 PIL 图像编码成 data URL，以便通过 OpenAI 图像消息传输。
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def encode_image_data_url_for_ocr(
    image: Image.Image,
    max_side: int = DEFAULT_OCR_IMAGE_MAX_SIDE,
) -> str:
    """
    Resize the image for OCR and encode it as a PNG data URL.
    """
    return encode_image_to_data_url(resize_image_for_ocr(image, max_side=max_side))


class DeepseekOCRClient:
    """
    使用 OpenAI 兼容 API 执行 OCR 推理。
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
    ) -> None:
        self.client = client
        self.model = model

    def _infer_with_retry(self, image: Image.Image, prompt: str) -> str:
        data_url = encode_image_data_url_for_ocr(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            }
        ]
        content = ""
        for _ in range(DEFAULT_OCR_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
                content = response.choices[0].message.content or ""
            except Exception as exc:
                print(f"OCR inference error: {exc}")
                continue
        return content
    
    def infer_with_prompt(
        self,
        image: Image.Image,
        prompt: str,
    ) -> str:
        content = self._infer_with_retry(image, prompt)
        
        if looks_invalid_ocr_output(content):
            crop_result = crop_main_text_region(image)
            if crop_result is not None:
                crop, crop_box = crop_result
                cropped_content = self._infer_with_retry(crop, prompt)
                return remap_ocr_coordinates_to_original(
                    cropped_content,
                    original_size=image.size,
                    crop_box=crop_box,
                )

        return content

    def infer_image(
        self,
        image: Image.Image,
    ) -> str:
        return self.infer_with_prompt(image, DEFAULT_OCR_PROMPT)

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
