#!/usr/bin/env python3
import base64
import io

from openai import OpenAI
from PIL import Image
from llm_util import has_low_diversity_or_repetition

DEFAULT_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
DEFAULT_LAYOUT_PROMPT = "<image>\n<|grounding|>Given the layout of the image."
DEFAULT_OCR_IMAGE_MAX_SIDE = 640
DEFAULT_OCR_MAX_RETRIES = 3
OCR_FAILURE_PREFIX = "【OCR "


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

    def infer_with_prompt(
        self,
        data_url: str,
        prompt: str,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            }
        ]
        last_error: Exception | None = None
        content = ""
        for _ in range(DEFAULT_OCR_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
                content = response.choices[0].message.content or ""
                if looks_invalid_ocr_output(content):
                    content = "【OCR output looks invalid, possibly due to an incompatible prompt or unsupported multimodal endpoint.】"
                    continue
                return content
            except Exception as exc:
                content = f"【OCR inference error: {exc}】"
        return content

    def infer(
        self,
        data_url: str,
    ) -> str:
        return self.infer_with_prompt(
            data_url=data_url,
            prompt=DEFAULT_OCR_PROMPT,
        )

    def infer_image(
        self,
        image: Image.Image,
    ) -> str:
        content = self.infer(data_url=encode_image_data_url_for_ocr(image))
        if not self._should_retry_with_body_crop(content):
            return content

        cropped = crop_main_text_region(image)
        if cropped is None:
            return content

        cropped_content = self.infer(data_url=encode_image_data_url_for_ocr(cropped))
        if self._should_retry_with_body_crop(cropped_content):
            return content
        return cropped_content

    def _should_retry_with_body_crop(self, content: str) -> bool:
        stripped = (content or "").strip()
        if not stripped:
            return True
        if stripped.startswith(OCR_FAILURE_PREFIX):
            return True
        if looks_invalid_ocr_output(stripped):
            return True
        return False


def crop_main_text_region(image: Image.Image) -> Image.Image | None:
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
    if bottom <= top or right <= left:
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
    return cropped
