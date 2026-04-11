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
        for _ in range(DEFAULT_OCR_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
                content = response.choices[0].message.content or ""
                if looks_invalid_ocr_output(content):
                    raise ValueError(f"OCR output looks invalid: {content!r}")
                return content
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def infer(
        self,
        data_url: str,
    ) -> str:
        return self.infer_with_prompt(
            data_url=data_url,
            prompt=DEFAULT_OCR_PROMPT,
        )
