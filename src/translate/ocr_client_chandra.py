#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from PIL import Image

from ocr_client import OCRPageRequest, OCRPageResult, looks_invalid_ocr_output


def _ensure_chandra_import_path() -> None:
    site_packages = (
        Path(__file__).resolve().parents[2]
        / ".chandra-env"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if site_packages.exists():
        site_packages_str = str(site_packages)
        if site_packages_str not in sys.path:
            sys.path.insert(0, site_packages_str)


_ensure_chandra_import_path()

from chandra.model.schema import BatchInputItem  # type: ignore  # noqa: E402
from chandra.model.util import detect_repeat_token, scale_to_fit  # type: ignore  # noqa: E402
from chandra.output import extract_images, parse_chunks, parse_markdown  # type: ignore  # noqa: E402
from chandra.prompts import PROMPT_MAPPING  # type: ignore  # noqa: E402


DEFAULT_CHANDRA_PROMPT_TYPE = "ocr_layout"
DEFAULT_CHANDRA_MAX_RETRIES = 2
DEFAULT_CHANDRA_MAX_FAILURE_RETRIES = 2
DEFAULT_CHANDRA_MAX_OUTPUT_TOKENS = 8192
DEFAULT_CHANDRA_TEMPERATURE = 0.0
DEFAULT_CHANDRA_TOP_P = 0.1


class ChandraOCRClient:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        *,
        prompt_type: str = DEFAULT_CHANDRA_PROMPT_TYPE,
        max_output_tokens: int = DEFAULT_CHANDRA_MAX_OUTPUT_TOKENS,
        max_retries: int = DEFAULT_CHANDRA_MAX_RETRIES,
        max_failure_retries: int = DEFAULT_CHANDRA_MAX_FAILURE_RETRIES,
        temperature: float = DEFAULT_CHANDRA_TEMPERATURE,
        top_p: float = DEFAULT_CHANDRA_TOP_P,
        bbox_scale: int = 1000,
    ) -> None:
        self.client = client
        self.model = model
        self.prompt_type = prompt_type
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.max_failure_retries = max_failure_retries
        self.temperature = temperature
        self.top_p = top_p
        self.bbox_scale = bbox_scale

    def _generate_once(self, item: BatchInputItem, *, temperature: float, top_p: float) -> tuple[str, bool]:
        prompt = item.prompt or PROMPT_MAPPING[item.prompt_type or DEFAULT_CHANDRA_PROMPT_TYPE]
        image = scale_to_fit(item.image)
        from ocr_client import encode_image_to_data_url

        content = [
            {
                "type": "image_url",
                "image_url": {"url": encode_image_to_data_url(image)},
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_output_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return response.choices[0].message.content or "", False
        except Exception as exc:
            print(f"Chandra OCR inference error: {exc}")
            return "", True

    def _should_retry(self, raw_text: str, error: bool, retries: int) -> bool:
        has_repeat = detect_repeat_token(raw_text) or (
            len(raw_text) > 50 and detect_repeat_token(raw_text, cut_from_end=50)
        )
        if retries < self.max_retries and has_repeat:
            print(f"Detected repeat token, retrying Chandra OCR (attempt {retries + 1})...")
            return True
        if retries < self.max_retries and error:
            print(f"Detected Chandra OCR error, retrying (attempt {retries + 1})...")
            time.sleep(2 * (retries + 1))
            return True
        if error and retries < self.max_failure_retries:
            print(f"Detected Chandra OCR error, retrying (attempt {retries + 1})...")
            time.sleep(2 * (retries + 1))
            return True
        return False

    def infer_image(self, image: Image.Image) -> str:
        item = BatchInputItem(image=image, prompt_type=self.prompt_type)
        raw_text, error = self._generate_once(
            item,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        retries = 0
        while self._should_retry(raw_text, error, retries):
            retries += 1
            retry_temperature = min(self.temperature + 0.2 * retries, 0.8)
            raw_text, error = self._generate_once(
                item,
                temperature=retry_temperature,
                top_p=0.95,
            )
        return raw_text

    def _save_extracted_images(self, request: OCRPageRequest, raw_text: str) -> None:
        chunks = parse_chunks(raw_text, request.image, bbox_scale=self.bbox_scale)
        images = extract_images(raw_text, chunks, request.image)
        if not images:
            return
        output_dir = Path(request.image_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_name, image in images.items():
            output_path = output_dir / image_name
            image.save(output_path)

    def build_markdown_from_raw(self, request: OCRPageRequest, raw_text: str) -> str:
        if not raw_text.strip():
            return ""
        try:
            markdown = parse_markdown(
                raw_text,
                include_headers_footers=False,
                include_images=True,
            )
            self._save_extracted_images(request, raw_text)
            return markdown
        except Exception as exc:
            print(f"Chandra OCR markdown conversion error: {exc}")
            return ""

    def recognize_page(self, request: OCRPageRequest) -> OCRPageResult:
        raw_text = self.infer_image(request.image)
        markdown = self.build_markdown_from_raw(request, raw_text)
        if looks_invalid_ocr_output(markdown):
            markdown = ""
        return OCRPageResult(
            raw_text=raw_text,
            markdown=markdown,
        )
