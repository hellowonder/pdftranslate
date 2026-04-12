#!/usr/bin/env python3
from __future__ import annotations

from openai import OpenAI

from ocr_client import OCRPageRequest, OCRPageResult, encode_image_to_data_url, looks_invalid_ocr_output
from ocr_deepseek_postprocess import build_deepseek_page_markdown

DEFAULT_GEMMA_OCR_PROMPT = (
    "<image>\n"
    "<|grounding|>Convert the document page to markdown. "
    "Preserve reading order, headings, lists, tables, formulas, image regions, and image captions. "
    "For figures and non-text graphics, include grounded image regions so they can be cropped from the page image."
)
DEFAULT_GEMMA_OCR_PROMPT2 = (
    "<image>\n"
    "<|grounding|>Extract this document page as markdown in reading order. "
    "Preserve headings, lists, tables, formulas, image regions, and captions. "
    "Keep grounded references for figures and diagrams."
)
DEFAULT_OCR_MAX_RETRIES = 3


class GemmaOCRClient:
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
                print(f"Gemma OCR inference error: {exc}")
                continue
        return "", False

    def infer_image(self, image) -> str:
        data_url = encode_image_to_data_url(image)
        content, suc = self._infer_with_retry(DEFAULT_GEMMA_OCR_PROMPT, data_url)
        if not suc:
            return content
        if not looks_invalid_ocr_output(content):
            return content

        print("Initial Gemma OCR output looks invalid, retrying with alternative prompt...")
        content, suc = self._infer_with_retry(DEFAULT_GEMMA_OCR_PROMPT2, data_url)
        if not suc:
            return content
        return content if not looks_invalid_ocr_output(content) else ""

    def build_markdown_from_raw(self, request: OCRPageRequest, raw_text: str) -> str:
        return build_deepseek_page_markdown(
            raw_text=raw_text,
            image=request.image,
            image_output_dir=request.image_output_dir,
            page_number=request.page_number,
        )

    def recognize_page(self, request: OCRPageRequest) -> OCRPageResult:
        raw_text = self.infer_image(request.image)
        markdown = self.build_markdown_from_raw(request, raw_text)
        return OCRPageResult(
            raw_text=raw_text,
            markdown=markdown,
        )
