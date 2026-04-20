#!/usr/bin/env python3
from __future__ import annotations

from openai import OpenAI

from ocr_client import OCRPageRequest, OCRPageResult, encode_image_to_data_url, looks_invalid_ocr_output
from ocr_gemma_postprocess import build_gemma_page_markdown

DEFAULT_MMLLM_OCR_PROMPT = '''Convert this book page image into clean, structured markdown.

Requirements:

1. Preserve the natural reading order of the page. 
   - If the page is multi-column, read column by column from left to right.

2. Preserve semantic structure:
   - headings, paragraphs, bold, italic, underline, strikethrough
   - lists (ordered and unordered)
   - tables
   - formulas

3. Use standard markdown:
   - headings with #, ##, etc.
   - lists with -, *, or numbers
   - tables using markdown table syntax
   - formulas as LaTeX when possible

4. Figures and graphics:
   - For every figure, diagram, chart, or non-text graphic, output a markdown image placeholder in this exact format:
     ![](image_X1_Y1_X2_Y2.png)
   - X1, Y1, X2, Y2 must be integers in a normalized coordinate system from 0 to 1000:
     (X1, Y1) = top-left corner
     (X2, Y2) = bottom-right corner
   - Coordinates must satisfy:
     0 <= X1 < X2 <= 1000
     0 <= Y1 < Y2 <= 1000
   - Do not output pixel coordinates
   - Do not invent filenames beyond this exact format

5. Captions:
   - Keep figure captions as normal markdown text placed immediately before or after the corresponding image placeholder

6. Do NOT convert the following into images:
   - normal text
   - tables that can be represented in markdown
   - formulas that can be written in LaTeX

7. Ignore non-content regions:
   - page numbers (top or bottom)
   - running headers and footers
   - margins, borders, and decorative elements
   - scanning artifacts or shadows

8. Layout handling:
   - Do not attempt pixel-perfect layout reproduction
   - Focus on correct reading order and logical grouping

9. Output constraints:
   - Output ONLY the final markdown
   - Do not include explanations, comments, or extra text'''

DEFAULT_OCR_MAX_RETRIES = 1
DEFAULT_OCR_REASONING_EFFORT = "none"


class MMLLMOcrClient:
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
                    extra_body={"reasoning": {"effort": DEFAULT_OCR_REASONING_EFFORT}},
                )
                content = response.choices[0].message.content or ""
                return content, True
            except Exception as exc:
                print(f"MMLLM OCR inference error: {exc}")
                continue
        return "", False

    def infer_image(self, image) -> str:
        data_url = encode_image_to_data_url(image)
        content, suc = self._infer_with_retry(DEFAULT_MMLLM_OCR_PROMPT, data_url)
        if not suc:
            return content
        if not looks_invalid_ocr_output(content):
            return content

        return content if not looks_invalid_ocr_output(content) else ""

    def build_markdown_from_raw(self, request: OCRPageRequest, raw_text: str) -> str:
        return build_gemma_page_markdown(
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
