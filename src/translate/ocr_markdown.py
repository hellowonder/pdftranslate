#!/usr/bin/env python3
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from PIL import Image
from tqdm import tqdm

from ocr_client import OCRClient, OCRPageRequest, init_ocr_client
from ocr_pdf_bold_styles import apply_bold_texts_to_markdown, extract_bold_text_by_page
from ocr_pdf_images import pdf_to_images_high_quality, resolve_page_numbers
from paged_markdown_io import read_page_markdown_files, write_page_markdown

PAGE_SEPARATOR = "\n<!--- Page Split --->\n"


@dataclass
class MarkdownExtractionResult:
    markdown_pages: List[str]
    page_numbers: List[int]
    images: List[Image.Image]


def write_ocr_metadata(path: str, page_numbers: List[int]) -> None:
    """
    持久化 OCR 阶段元数据，供后续阶段直接恢复上下文。
    """
    payload = {
        "page_numbers": list(page_numbers),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_ocr_metadata(path: str) -> List[int]:
    """
    读取 OCR 阶段元数据中的页码列表。
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    page_numbers = payload.get("page_numbers")
    if not isinstance(page_numbers, list) or not all(isinstance(num, int) for num in page_numbers):
        raise ValueError(f"Invalid OCR metadata file: {path}")
    return page_numbers


def load_ocr_stage_outputs(
    input_pdf: str,
    original_markdown_path: str,
    page_output_dir: str,
    ocr_metadata_path: Optional[str],
    dpi: int,
    pdf_loader: Optional[Callable[[str, int, Optional[List[int]]], List[Image.Image]]] = None,
) -> MarkdownExtractionResult:
    """
    从 OCR 阶段持久化产物恢复原文 Markdown、页码和页面图像。
    """
    page_numbers = discover_page_numbers_from_stage_dir(page_output_dir)
    if not page_numbers and ocr_metadata_path and Path(ocr_metadata_path).exists():
        page_numbers = read_ocr_metadata(ocr_metadata_path)
    with open(original_markdown_path, "r", encoding="utf-8") as f:
        content = f.read()
    markdown_pages = content.split(PAGE_SEPARATOR) if page_numbers else []
    if page_numbers and len(markdown_pages) != len(page_numbers):
        raise ValueError(
            f"Markdown page count mismatch for {original_markdown_path}: "
            f"expected {len(page_numbers)}, got {len(markdown_pages)}."
        )
    if pdf_loader is None:
        pdf_loader = pdf_to_images_high_quality
    images = pdf_loader(
        input_pdf,
        dpi=dpi,
        page_numbers=page_numbers,
    )
    return MarkdownExtractionResult(markdown_pages, page_numbers, images)


def build_raw_ocr_output_path(raw_output_dir: str, page_number: int) -> Path:
    raw_dir = Path(raw_output_dir)
    return raw_dir / f"page_{page_number:04d}.md"


def build_ocr_input_image_path(input_image_dir: str, page_number: int) -> Path:
    return Path(input_image_dir) / f"page_{page_number:04d}.png"


def save_ocr_input_image(input_image_dir: str, page_number: int, image: Image.Image) -> None:
    path = build_ocr_input_image_path(input_image_dir, page_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save the exact resized image that will be sent to OCR for debugging and reuse.
    image.save(path, format="PNG")


def raw_ocr_outputs_exist(raw_output_dir: str, page_number: int) -> bool:
    return build_raw_ocr_output_path(raw_output_dir, page_number).exists()


def all_raw_ocr_outputs_exist(raw_output_dir: str, page_numbers: Sequence[int]) -> bool:
    return all(raw_ocr_outputs_exist(raw_output_dir, page_number) for page_number in page_numbers)


def write_raw_ocr_output(raw_output_dir: str, page_number: int, content: str) -> None:
    raw_path = build_raw_ocr_output_path(raw_output_dir, page_number)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(content, encoding="utf-8")


def discover_page_numbers_from_stage_dir(stage_dir: str, suffix: str = ".md") -> List[int]:
    page_numbers: List[int] = []
    for path in sorted(Path(stage_dir).glob(f"page_*{suffix}")):
        stem = path.name
        number_part = stem[len("page_") : -len(suffix)] if suffix else stem[len("page_") :]
        if len(number_part) == 4 and number_part.isdigit():
            page_numbers.append(int(number_part))
    return page_numbers


def apply_pdf_bold_marks(
    pdf_path: str,
    page_numbers: List[int],
    pages: List[str],
) -> List[str]:
    try:
        bold_texts_by_page = extract_bold_text_by_page(pdf_path, page_numbers)
    except Exception:
        bold_texts_by_page = {}
    final_pages: List[str] = []
    for page_number, content in zip(page_numbers, pages):
        processed_content = content
        bold_texts = bold_texts_by_page.get(page_number, [])
        if bold_texts:
            processed_content = apply_bold_texts_to_markdown(processed_content, bold_texts)
        final_pages.append(processed_content)
    return final_pages


def run_ocr_pages(
    ocr_client: OCRClient,
    images: Sequence[Image.Image],
    raw_output_dir: str,
    input_image_dir: str,
    image_output_dir: str,
    page_numbers: Sequence[int],
    ocr_workers: int,
) -> List[str]:
    page_jobs = list(zip(page_numbers, images))
    if not page_jobs:
        return []

    page_results: dict[int, str] = {}

    def _handle_page(job: tuple[int, Image.Image]) -> None:
        page_number, image = job
        if raw_ocr_outputs_exist(raw_output_dir, page_number):
            raw_text = build_raw_ocr_output_path(raw_output_dir, page_number).read_text(encoding="utf-8")
            page_results[page_number] = ocr_client.build_markdown_from_raw(
                OCRPageRequest(
                    page_number=page_number,
                    image=image,
                    image_output_dir=image_output_dir,
                ),
                raw_text,
            )
            return
        if is_nearly_blank_page(image):
            write_raw_ocr_output(raw_output_dir, page_number, "")
            page_results[page_number] = ""
            return

        save_ocr_input_image(input_image_dir, page_number, image)
        result = ocr_client.recognize_page(
            OCRPageRequest(
                page_number=page_number,
                image=image,
                image_output_dir=image_output_dir,
            )
        )
        write_raw_ocr_output(raw_output_dir, page_number, result.raw_text)
        page_results[page_number] = result.markdown

    max_workers = min(max(1, ocr_workers), len(page_jobs))
    if max_workers <= 1:
        for job in page_jobs:
            _handle_page(job)
        return [page_results.get(page_number, "") for page_number in page_numbers]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(_handle_page, page_jobs), total=len(page_jobs)))
    return [page_results.get(page_number, "") for page_number in page_numbers]


def write_processed_ocr_pages(page_output_dir: str, page_numbers: Sequence[int], pages: Sequence[str]) -> None:
    for page_number, page_content in zip(page_numbers, pages):
        write_page_markdown(page_output_dir, page_number, page_content)


@dataclass
class OCRMarkdownGenerator:
    ocr_client: OCRClient
    pdf_loader: Callable[[str, int, Optional[List[int]]], List[Image.Image]] = pdf_to_images_high_quality
    ocr_workers: int = 8

    def handle_page(
        self,
        page_number: int,
        image: Image.Image,
        image_output_dir: str,
        page_output_dir: str,
        raw_output_dir: Optional[str] = None,
        input_image_dir: Optional[str] = None,
        bold_texts: Optional[List[str]] = None,
    ) -> str:
        page_markdown_path = Path(page_output_dir) / f"page_{page_number:04d}.md"
        if page_markdown_path.exists() and (not raw_output_dir or raw_ocr_outputs_exist(raw_output_dir, page_number)):
            return page_markdown_path.read_text(encoding="utf-8")

        if is_nearly_blank_page(image):
            if raw_output_dir:
                write_raw_ocr_output(raw_output_dir, page_number, "")
            page_markdown_path.parent.mkdir(parents=True, exist_ok=True)
            page_markdown_path.write_text("", encoding="utf-8")
            return ""

        if input_image_dir:
            save_ocr_input_image(input_image_dir, page_number, image)
        result = self.ocr_client.recognize_page(
            OCRPageRequest(
                page_number=page_number,
                image=image,
                image_output_dir=image_output_dir,
            )
        )
        if raw_output_dir:
            write_raw_ocr_output(raw_output_dir, page_number, result.raw_text)

        processed_content = result.markdown
        if bold_texts:
            processed_content = apply_bold_texts_to_markdown(processed_content, bold_texts)
        page_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        page_markdown_path.write_text(processed_content, encoding="utf-8")
        return processed_content

    def extract_bold_texts_by_page(self, pdf_path: str, page_numbers: List[int]) -> Dict[int, List[str]]:
        try:
            return extract_bold_text_by_page(pdf_path, page_numbers)
        except Exception:
            return {}


def is_nearly_blank_page(
    image: Image.Image,
    white_threshold: int = 250,
    non_white_ratio_threshold: float = 0.002,
) -> bool:
    """
    Detect mostly empty pages so OCR failures on blank front-matter pages do not
    abort the full translation job.
    """
    grayscale = image.convert("L")
    sampled = grayscale.resize((max(1, grayscale.width // 4), max(1, grayscale.height // 4)))
    pixels = sampled.load()
    total_pixels = sampled.width * sampled.height
    non_white_pixels = 0
    for y in range(sampled.height):
        for x in range(sampled.width):
            if pixels[x, y] < white_threshold:  # type: ignore
                non_white_pixels += 1
    return (non_white_pixels / max(1, total_pixels)) <= non_white_ratio_threshold
