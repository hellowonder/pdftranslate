#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlsplit

from PIL import Image
from pypdf import PdfReader
import requests

from ocr_markdown import (
    all_raw_ocr_outputs_exist,
    apply_pdf_bold_marks,
    init_ocr_client,
    run_ocr_pages,
    write_processed_ocr_pages,
)
from ocr_pdf_images import pdf_to_images_high_quality
from paged_markdown_io import page_markdown_outputs_exist, read_page_markdown_files, write_paged_markdown_document
from translate_page_merge import BoundaryDecision, merge_cross_page_paragraphs
from translate_service import init_translation_service

def write_page_merge_metadata(path: str, decisions: List[BoundaryDecision]) -> None:
    payload = [
        {
            "page_number_left": item.page_number_left,
            "page_number_right": item.page_number_right,
            "decision": item.decision,
            "left_block": item.left_block,
            "right_block": item.right_block,
        }
        for item in decisions
    ]
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def page_merge_outputs_exist(output_paths: dict[str, str], page_numbers: List[int]) -> bool:
    return (
        Path(output_paths["ocr_page_merge"]).exists()
        and page_markdown_outputs_exist(output_paths["ocr_dir"], page_numbers)
    )

def should_manage_vllm_ocr_lifecycle(
    args: argparse.Namespace,
    output_paths: dict[str, str],
    page_numbers: List[int],
) -> bool:
    return bool(getattr(args, "vllm_sleep", False))


def build_vllm_control_url(ocr_base_url: str, path: str) -> str:
    parsed = urlsplit((ocr_base_url or "").strip())
    if parsed.scheme and parsed.netloc:
        base = f"{parsed.scheme}://{parsed.netloc}"
    else:
        trimmed = (ocr_base_url or "").strip().rstrip("/")
        if trimmed.endswith("/v1"):
            base = trimmed[:-3].rstrip("/")
        else:
            base = trimmed
    return f"{base}/{path.lstrip('/')}"


def notify_vllm_control_endpoint(ocr_base_url: str, path: str) -> None:
    try:
        control_url = build_vllm_control_url(ocr_base_url, path)
        print(f"Calling vLLM control endpoint: {control_url}", file=sys.stderr)
        response = requests.post(control_url, timeout=60)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to notify vLLM control endpoint at {ocr_base_url} with path {path}: {e}", file=sys.stderr)

def run_markdown_merge_page_break_step(
    args: argparse.Namespace,
    output_paths: dict[str, str],
    page_numbers: List[int],
    processed_pages: List[str],
) -> List[str]:
    if page_merge_outputs_exist(output_paths, page_numbers):
        return read_page_markdown_files(output_paths["ocr_dir"], page_numbers)

    print("Resolving page breaks across page boundaries...", file=sys.stderr)
    markdown_pages, decisions = merge_cross_page_paragraphs(
        list(processed_pages),
        page_numbers=page_numbers,
    )
    write_processed_ocr_pages(output_paths["ocr_dir"], page_numbers, markdown_pages)
    write_page_merge_metadata(output_paths["ocr_page_merge"], decisions)
    print("Finished resolving page breaks across page boundaries.", file=sys.stderr)
    return markdown_pages


def run_ocr_stage(args: argparse.Namespace, output_paths: dict[str, str], page_numbers: List[int]) -> None:
    manage_vllm_lifecycle = should_manage_vllm_ocr_lifecycle(args, output_paths, page_numbers)
    images = None
    processed_pages: List[str] | None = None
    if not all_raw_ocr_outputs_exist(output_paths["ocr_raw_dir"], page_numbers):
        # Rasterize the target PDF pages into in-memory page images used by OCR and later OCR postprocess.
        # Output: no file written here; returns page images for downstream steps.
        images = pdf_to_images_high_quality(
            output_paths["input_pdf"],
            dpi=args.ocr_dpi,
            page_numbers=page_numbers,
        )

        # Run expensive OCR inference only for pages whose raw outputs are missing.
        # Output: ocr/ocr_raw/page_XXXX.md
        if manage_vllm_lifecycle:
            notify_vllm_control_endpoint(args.ocr_base_url, "wake_up")
        try:
            processed_pages = run_ocr_pages(
                ocr_client=init_ocr_client(args),
                images=images,
                raw_output_dir=output_paths["ocr_raw_dir"],
                input_image_dir=output_paths["ocr_input_images_dir"],
                image_output_dir=output_paths["ocr_images_dir"],
                page_numbers=page_numbers,
                ocr_workers=args.ocr_workers,
            )
        finally:
            if manage_vllm_lifecycle:
                notify_vllm_control_endpoint(args.ocr_base_url, "sleep?level=1")
    else:
        if manage_vllm_lifecycle:
            notify_vllm_control_endpoint(args.ocr_base_url, "sleep?level=1")

    
    # Post OCR processing to clean up markdown and resolve page breaks, producing the final per-page markdown
    # consumed by translate/render stages.
    if not page_merge_outputs_exist(output_paths, page_numbers):
        if images is None:
            images = pdf_to_images_high_quality(
                output_paths["input_pdf"],
                dpi=args.ocr_dpi,
                page_numbers=page_numbers,
            )
        # Convert raw OCR outputs into page markdown, including cropped images under ocr/images/.
        if processed_pages is None:
            processed_pages = run_ocr_pages(
                ocr_client=init_ocr_client(args),
                images=images,
                raw_output_dir=output_paths["ocr_raw_dir"],
                input_image_dir=output_paths["ocr_input_images_dir"],
                image_output_dir=output_paths["ocr_images_dir"],
                page_numbers=page_numbers,
                ocr_workers=args.ocr_workers,
            )
        processed_pages = apply_pdf_bold_marks(output_paths["input_pdf"], page_numbers, processed_pages)

        # Resolve paragraph continuation across page breaks and persist the final page-level OCR markdown.
        # Output: ocr/page_XXXX.md and ocr/<base>_page_merge.json
        _ = run_markdown_merge_page_break_step(args, output_paths, page_numbers, processed_pages)

    # Concatenate final page markdown into the stage-level OCR document consumed by translate/render stages.
    # Output: ocr/<base>_original.md
    print(f"Extracted {len(page_numbers)} pages...", file=sys.stderr)
    return
