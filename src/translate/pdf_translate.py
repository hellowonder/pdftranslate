#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from typing import List, Optional, Set

from pypdf import PdfReader

from document_io import normalize_document_input
from ocr_stage import run_ocr_stage
from translate_service import add_translation_arguments, validate_translation_args
from translate_stage import run_translate_stage
from render_weasyprint import DEFAULT_KATEX_CSS_PATH
from render_stage import run_render_stage

STAGE_SEQUENCE = ("ocr", "translate", "render")


def build_output_paths(input_pdf: str, output_dir: str) -> dict[str, str]:
    base_name = os.path.splitext(os.path.basename(input_pdf))[0]
    return {
        "input_pdf": os.path.abspath(input_pdf),
        "output_dir": os.path.abspath(output_dir),
        "ocr_dir": os.path.join(output_dir, "ocr"),
        "ocr_raw_dir": os.path.join(output_dir, "ocr", "ocr_raw"),
        "ocr_input_images_dir": os.path.join(output_dir, "ocr", "ocr_input_images"),
        "ocr_images_dir": os.path.join(output_dir, "ocr", "images"),
        "ocr_metadata": os.path.join(output_dir, "ocr", f"{base_name}_ocr.json"),
        "ocr_page_merge": os.path.join(output_dir, "ocr", f"{base_name}_page_merge.json"),
        "translate_dir": os.path.join(output_dir, "translate"),
        "translate_images_dir": os.path.join(output_dir, "translate", "images"),
        "translated_markdown": os.path.join(output_dir, "translate", f"{base_name}_cn.md"),
        "render_dir": os.path.join(output_dir, "render"),
        "original_pdf": os.path.join(output_dir, f"{base_name}_original.pdf"),
        "translated_pdf": os.path.join(output_dir, f"{base_name}_cn.pdf"),
        "interleaved_pdf": os.path.join(output_dir, f"{base_name}_interleaved.pdf"),
    }

def resolve_page_numbers(pages_arg: Optional[str], total_pages: int) -> List[int]:
    """
    解析命令行中的页码参数，返回合法且升序的 1 基页码列表。

    参数:
        pages_arg: 逗号分隔的页码或区间，例如 ``1,3-5``；为空时表示全部页面。
        total_pages: PDF 总页数，用于过滤越界页码。
    返回:
        List[int]: 过滤、去重并排序后的合法页码列表。
    """
    if not pages_arg:
        return list(range(1, total_pages + 1))
    selected: Set[int] = set()
    for part in pages_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                raise ValueError(f"Invalid page range: '{part}'")
            if start > end:
                start, end = end, start
            for num in range(start, end + 1):
                selected.add(num)
        else:
            try:
                selected.add(int(part))
            except ValueError:
                raise ValueError(f"Invalid page number: '{part}'")
    filtered = sorted(num for num in selected if 1 <= num <= total_pages)
    if not filtered:
        raise ValueError("No valid page numbers selected after filtering.")
    return filtered

def resolve_target_page_numbers(args: argparse.Namespace, pdf_reader: PdfReader) -> List[int]:
    total_pages = len(pdf_reader.pages)
    return resolve_page_numbers(getattr(args, "pages", None), total_pages)


def parse_stage_selection(stage_value: str | None) -> List[str]:
    raw = (stage_value or "").strip()
    if not raw:
        raise ValueError("At least one stage must be specified for --stages")
    requested: List[str] = []
    for part in raw.split(","):
        stage = part.strip().lower()
        if not stage:
            continue
        if stage not in STAGE_SEQUENCE:
            raise ValueError(f"Unknown stage '{stage}'. Supported stages: {', '.join(STAGE_SEQUENCE)}")
        if stage not in requested:
            requested.append(stage)
    if not requested:
        raise ValueError("At least one stage must be specified for --stages")
    return [stage for stage in STAGE_SEQUENCE if stage in requested]


def parse_args():
    parser = argparse.ArgumentParser(
        description="OCR PDF/DjVu to Markdown, translate via external OpenAI-compatible APIs, and build interleaved PDF."
    )
    parser.add_argument("--input", required=True, help="Input PDF or DjVu file.")
    parser.add_argument("--output-dir", required=True, help="Directory to store OCR, translation, and render outputs.")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the output directory before running.",
    )
    parser.add_argument("--pages", help="Comma-separated 1-based page numbers or ranges to process, e.g. 1,3,5-7.")
    parser.add_argument(
        "--vllm-sleep",
        dest="vllm_sleep",
        action="store_true",
        default=True,
        help="Call wake_up/sleep control endpoints around OCR and translation model requests. This is for vllm to release GPU memory between stages and can be helpful for large models on limited hardware.",
    )
    parser.add_argument(
        "--no-vllm-sleep",
        dest="vllm_sleep",
        action="store_false",
        help="Disable wake_up/sleep control endpoint calls around OCR and translation.",
    )
    parser.add_argument(
        "--ocr-base-url",
        # default="http://localhost:11434/v1",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible OCR base URL, for example an Ollama endpoint.",
    )
    parser.add_argument(
        "--ocr-api-key",
        default=os.environ.get("OCR_API_KEY") or os.environ.get("OLLAMA_API_KEY", "ollama"),
        help="API key for the OCR endpoint.",
    )
    parser.add_argument(
        "--ocr-model",
        # default="deepseek-ocr:3b",
        default="deepseek-ai/DeepSeek-OCR",
        help="OCR model name exposed by the OCR endpoint.",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=96,
        help="DPI used when rasterizing document pages for OCR.",
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=8,
        help="Parallel workers for OCR page requests.",
    )
    parser.add_argument("--font-path", help="TrueType font path for rendering text (should support Chinese).")
    parser.add_argument(
        "--font-size",
        default="9.5pt",
        help=(
            "Final body font size for rendered pages. Accepts CSS/LaTeX lengths like 10.5pt, 4mm, 0.15in; "
            "plain numbers are treated as pt."
        ),
    )
    parser.add_argument("--page-width", type=int, default=1240, help="Fallback translation page width (px) if original missing.")
    parser.add_argument("--min-page-height", type=int, default=1754, help="Fallback translation page height (px) if original missing.")
    parser.add_argument(
        "--margin",
        help=(
            "Page margin for translation pages. Accepts CSS/LaTeX lengths like 0.5in, 12mm, 24px; "
            "plain numbers are treated as px. If omitted, margin is derived from page size."
        ),
    )
    parser.add_argument("--image-spacing", type=int, default=16, help="Spacing after inlined images, in pixels.")
    parser.add_argument("--layout-workers", type=int, default=8, help="Parallel workers for PDF layout rendering.")
    parser.add_argument(
        "--generate-interleave-pdf",
        dest="generate_interleave_pdf",
        action="store_true",
        default=True,
        help="Generate the interleaved PDF that alternates translated pages with original pages.",
    )
    parser.add_argument(
        "--no-generate-interleave-pdf",
        dest="generate_interleave_pdf",
        action="store_false",
        help="Disable generation of the interleaved PDF.",
    )
    parser.add_argument(
        "--generate-translation-only-pdf",
        dest="generate_translation_only_pdf",
        action="store_true",
        default=False,
        help="Generate the translation-only PDF.",
    )
    parser.add_argument(
        "--no-generate-translation-only-pdf",
        dest="generate_translation_only_pdf",
        action="store_false",
        help="Disable generation of the translation-only PDF.",
    )
    parser.add_argument("--no-translation", action="store_true", help="whether skip translation")
    parser.add_argument(
        "--skip-first-page-translation",
        action="store_true",
        help="Skip translating page 1 of the PDF.",
    )
    add_translation_arguments(parser)
    parser.add_argument(
        "--katex-css-path",
        help=f"Path to a local katex.min.css for math rendering (defaults to {DEFAULT_KATEX_CSS_PATH})."
    )
    parser.add_argument(
        "--stages",
        default="ocr,translate,render",
        help="Comma-separated stages to run (choices: ocr, translate, render).",
    )
    args = parser.parse_args()
    try:
        args.selected_stages = parse_stage_selection(args.stages)
        validate_translation_args(args)
        if "render" in args.selected_stages and not (
            args.generate_interleave_pdf or args.generate_translation_only_pdf
        ):
            parser.error(
                "At least one render output must be enabled: "
                "--generate-interleave-pdf or --generate-translation-only-pdf."
            )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def prepare_output_paths(input_pdf: str, output_dir: str, clear: bool = False) -> dict[str, str]:
    input_pdf = os.path.abspath(input_pdf)
    output_dir = os.path.abspath(output_dir)
    if clear and os.path.isdir(output_dir):
        for entry in os.listdir(output_dir):
            entry_path = os.path.join(output_dir, entry)
            if os.path.isdir(entry_path) and not os.path.islink(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.unlink(entry_path)
    os.makedirs(output_dir, exist_ok=True)
    output_paths = build_output_paths(input_pdf, output_dir)
    os.makedirs(output_paths["ocr_dir"], exist_ok=True)
    os.makedirs(output_paths["ocr_raw_dir"], exist_ok=True)
    os.makedirs(output_paths["ocr_input_images_dir"], exist_ok=True)
    os.makedirs(output_paths["ocr_images_dir"], exist_ok=True)
    os.makedirs(output_paths["translate_dir"], exist_ok=True)
    os.makedirs(output_paths["translate_images_dir"], exist_ok=True)
    os.makedirs(output_paths["render_dir"], exist_ok=True)
    output_paths["input_pdf"] = normalize_document_input(input_pdf, output_paths["render_dir"])
    return output_paths

def main():
    args = parse_args()
    output_paths = prepare_output_paths(args.input, args.output_dir, clear=args.clear)

    pdf_reader = PdfReader(output_paths["input_pdf"])
    page_numbers = resolve_target_page_numbers(args, pdf_reader)

    selected_set = set(args.selected_stages)
    if "ocr" in selected_set:
        run_ocr_stage(args, output_paths, page_numbers)
    if "translate" in selected_set:
        run_translate_stage(args, output_paths, page_numbers)
    if "render" in selected_set:
        run_render_stage(args, pdf_reader, output_paths, page_numbers)

if __name__ == "__main__":
    main()
