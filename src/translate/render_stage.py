#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from pypdf import PdfReader

from document_io import is_pdf_input
from ocr_pdf_images import pdf_to_images_high_quality
from render_runner import PDFRenderRunner
from render_weasyprint import RenderSettings, load_katex_css, resolve_font_path
from translate_stage import load_translate_stage_outputs


def render_stage_outputs_exist(args, output_paths: dict[str, str]) -> bool:
    required_paths = [output_paths["translated_pdf"]]
    if args.generate_interleave_pdf:
        required_paths.append(output_paths["interleaved_pdf"])
    return all(Path(path).exists() for path in required_paths)


def collect_original_pdf_metadata(
    pdf_reader: PdfReader,
    page_numbers: Sequence[int],
) -> Tuple[List[Optional[object]], List[Tuple[float, float]]]:
    selected_pdf_pages: List[Optional[object]] = []
    selected_page_sizes: List[Tuple[float, float]] = []
    try:
        total_original_pages = len(pdf_reader.pages)
        for page_number in page_numbers:
            if 1 <= page_number <= total_original_pages:
                page = pdf_reader.pages[page_number - 1]
                selected_pdf_pages.append(page)
                box = page.cropbox if page.cropbox is not None else page.mediabox
                width_pt = float(box.width)
                height_pt = float(box.height)
                rotation = int(page.get("/Rotate", 0) or 0) % 360
                if rotation in (90, 270):
                    width_pt, height_pt = height_pt, width_pt
                pt_to_px = 96.0 / 72.0
                selected_page_sizes.append((width_pt * pt_to_px, height_pt * pt_to_px))
            else:
                selected_pdf_pages.append(None)
                selected_page_sizes.append((0.0, 0.0))
    except Exception:
        selected_pdf_pages = [None] * len(page_numbers)
        selected_page_sizes = [(0.0, 0.0)] * len(page_numbers)
    return selected_pdf_pages, selected_page_sizes


def render_translation_only_pdf(
    args,
    output_pdf_path: str,
    markdown_pages: Sequence[str],
    selected_page_sizes: Sequence[Tuple[float, float]],
    image_root: str,
) -> None:
    settings = RenderSettings(
        font_path=resolve_font_path(args.font_path),
        font_size=args.font_size,
        page_width=args.page_width,
        page_height=args.min_page_height,
        margin=args.margin,
        image_root=image_root,
        image_spacing=args.image_spacing,
        katex_css=load_katex_css(args.katex_css_path),
    )
    full_pdf_writer = PDFRenderRunner(
        settings=settings,
        original_page_sizes=selected_page_sizes,
    ).build_translation_only(markdown_pages)
    with open(output_pdf_path, "wb") as f:
        full_pdf_writer.write(f)


def render_interleaved_pdf(
    args,
    output_paths: dict[str, str],
    translated_pages: Sequence[str],
    selected_pdf_pages: Sequence[Optional[object]],
    selected_page_sizes: Sequence[Tuple[float, float]],
) -> None:
    settings = RenderSettings(
        font_path=resolve_font_path(args.font_path),
        font_size=args.font_size,
        page_width=args.page_width,
        page_height=args.min_page_height,
        margin=args.margin,
        image_root=output_paths["translate_images_dir"],
        image_spacing=args.image_spacing,
        katex_css=load_katex_css(args.katex_css_path),
    )
    pdf_writer = PDFRenderRunner(
        settings=settings,
        max_workers=max(1, args.layout_workers),
        original_pdf_pages=selected_pdf_pages,
        original_page_sizes=selected_page_sizes,
    ).build_interleaved(translated_pages)

    with open(output_paths["interleaved_pdf"], "wb") as f:
        pdf_writer.write(f)


def load_render_page_images_step(
    args,
    output_paths: dict[str, str],
    page_numbers: Sequence[int],
) -> Sequence[object]:
    if is_pdf_input(output_paths["input_pdf"]):
        return []
    needs_page_images = (
        not Path(output_paths["translated_pdf"]).exists()
        or (args.generate_interleave_pdf and not Path(output_paths["interleaved_pdf"]).exists())
    )
    if not needs_page_images:
        return []
    return pdf_to_images_high_quality(
        output_paths["input_pdf"],
        dpi=args.ocr_dpi,
        page_numbers=list(page_numbers),
    )

def run_render_stage(
    args,
    pdf_reader: PdfReader,
    output_paths: dict[str, str],
    page_numbers: Sequence[int],
) -> None:
    if render_stage_outputs_exist(args, output_paths):
        print("Render stage outputs already exist, skipping render stage.", file=sys.stderr)
        return

    print(
        "Starting render stage. Expected outputs: "
        f"{output_paths['translated_pdf']}"
        + (f" and {output_paths['interleaved_pdf']}" if args.generate_interleave_pdf else "")
        + ".",
        file=sys.stderr,
    )
    translated_pages = load_translate_stage_outputs(output_paths, page_numbers)

    # Read original PDF page objects and page sizes used by downstream renderers.
    # Output: no file written here; returns render context for PDF generation.
    selected_pdf_pages, selected_page_sizes = collect_original_pdf_metadata(pdf_reader, page_numbers)

    if not Path(output_paths["translated_pdf"]).exists():
        render_translation_only_pdf(
            args,
            output_pdf_path=output_paths["translated_pdf"],
            markdown_pages=translated_pages,
            selected_page_sizes=selected_page_sizes,
            image_root=output_paths["translate_images_dir"],
        )

    if args.generate_interleave_pdf and not Path(output_paths["interleaved_pdf"]).exists():
        render_interleaved_pdf(
            args,
            output_paths,
            translated_pages,
            selected_pdf_pages,
            selected_page_sizes,
        )
