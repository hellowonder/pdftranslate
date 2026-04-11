#!/usr/bin/env python3
import shutil
import sys
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Sequence, Tuple

from tqdm import tqdm

from ocr_stage import notify_vllm_control_endpoint
from paged_markdown_io import (
    build_page_markdown_path,
    page_markdown_outputs_exist,
    read_page_markdown_files,
    read_paged_markdown_document,
    write_paged_markdown_document,
)
from translate_service import init_translation_service


def image_dirs_match(source_dir: str, target_dir: str) -> bool:
    if not Path(target_dir).exists():
        return False
    source_files = sorted(path.relative_to(source_dir).as_posix() for path in Path(source_dir).glob("*") if path.is_file())
    target_files = sorted(path.relative_to(target_dir).as_posix() for path in Path(target_dir).glob("*") if path.is_file())
    return source_files == target_files


def sync_stage_images(source_dir: str, target_dir: str) -> None:
    start_time = time.monotonic()
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    copied_count = 0
    skipped_count = 0
    for source_path in Path(source_dir).glob("*"):
        if not source_path.is_file():
            continue
        target_path = Path(target_dir) / source_path.name
        if target_path.exists():
            skipped_count += 1
            continue
        shutil.copy2(source_path, target_path)
        copied_count += 1
    elapsed = time.monotonic() - start_time
    print(
        f"Synced translate images from {source_dir} to {target_dir}: "
        f"copied={copied_count}, skipped_existing={skipped_count}, elapsed={elapsed:.2f}s",
        file=sys.stderr,
    )


def translate_stage_outputs_exist(output_paths: dict[str, str], page_numbers: Sequence[int] | None = None) -> bool:
    if not Path(output_paths["translated_markdown"]).exists():
        return False
    if page_numbers is None:
        raise ValueError("page_numbers is required for translate stage output checks.")
    return page_markdown_outputs_exist(output_paths["translate_dir"], page_numbers, suffix="_cn") and image_dirs_match(
        output_paths["ocr_images_dir"],
        output_paths["translate_images_dir"],
    )


def should_manage_translation_vllm_lifecycle(args) -> bool:
    return bool(getattr(args, "vllm_sleep", False)) and not getattr(args, "no_translation", False)


def translate_page_markdown(
    args,
    output_paths: dict[str, str],
    page_number: int,
    page_text: str,
    translation_service,
) -> str:
    page_path = build_page_markdown_path(output_paths["translate_dir"], page_number, suffix="_cn")
    if Path(page_path).exists():
        print(f"Page {page_number}: translation cache hit at {page_path}", file=sys.stderr)
        return Path(page_path).read_text(encoding="utf-8")
    start_time = time.monotonic()
    print(
        f"Page {page_number}: starting translation, chars={len(page_text)}, output={page_path}",
        file=sys.stderr,
    )
    if args.skip_first_page_translation and page_number == 1:
        translated_text = page_text
    elif args.no_translation:
        translated_text = page_text
    else:
        translated_text = translation_service.translate_text_block(page_text)
    Path(page_path).write_text(translated_text, encoding="utf-8")
    elapsed = time.monotonic() - start_time
    print(
        f"Page {page_number}: finished translation, output_chars={len(translated_text)}, elapsed={elapsed:.2f}s",
        file=sys.stderr,
    )
    return translated_text


def run_translate_pages_step(
    args,
    output_paths: dict[str, str],
    page_numbers: Sequence[int]
) -> List[str]:
    stage_start = time.monotonic()
    # Load the final OCR page markdown that serves as the translation stage input.
    # Input: ocr/page_XXXX.md
    markdown_pages = read_page_markdown_files(output_paths["ocr_dir"], page_numbers)
    total_chars = sum(len(page) for page in markdown_pages)
    print(
        f"Loaded {len(markdown_pages)} OCR markdown pages for translation: total_chars={total_chars}",
        file=sys.stderr,
    )
    translation_service = None if args.no_translation else init_translation_service(args)
    translated_pages = [""] * len(markdown_pages)
    page_jobs = list(enumerate(zip(page_numbers, markdown_pages)))

    def _translate_page(job: Tuple[int, Tuple[int, str]]) -> Tuple[int, str]:
        idx, (page_number, page_text) = job
        translated_text = translate_page_markdown(
            args,
            output_paths,
            page_number,
            page_text,
            translation_service,
        )
        return idx, translated_text

    max_workers = min(max(1, args.translation_workers), len(page_jobs))
    if not page_jobs:
        return translated_pages
    print(
        f"Translation page scheduling: workers={max_workers}, pages={len(page_jobs)}, "
        f"no_translation={getattr(args, 'no_translation', False)}, "
        f"skip_first_page_translation={getattr(args, 'skip_first_page_translation', False)}",
        file=sys.stderr,
    )

    progress_label = f"Translating pages ({max_workers} workers)"
    use_tqdm = hasattr(tqdm, "get_lock")
    if use_tqdm:
        progress_context = tqdm(total=len(page_jobs), desc=progress_label, unit="page")
    else:
        progress_context = nullcontext()
        print(f"{progress_label}: 0/{len(page_jobs)}", file=sys.stderr)
    with progress_context as progress_bar:
        if max_workers <= 1:
            for completed_count, job in enumerate(page_jobs, start=1):
                idx, translated_text = _translate_page(job)
                translated_pages[idx] = translated_text
                if progress_bar is not None:
                    progress_bar.update(1)
                else:
                    print(f"{progress_label}: {completed_count}/{len(page_jobs)}", file=sys.stderr)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_translate_page, job) for job in page_jobs]
                for completed_count, future in enumerate(as_completed(futures), start=1):
                    idx, translated_text = future.result()
                    translated_pages[idx] = translated_text
                    if progress_bar is not None:
                        progress_bar.update(1)
                    else:
                        print(f"{progress_label}: {completed_count}/{len(page_jobs)}", file=sys.stderr)
    elapsed = time.monotonic() - stage_start
    print(
        f"Finished page translation step: pages={len(page_jobs)}, total_chars={total_chars}, elapsed={elapsed:.2f}s",
        file=sys.stderr,
    )
    return translated_pages


def run_translate_stage(args, output_paths: dict[str, str], page_numbers: Sequence[int]) -> None:
    stage_start = time.monotonic()
    if translate_stage_outputs_exist(output_paths, page_numbers):
        print("Translate stage outputs already exist, skipping translate stage.", file=sys.stderr)
        return

    print(
        "Starting translate stage. Expected output: "
        f"{output_paths['translated_markdown']}. "
        f"pages={len(page_numbers)}, workers={getattr(args, 'translation_workers', 'n/a')}, "
        f"base_url={getattr(args, 'translation_base_url', 'n/a')}, "
        f"model={getattr(args, 'translation_model', 'n/a')}",
        file=sys.stderr,
    )
    manage_vllm_lifecycle = should_manage_translation_vllm_lifecycle(args)
    if manage_vllm_lifecycle:
        notify_vllm_control_endpoint(args.translation_base_url, "wake_up")
    try:
        # Copy OCR-extracted images into translate/images/ so translated markdown can reference the same assets.
        # Output: translate/images/*
        sync_stage_images(output_paths["ocr_images_dir"], output_paths["translate_images_dir"])

        # Translate each page independently and persist the per-page translation result.
        # Output: translate/page_XXXX_cn.md
        run_translate_pages_step(args, output_paths, page_numbers)

        # Concatenate translated pages into the stage-level markdown consumed by render stage.
        # Output: translate/<base>_cn.md
        concat_start = time.monotonic()
        translated_pages = read_page_markdown_files(output_paths["translate_dir"], page_numbers, suffix="_cn")
        write_paged_markdown_document(output_paths["translated_markdown"], translated_pages)
        print(
            f"Wrote merged translated markdown to {output_paths['translated_markdown']} "
            f"in {time.monotonic() - concat_start:.2f}s",
            file=sys.stderr,
        )
    finally:
        if manage_vllm_lifecycle:
            notify_vllm_control_endpoint(args.translation_base_url, "sleep?level=1")
    print(
        f"Translate stage completed in {time.monotonic() - stage_start:.2f}s",
        file=sys.stderr,
    )
    return


def load_translate_stage_outputs(output_paths: dict[str, str], page_numbers: Sequence[int]) -> List[str]:
    if Path(output_paths["translated_markdown"]).exists():
        return read_paged_markdown_document(output_paths["translated_markdown"], expected_pages=len(page_numbers))
    translated_pages = read_page_markdown_files(output_paths["translate_dir"], page_numbers, suffix="_cn")
    write_paged_markdown_document(output_paths["translated_markdown"], translated_pages)
    return translated_pages
