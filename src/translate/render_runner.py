#!/usr/bin/env python3
"""
负责调度 WeasyPrint 渲染并组装最终 PDF。
"""
from __future__ import annotations

import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pypdf import PdfReader, PdfWriter
from tqdm import tqdm

from render_weasyprint import RenderSettings, render_markdown_pages

LAYOUT_JOB_PAGE_CHUNK_SIZE = 8


@dataclass(frozen=True)
class RenderJob:
    indices: Tuple[int, ...]
    markdown_pages: Tuple[str, ...]
    settings: RenderSettings


def _render_job(job: RenderJob) -> List[Tuple[int, bytes]]:
    rendered_items = render_markdown_pages(
        markdown_pages=job.markdown_pages,
        settings=job.settings,
        mode="per_markdown",
        return_mode="pdf_bytes",
        page_indices=job.indices,
    )
    return rendered_items or []


@dataclass
class PDFRenderRunner:
    settings: RenderSettings
    max_workers: int = 1
    original_pdf_pages: Optional[Sequence] = None
    original_page_sizes: Optional[Sequence[Tuple[float, float]]] = None

    def _page_size_for_index(self, idx: int) -> Tuple[int, int]:
        width = int(round(self.settings.page_width))
        height = int(round(self.settings.page_height))
        if self.original_page_sizes is not None and idx < len(self.original_page_sizes):
            real_width, real_height = self.original_page_sizes[idx]
            if real_width > 0 and real_height > 0:
                width = int(round(real_width))
                height = int(round(real_height))
        return width, height

    def _build_render_jobs(
        self,
        translations: Sequence[str],
    ) -> List[RenderJob]:
        jobs: List[RenderJob] = []
        current_size: Optional[Tuple[int, int]] = None
        current_indices: List[int] = []
        current_pages: List[str] = []

        def _flush_current() -> None:
            nonlocal current_size, current_indices, current_pages
            if current_size is None or not current_indices:
                return
            width, height = current_size
            jobs.append(
                RenderJob(
                    indices=tuple(current_indices),
                    markdown_pages=tuple(current_pages),
                    settings=replace(self.settings, page_width=width, page_height=height),
                )
            )
            current_size = None
            current_indices = []
            current_pages = []

        for idx, translation_md in enumerate(translations):
            if not (translation_md or "").strip():
                continue
            size_key = self._page_size_for_index(idx)
            if current_size != size_key or len(current_indices) >= LAYOUT_JOB_PAGE_CHUNK_SIZE:
                _flush_current()
                current_size = size_key
            current_indices.append(idx)
            current_pages.append(translation_md or "")

        _flush_current()
        return jobs

    def _render_translation_page_batches(
        self,
        translations: Sequence[str],
    ) -> Iterable[List[Tuple[int, bytes]]]:
        jobs = self._build_render_jobs(translations)
        if not jobs:
            return

        total_jobs = sum(len(job.indices) for job in jobs)
        if total_jobs == 0:
            return

        desc = "Rendering translation pages"
        if self.max_workers <= 1 or len(jobs) <= 1:
            with tqdm(total=total_jobs, desc=desc, unit="page") as pbar:
                for job in jobs:
                    rendered_items = _render_job(job)
                    pbar.update(len(rendered_items))
                    yield rendered_items
            return

        max_workers = min(self.max_workers, len(jobs))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_render_job, job): job for job in jobs}
            with tqdm(total=total_jobs, desc=desc, unit="page") as pbar:
                for future in as_completed(futures):
                    rendered_items = future.result()
                    pbar.update(len(rendered_items))
                    yield rendered_items

    def build_interleaved(self, translations: Sequence[str]) -> PdfWriter:
        total_tasks = len(translations)
        if total_tasks == 0:
            return PdfWriter()

        writer = PdfWriter()
        rendered_items_by_idx: Dict[int, bytes] = {}
        next_idx_to_write = 0

        def _append_original_page(idx: int) -> None:
            original_page = None
            if self.original_pdf_pages is not None and idx < len(self.original_pdf_pages):
                original_page = self.original_pdf_pages[idx]
            if original_page is None:
                raise ValueError("Original PDF page object is required when building interleaved output.")
            writer.add_page(original_page)

        with tqdm(total=total_tasks, desc="Building interleaved PDF", unit="page") as pbar:
            for rendered_items in self._render_translation_page_batches(translations):
                for idx, pdf_bytes in rendered_items:
                    rendered_items_by_idx[idx] = pdf_bytes

                while next_idx_to_write < total_tasks:
                    rendered_pdf = rendered_items_by_idx.pop(next_idx_to_write, None)
                    if rendered_pdf is not None:
                        pdf_reader = PdfReader(io.BytesIO(rendered_pdf))
                        for page in pdf_reader.pages:
                            writer.add_page(page)
                    elif (translations[next_idx_to_write] or "").strip():
                        break

                    _append_original_page(next_idx_to_write)
                    next_idx_to_write += 1
                    pbar.update(1)

            while next_idx_to_write < total_tasks:
                rendered_pdf = rendered_items_by_idx.pop(next_idx_to_write, None)
                if rendered_pdf is not None:
                    pdf_reader = PdfReader(io.BytesIO(rendered_pdf))
                    for page in pdf_reader.pages:
                        writer.add_page(page)
                elif (translations[next_idx_to_write] or "").strip():
                    raise ValueError(f"Missing rendered translation page for index {next_idx_to_write}.")

                _append_original_page(next_idx_to_write)
                next_idx_to_write += 1
                pbar.update(1)
        return writer

    def build_translation_only(self, translations: Sequence[str]) -> PdfWriter:
        settings = self.settings
        if self.original_page_sizes:
            for real_width, real_height in self.original_page_sizes:
                if real_width > 0 and real_height > 0:
                    settings = replace(
                        self.settings,
                        page_width=int(round(real_width)),
                        page_height=int(round(real_height)),
                    )
                    break

        pdf_bytes = render_markdown_pages(
            markdown_pages=translations,
            settings=settings,
            mode="continuous",
            return_mode="pdf_bytes",
        )
        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        return writer
