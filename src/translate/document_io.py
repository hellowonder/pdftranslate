#!/usr/bin/env python3
"""
通用分页文档输入辅助函数。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import fitz


def is_pdf_input(path: str) -> bool:
    return Path(path).suffix.lower() == ".pdf"


def is_djvu_input(path: str) -> bool:
    return Path(path).suffix.lower() in {".djvu", ".djv"}


def normalize_document_input(path: str, working_dir: str) -> str:
    source_path = str(Path(path).resolve())
    if is_pdf_input(source_path):
        return source_path
    if not is_djvu_input(source_path):
        return source_path
    return convert_djvu_to_pdf(source_path, working_dir)


def convert_djvu_to_pdf(path: str, working_dir: str) -> str:
    ddjvu_bin = shutil.which("ddjvu")
    if not ddjvu_bin:
        raise RuntimeError(
            "DjVu input requires 'ddjvu', but it was not found in PATH. "
            "Install djvulibre-bin (or the equivalent package for your system)."
        )

    source = Path(path).resolve()
    output_dir = Path(working_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source.stem}.source.pdf"
    if output_path.exists():
        return str(output_path)

    result = subprocess.run(
        [ddjvu_bin, "-format=pdf", str(source), str(output_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "unknown ddjvu error").strip()
        raise RuntimeError(f"Failed to convert DjVu to PDF via ddjvu: {details}")
    if not output_path.exists():
        raise RuntimeError(f"ddjvu reported success but did not create output PDF: {output_path}")
    return str(output_path)

