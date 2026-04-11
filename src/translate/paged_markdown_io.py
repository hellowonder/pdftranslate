#!/usr/bin/env python3
from pathlib import Path
from typing import List, Optional, Sequence

PAGE_SEPARATOR = "\n<!--- Page Split --->\n"


def write_paged_markdown_document(path: str, pages: Sequence[str]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(PAGE_SEPARATOR.join(page.strip() for page in pages))


def read_paged_markdown_document(path: str, expected_pages: Optional[int] = None) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if expected_pages == 0:
        return []
    pages = content.split(PAGE_SEPARATOR)
    if expected_pages is not None and len(pages) != expected_pages:
        raise ValueError(
            f"Markdown page count mismatch for {path}: expected {expected_pages}, got {len(pages)}."
        )
    return pages


def build_page_markdown_path(stage_dir: str, page_number: int, suffix: str = "") -> str:
    return str(Path(stage_dir) / f"page_{page_number:04d}{suffix}.md")


def write_page_markdown(stage_dir: str, page_number: int, content: str, suffix: str = "") -> None:
    Path(stage_dir).mkdir(parents=True, exist_ok=True)
    Path(build_page_markdown_path(stage_dir, page_number, suffix=suffix)).write_text(content, encoding="utf-8")


def read_page_markdown_files(stage_dir: str, page_numbers: Sequence[int], suffix: str = "") -> List[str]:
    return [
        Path(build_page_markdown_path(stage_dir, page_number, suffix=suffix)).read_text(encoding="utf-8")
        for page_number in page_numbers
    ]


def page_markdown_outputs_exist(stage_dir: str, page_numbers: Sequence[int], suffix: str = "") -> bool:
    if not page_numbers:
        return False
    return all(Path(build_page_markdown_path(stage_dir, page_number, suffix=suffix)).exists() for page_number in page_numbers)
