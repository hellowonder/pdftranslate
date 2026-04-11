#!/usr/bin/env python3
"""
Add a bottom-right badge (default: "CN") to the cover image of an EPUB in-place.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from ebooklib import epub

from epub_translate import (
    DEFAULT_TRANSLATION_CLASS,
    SUPPORTED_BLOCK_TAGS,
    EpubProcessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paint a badge onto an EPUB cover image in-place.")
    parser.add_argument("input", help="Path to the EPUB file to modify.")
    parser.add_argument(
        "--badge-text",
        default="CN",
        help='Badge text to paint on the cover (default: "CN").',
    )
    return parser.parse_args()


def write_in_place(book: epub.EpubBook, path: str) -> None:
    """
    Write to a temp file in the same directory then replace the original to avoid corruption.
    """
    directory, base = os.path.split(os.path.abspath(path))
    os.makedirs(directory or ".", exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=base + ".", suffix=".tmp", dir=directory or ".")
    os.close(fd)
    try:
        epub.write_epub(temp_path, book)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Input EPUB not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    book = epub.read_epub(input_path)
    processor = EpubProcessor(
        translation_service=None,
        translation_class=DEFAULT_TRANSLATION_CLASS,
        block_tags=SUPPORTED_BLOCK_TAGS,
        translation_workers=1,
    )
    processor._add_cover_badge(book, args.badge_text)
    write_in_place(book, input_path)
    print(f"Badge applied to cover and saved to {input_path}")


if __name__ == "__main__":
    main()
