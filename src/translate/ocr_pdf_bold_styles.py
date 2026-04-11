#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import fitz


LINE_MERGE_GAP_RATIO = 0.5


def normalize_text_for_matching(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def is_bold_span(span: dict) -> bool:
    font = (span.get("font") or "").lower()
    flags = span.get("flags") or 0
    return any(token in font for token in ("bold", "black", "heavy", "semibold")) or bool(flags & 16)


def _span_text_width(span: dict) -> float:
    bbox = span.get("bbox") or ()
    if len(bbox) >= 4:
        try:
            return max(0.0, float(bbox[2]) - float(bbox[0]))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _span_gap(previous_span: dict, current_span: dict) -> Optional[float]:
    previous_bbox = previous_span.get("bbox") or ()
    current_bbox = current_span.get("bbox") or ()
    if len(previous_bbox) < 4 or len(current_bbox) < 4:
        return None
    try:
        return float(current_bbox[0]) - float(previous_bbox[2])
    except (TypeError, ValueError):
        return None


def _should_merge_bold_spans(previous_span: dict, current_span: dict) -> bool:
    previous_text = previous_span.get("text", "")
    current_text = current_span.get("text", "")
    if not previous_text or not current_text:
        return False

    gap = _span_gap(previous_span, current_span)
    if gap is None:
        return True
    if gap < 0:
        return True

    previous_size = previous_span.get("size") or 0
    current_size = current_span.get("size") or 0
    reference_size = max(float(previous_size), float(current_size), 1.0)
    threshold = reference_size * LINE_MERGE_GAP_RATIO
    return gap <= threshold


def extract_bold_texts_from_page_dict(page_dict: dict) -> List[str]:
    bold_items: List[str] = []
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            current_parts: List[str] = []
            previous_bold_span: Optional[dict] = None
            for span in line.get("spans", []):
                if not is_bold_span(span):
                    if current_parts:
                        normalized = normalize_text_for_matching("".join(current_parts))
                        if normalized:
                            bold_items.append(normalized)
                    current_parts = []
                    previous_bold_span = None
                    continue

                span_text = span.get("text", "")
                if not span_text.strip():
                    continue

                if current_parts and previous_bold_span is not None and not _should_merge_bold_spans(previous_bold_span, span):
                    normalized = normalize_text_for_matching("".join(current_parts))
                    if normalized:
                        bold_items.append(normalized)
                    current_parts = []

                if current_parts:
                    gap = _span_gap(previous_bold_span, span) if previous_bold_span is not None else None
                    if gap is not None and gap > 0:
                        current_parts.append(" ")
                current_parts.append(span_text)
                previous_bold_span = span

            if current_parts:
                normalized = normalize_text_for_matching("".join(current_parts))
                if normalized:
                    bold_items.append(normalized)
    return bold_items


def extract_bold_text_by_page(pdf_path: str, page_numbers: Sequence[int]) -> Dict[int, List[str]]:
    doc = fitz.open(pdf_path)
    try:
        bold_by_page: Dict[int, List[str]] = {}
        for page_number in page_numbers:
            if page_number < 1 or page_number > len(doc):
                bold_by_page[page_number] = []
                continue
            page = doc[page_number - 1]
            text = page.get_text("dict")
            bold_by_page[page_number] = extract_bold_texts_from_page_dict(text)
        return bold_by_page
    finally:
        doc.close()


def build_normalized_index_map(text: str) -> Tuple[str, List[int]]:
    normalized_chars: List[str] = []
    index_map: List[int] = []
    pending_space = False
    pending_space_index: Optional[int] = None

    for index, char in enumerate(text):
        if char.isspace():
            if normalized_chars:
                pending_space = True
                if pending_space_index is None:
                    pending_space_index = index
            continue
        if pending_space:
            normalized_chars.append(" ")
            index_map.append(pending_space_index if pending_space_index is not None else index)
            pending_space = False
            pending_space_index = None
        normalized_chars.append(char)
        index_map.append(index)

    return "".join(normalized_chars), index_map


def apply_bold_texts_to_markdown(markdown: str, bold_texts: Sequence[str]) -> str:
    normalized_markdown, index_map = build_normalized_index_map(markdown)
    if not normalized_markdown:
        return markdown

    ranges: List[Tuple[int, int]] = []
    search_start = 0
    for bold_text in bold_texts:
        needle = normalize_text_for_matching(bold_text)
        if not needle:
            continue
        found_at = normalized_markdown.find(needle, search_start)
        if found_at < 0:
            found_at = normalized_markdown.find(needle)
        if found_at < 0:
            continue

        norm_end = found_at + len(needle) - 1
        original_start = index_map[found_at]
        original_end = index_map[norm_end] + 1

        if ranges and original_start < ranges[-1][1]:
            continue
        if markdown[original_start:original_end].startswith("**") and markdown[original_start:original_end].endswith("**"):
            search_start = found_at + len(needle)
            continue

        ranges.append((original_start, original_end))
        search_start = found_at + len(needle)

    if not ranges:
        return markdown

    enriched = markdown
    for start, end in reversed(ranges):
        enriched = enriched[:start] + "**" + enriched[start:end] + "**" + enriched[end:]
    return enriched
