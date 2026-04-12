#!/usr/bin/env python3
"""
Utilities for recovering bold styling from the source PDF and reapplying it to OCR markdown.

This module works in two stages:
1. Read PDF text spans with PyMuPDF, detect bold spans from font metadata, merge nearby bold
   spans on the same line, and record each bold snippet together with a small left/right text
   context window as a ``BoldAnchor``.
2. Match those anchors back onto OCR-produced markdown using normalized text plus surrounding
   context, then wrap the matched markdown ranges in ``**...**``.

The matching is intentionally conservative:
- markdown syntax such as image/link destinations and inline code is protected
- very short bold snippets are ignored as noise
- repeated text is disambiguated with left/right context instead of raw substring matching
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import fitz


LINE_MERGE_GAP_RATIO = 0.5
MIN_BOLD_TEXT_LENGTH = 2
CONTEXT_WINDOW_CHARS = 24


@dataclass(frozen=True)
class BoldAnchor:
    """A bold snippet plus the surrounding plain-text context used for matching."""
    text: str
    left_context: str = ""
    right_context: str = ""


def normalize_text_for_matching(text: str) -> str:
    """Collapse whitespace so PDF-extracted text and OCR markdown align more reliably."""
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = (
        normalized.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )
    return re.sub(r"\s+", " ", normalized.strip())


def _markdown_protected_ranges(markdown: str) -> List[Tuple[int, int]]:
    """Return markdown syntax spans that must not be wrapped in bold markers."""
    ranges: List[Tuple[int, int]] = []
    patterns = (
        r"!\[[^\]]*\]\([^)]+\)",
        r"\[[^\]]+\]\([^)]+\)",
        r"`[^`\n]+`",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, markdown):
            ranges.append((match.start(), match.end()))
    ranges.sort()
    return ranges


def _range_overlaps_protected(start: int, end: int, protected_ranges: Sequence[Tuple[int, int]]) -> bool:
    """Check whether a candidate bold range intersects a protected markdown syntax span."""
    for protected_start, protected_end in protected_ranges:
        if protected_end <= start:
            continue
        if protected_start >= end:
            break
        return True
    return False


def _range_inside_heading_content(markdown: str, start: int, end: int) -> bool:
    """Skip wrapping text that is already part of a Markdown heading line."""
    line_start = markdown.rfind("\n", 0, start) + 1
    line_end = markdown.find("\n", end)
    if line_end < 0:
        line_end = len(markdown)
    line = markdown[line_start:line_end]
    if not re.match(r"^\s*#{1,6}\s+", line):
        return False

    content_start_in_line = 0
    while content_start_in_line < len(line) and line[content_start_in_line].isspace():
        content_start_in_line += 1
    while content_start_in_line < len(line) and line[content_start_in_line] == "#":
        content_start_in_line += 1
    while content_start_in_line < len(line) and line[content_start_in_line].isspace():
        content_start_in_line += 1

    content_start = line_start + content_start_in_line
    return start >= content_start and end <= line_end


def is_bold_span(span: dict) -> bool:
    """Heuristically decide whether a PDF text span is bold based on font metadata."""
    font = (span.get("font") or "").lower()
    flags = span.get("flags") or 0
    return any(token in font for token in ("bold", "black", "heavy", "semibold")) or bool(flags & 16)


def _span_gap(previous_span: dict, current_span: dict) -> Optional[float]:
    """Measure horizontal distance between two neighboring spans on the same line."""
    previous_bbox = previous_span.get("bbox") or ()
    current_bbox = current_span.get("bbox") or ()
    if len(previous_bbox) < 4 or len(current_bbox) < 4:
        return None
    try:
        return float(current_bbox[0]) - float(previous_bbox[2])
    except (TypeError, ValueError):
        return None


def _should_merge_bold_spans(previous_span: dict, current_span: dict) -> bool:
    """Treat nearby bold spans as one phrase when their horizontal gap is small enough."""
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


def _append_span_text(parts: List[str], previous_span: Optional[dict], span: dict) -> None:
    """Append a span to a reconstructed line, inserting a space when the PDF gap implies one."""
    if parts and previous_span is not None:
        gap = _span_gap(previous_span, span)
        if gap is not None and gap > 0:
            parts.append(" ")
    parts.append(span.get("text", ""))


def _build_line_text_and_positions(line: dict) -> Tuple[str, List[Tuple[dict, int, int]]]:
    """Rebuild a PDF line into plain text and map each span back to character offsets."""
    parts: List[str] = []
    positions: List[Tuple[dict, int, int]] = []
    previous_span: Optional[dict] = None
    for span in line.get("spans", []):
        span_text = span.get("text", "")
        if not span_text:
            continue
        start = len("".join(parts))
        _append_span_text(parts, previous_span, span)
        end = len("".join(parts))
        positions.append((span, start, end))
        previous_span = span
    return "".join(parts), positions


def _extract_context_window(text: str, start: int, end: int, window_chars: int = CONTEXT_WINDOW_CHARS) -> Tuple[str, str]:
    """Capture a small normalized left/right context window around a bold snippet."""
    left_source = text[:start]
    right_source = text[end:]
    left_context = normalize_text_for_matching(left_source[-window_chars:])
    right_context = normalize_text_for_matching(right_source[:window_chars])
    return left_context, right_context


def _anchor_from_line_text(line_text: str, start: int, end: int) -> Optional[BoldAnchor]:
    """Build a matching anchor from a line slice, skipping snippets that are too short to trust."""
    text = normalize_text_for_matching(line_text[start:end])
    if len(text) < MIN_BOLD_TEXT_LENGTH:
        return None
    left_context, right_context = _extract_context_window(line_text, start, end)
    return BoldAnchor(
        text=text,
        left_context=left_context,
        right_context=right_context,
    )


def _append_anchor_if_valid(
    anchors: List[BoldAnchor],
    line_text: str,
    start: Optional[int],
    end: Optional[int],
) -> None:
    """Create and append an anchor only when the tracked range is complete and trustworthy."""
    if start is None or end is None:
        return
    anchor = _anchor_from_line_text(line_text, start, end)
    if anchor is not None:
        anchors.append(anchor)


def extract_bold_texts_from_line(line: dict) -> List[BoldAnchor]:
    """Extract bold anchors from one PDF line."""
    line_text, positions = _build_line_text_and_positions(line)
    anchors: List[BoldAnchor] = []
    current_bold_start: Optional[int] = None
    current_bold_end: Optional[int] = None
    previous_bold_span: Optional[dict] = None

    for span, span_start, span_end in positions:
        span_text = span.get("text", "")
        span_is_bold = is_bold_span(span)
        has_visible_text = bool(span_text.strip())

        if not span_is_bold:
            _append_anchor_if_valid(anchors, line_text, current_bold_start, current_bold_end)
            current_bold_start = None
            current_bold_end = None
            previous_bold_span = None
            continue

        if not has_visible_text:
            if current_bold_start is not None:
                current_bold_end = span_end
                previous_bold_span = span
            continue

        starts_new_anchor = (
            current_bold_start is not None
            and previous_bold_span is not None
            and not _should_merge_bold_spans(previous_bold_span, span)
        )
        if starts_new_anchor:
            _append_anchor_if_valid(anchors, line_text, current_bold_start, current_bold_end)
            current_bold_start = span_start

        if current_bold_start is None:
            current_bold_start = span_start
        current_bold_end = span_end
        previous_bold_span = span

    _append_anchor_if_valid(anchors, line_text, current_bold_start, current_bold_end)
    return anchors


def extract_bold_texts_from_page_dict(page_dict: dict) -> List[BoldAnchor]:
    """Extract bold phrases from a PyMuPDF page dict and attach local textual context to each one."""
    bold_items: List[BoldAnchor] = []
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            bold_items.extend(extract_bold_texts_from_line(line))
    return bold_items


def extract_bold_texts_from_page(doc: fitz.Document, page_number: int) -> List[BoldAnchor]:
    """Extract bold anchors from one 1-based PDF page number in an already-open document."""
    if page_number < 1 or page_number > len(doc):
        return []
    page = doc[page_number - 1]
    return extract_bold_texts_from_page_dict(page.get_text("dict"))


def extract_bold_texts_for_page(pdf_path: str, page_number: int) -> List[BoldAnchor]:
    """Open the PDF, read one page, and return its bold anchors."""
    doc = fitz.open(pdf_path)
    try:
        return extract_bold_texts_from_page(doc, page_number)
    finally:
        doc.close()


def extract_bold_text_by_page(pdf_path: str, page_numbers: Sequence[int]) -> Dict[int, List[BoldAnchor]]:
    """Read the requested PDF pages and return bold anchors keyed by 1-based page number."""
    doc = fitz.open(pdf_path)
    try:
        bold_by_page: Dict[int, List[BoldAnchor]] = {}
        for page_number in page_numbers:
            bold_by_page[page_number] = extract_bold_texts_from_page(doc, page_number)
        return bold_by_page
    finally:
        doc.close()


def build_normalized_index_map(text: str) -> Tuple[str, List[int]]:
    """Normalize whitespace while preserving a map back to original character offsets."""
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


def _best_context_span(
    markdown: str,
    normalized_markdown: str,
    index_map: Sequence[int],
    anchor: BoldAnchor,
    protected_ranges: Sequence[Tuple[int, int]],
    occupied_ranges: Sequence[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """Find the markdown span whose surrounding text best matches a bold anchor."""
    needle = normalize_text_for_matching(anchor.text)
    if len(needle) < MIN_BOLD_TEXT_LENGTH:
        return None

    left_context = normalize_text_for_matching(anchor.left_context)
    right_context = normalize_text_for_matching(anchor.right_context)

    best_candidate: Optional[Tuple[int, int]] = None
    best_score = -1
    candidate_count = 0
    unique_candidate: Optional[Tuple[int, int]] = None
    found_at = normalized_markdown.find(needle)
    while found_at >= 0:
        norm_end = found_at + len(needle)
        original_start = index_map[found_at]
        original_end = index_map[norm_end - 1] + 1

        if _range_overlaps_protected(original_start, original_end, protected_ranges):
            found_at = normalized_markdown.find(needle, found_at + 1)
            continue
        if _range_inside_heading_content(markdown, original_start, original_end):
            found_at = normalized_markdown.find(needle, found_at + 1)
            continue
        if any(original_start < end and original_end > start for start, end in occupied_ranges):
            found_at = normalized_markdown.find(needle, found_at + 1)
            continue
        if normalized_markdown[found_at:norm_end].startswith("**") and normalized_markdown[found_at:norm_end].endswith("**"):
            found_at = normalized_markdown.find(needle, found_at + 1)
            continue

        candidate_count += 1
        unique_candidate = (original_start, original_end)
        context_window = max(len(left_context) + CONTEXT_WINDOW_CHARS, len(right_context) + CONTEXT_WINDOW_CHARS, CONTEXT_WINDOW_CHARS)
        left_window = normalize_text_for_matching(normalized_markdown[max(0, found_at - context_window) : found_at])
        right_window = normalize_text_for_matching(normalized_markdown[norm_end : norm_end + context_window])
        left_score = len(left_context) if left_context and left_window.endswith(left_context) else 0
        right_score = len(right_context) if right_context and right_window.startswith(right_context) else 0
        if not left_context and not right_context:
            found_at = normalized_markdown.find(needle, found_at + 1)
            continue
        if left_context and right_context and not left_score and not right_score:
            found_at = normalized_markdown.find(needle, found_at + 1)
            continue
        if left_context and not right_context and not left_score:
            found_at = normalized_markdown.find(needle, found_at + 1)
            continue
        if right_context and not left_context and not right_score:
            found_at = normalized_markdown.find(needle, found_at + 1)
            continue

        score = left_score + right_score
        if score > best_score:
            best_score = score
            best_candidate = (original_start, original_end)

        found_at = normalized_markdown.find(needle, found_at + 1)

    if best_candidate is not None:
        return best_candidate
    if candidate_count == 1:
        return unique_candidate
    return None


def apply_bold_texts_to_markdown(markdown: str, bold_texts: Sequence[BoldAnchor]) -> str:
    """Apply bold anchors to markdown by matching both the target text and its nearby context."""
    normalized_markdown, index_map = build_normalized_index_map(markdown)
    if not normalized_markdown:
        return markdown

    ranges: List[Tuple[int, int]] = []
    protected_ranges = _markdown_protected_ranges(markdown)
    for anchor in bold_texts:
        candidate = _best_context_span(
            markdown=markdown,
            normalized_markdown=normalized_markdown,
            index_map=index_map,
            anchor=anchor,
            protected_ranges=protected_ranges,
            occupied_ranges=ranges,
        )
        if candidate is not None:
            ranges.append(candidate)

    if not ranges:
        return markdown

    enriched = markdown
    for start, end in sorted(ranges, key=lambda item: item[0], reverse=True):
        enriched = enriched[:start] + "**" + enriched[start:end] + "**" + enriched[end:]
    return enriched


def apply_pdf_bold_onepage(doc: fitz.Document, page_number: int, page_markdown: str) -> str:
    """Extract bold anchors for one page from an open PDF and apply them to that page markdown."""
    try:
        bold_texts = extract_bold_texts_from_page(doc, page_number)
    except Exception:
        bold_texts = []
    if not bold_texts:
        return page_markdown
    return apply_bold_texts_to_markdown(page_markdown, bold_texts)


def apply_pdf_bold_marks(
    pdf_path: str,
    page_numbers: List[int],
    pages: List[str],
) -> List[str]:
    """Open the PDF once and apply bold styling page-by-page to the corresponding markdown pages."""
    doc = fitz.open(pdf_path)
    try:
        return [
            apply_pdf_bold_onepage(doc, page_number, page_markdown)
            for page_number, page_markdown in zip(page_numbers, pages)
        ]
    finally:
        doc.close()
