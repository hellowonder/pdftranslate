#!/usr/bin/env python3
"""
Utilities for recovering text styling from the source PDF and reapplying it to OCR markdown.

This module works in two stages:
1. Read PDF text spans with PyMuPDF, detect styled spans from font metadata, merge nearby
   spans on the same line, and record each styled snippet together with a small left/right
   text context window as a ``StyleAnchor``.
2. Match those anchors back onto OCR-produced markdown using normalized text plus surrounding
   context, then wrap the matched markdown ranges in Markdown emphasis markers.

The matching is intentionally conservative:
- markdown syntax such as image/link destinations and inline code is protected
- very short styled snippets are ignored as noise
- repeated text is disambiguated with left/right context instead of raw substring matching
- overlapping matches are skipped unless they refer to the exact same source range
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import fitz


LINE_MERGE_GAP_RATIO = 0.5
MIN_STYLE_TEXT_LENGTH = 2
CONTEXT_WINDOW_CHARS = 24
STYLE_BOLD = "bold"
STYLE_ITALIC = "italic"


@dataclass(frozen=True)
class StyleAnchor:
    """A styled snippet plus the surrounding plain-text context used for matching."""

    text: str
    style: str
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
    """Return markdown syntax spans that must not be wrapped in style markers."""
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
    """Check whether a candidate style range intersects a protected markdown syntax span."""
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


def is_italic_span(span: dict) -> bool:
    """Heuristically decide whether a PDF text span is italic/oblique based on font metadata."""
    font = (span.get("font") or "").lower()
    flags = span.get("flags") or 0
    return any(token in font for token in ("italic", "oblique", "slanted")) or bool(flags & 2)


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


def _should_merge_styled_spans(previous_span: dict, current_span: dict) -> bool:
    """Treat nearby styled spans as one phrase when their horizontal gap is small enough."""
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
    current_length = 0
    for span in line.get("spans", []):
        span_text = span.get("text", "")
        if not span_text:
            continue
        start = current_length
        if parts and previous_span is not None:
            gap = _span_gap(previous_span, span)
            if gap is not None and gap > 0:
                parts.append(" ")
                current_length += 1
        parts.append(span_text)
        current_length += len(span_text)
        end = current_length
        positions.append((span, start, end))
        previous_span = span
    return "".join(parts), positions


def _extract_context_window(text: str, start: int, end: int, window_chars: int = CONTEXT_WINDOW_CHARS) -> Tuple[str, str]:
    """Capture a small normalized left/right context window around a styled snippet."""
    left_source = text[:start]
    right_source = text[end:]
    left_context = normalize_text_for_matching(left_source[-window_chars:])
    right_context = normalize_text_for_matching(right_source[:window_chars])
    return left_context, right_context


def _anchor_from_line_text(line_text: str, start: int, end: int, style: str) -> Optional[StyleAnchor]:
    """Build a matching anchor from a line slice, skipping snippets that are too short to trust."""
    text = normalize_text_for_matching(line_text[start:end])
    if len(text) < MIN_STYLE_TEXT_LENGTH:
        return None
    left_context, right_context = _extract_context_window(line_text, start, end)
    return StyleAnchor(
        text=text,
        style=style,
        left_context=left_context,
        right_context=right_context,
    )


def _append_anchor_if_valid(
    anchors: List[StyleAnchor],
    line_text: str,
    start: Optional[int],
    end: Optional[int],
    style: str,
) -> None:
    """Create and append an anchor only when the tracked range is complete and trustworthy."""
    if start is None or end is None:
        return
    anchor = _anchor_from_line_text(line_text, start, end, style)
    if anchor is not None:
        anchors.append(anchor)


def extract_style_texts_from_line(line: dict, style: str) -> List[StyleAnchor]:
    """Extract anchors for one style from one PDF line."""
    if style == STYLE_BOLD:
        style_predicate = is_bold_span
    elif style == STYLE_ITALIC:
        style_predicate = is_italic_span
    else:
        raise ValueError(f"Unsupported style: {style}")

    line_text, positions = _build_line_text_and_positions(line)
    anchors: List[StyleAnchor] = []
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    previous_style_span: Optional[dict] = None

    for span, span_start, span_end in positions:
        span_text = span.get("text", "")
        span_has_style = style_predicate(span)
        has_visible_text = bool(span_text.strip())

        if not span_has_style:
            _append_anchor_if_valid(anchors, line_text, current_start, current_end, style)
            current_start = None
            current_end = None
            previous_style_span = None
            continue

        if not has_visible_text:
            if current_start is not None:
                current_end = span_end
                previous_style_span = span
            continue

        starts_new_anchor = (
            current_start is not None
            and previous_style_span is not None
            and not _should_merge_styled_spans(previous_style_span, span)
        )
        if starts_new_anchor:
            _append_anchor_if_valid(anchors, line_text, current_start, current_end, style)
            current_start = span_start

        if current_start is None:
            current_start = span_start
        current_end = span_end
        previous_style_span = span

    _append_anchor_if_valid(anchors, line_text, current_start, current_end, style)
    return anchors


def extract_bold_texts_from_line(line: dict) -> List[StyleAnchor]:
    """Extract bold anchors from one PDF line."""
    return extract_style_texts_from_line(line, STYLE_BOLD)


def extract_italic_texts_from_line(line: dict) -> List[StyleAnchor]:
    """Extract italic anchors from one PDF line."""
    return extract_style_texts_from_line(line, STYLE_ITALIC)


def extract_style_texts_from_page_dict(page_dict: dict, style: str) -> List[StyleAnchor]:
    """Extract styled phrases from a PyMuPDF page dict and attach local textual context."""
    style_items: List[StyleAnchor] = []
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            style_items.extend(extract_style_texts_from_line(line, style))
    return style_items


def extract_bold_texts_from_page_dict(page_dict: dict) -> List[StyleAnchor]:
    """Extract bold phrases from a PyMuPDF page dict and attach local textual context."""
    return extract_style_texts_from_page_dict(page_dict, STYLE_BOLD)


def extract_italic_texts_from_page_dict(page_dict: dict) -> List[StyleAnchor]:
    """Extract italic phrases from a PyMuPDF page dict and attach local textual context."""
    return extract_style_texts_from_page_dict(page_dict, STYLE_ITALIC)


def extract_style_texts_from_page(doc: fitz.Document, page_number: int, style: str) -> List[StyleAnchor]:
    """Extract anchors for one style from one 1-based PDF page number in an already-open document."""
    if page_number < 1 or page_number > len(doc):
        return []
    page = doc[page_number - 1]
    return extract_style_texts_from_page_dict(page.get_text("dict"), style)


def extract_bold_texts_from_page(doc: fitz.Document, page_number: int) -> List[StyleAnchor]:
    """Extract bold anchors from one page in an already-open document."""
    return extract_style_texts_from_page(doc, page_number, STYLE_BOLD)


def extract_italic_texts_from_page(doc: fitz.Document, page_number: int) -> List[StyleAnchor]:
    """Extract italic anchors from one page in an already-open document."""
    return extract_style_texts_from_page(doc, page_number, STYLE_ITALIC)


def extract_style_texts_for_page(pdf_path: str, page_number: int, style: str) -> List[StyleAnchor]:
    """Open the PDF, read one page, and return anchors for the requested style."""
    doc = fitz.open(pdf_path)
    try:
        return extract_style_texts_from_page(doc, page_number, style)
    finally:
        doc.close()


def extract_bold_texts_for_page(pdf_path: str, page_number: int) -> List[StyleAnchor]:
    """Open the PDF, read one page, and return its bold anchors."""
    return extract_style_texts_for_page(pdf_path, page_number, STYLE_BOLD)


def extract_italic_texts_for_page(pdf_path: str, page_number: int) -> List[StyleAnchor]:
    """Open the PDF, read one page, and return its italic anchors."""
    return extract_style_texts_for_page(pdf_path, page_number, STYLE_ITALIC)


def extract_style_text_by_page(pdf_path: str, page_numbers: Sequence[int], style: str) -> Dict[int, List[StyleAnchor]]:
    """Read the requested PDF pages and return style anchors keyed by 1-based page number."""
    doc = fitz.open(pdf_path)
    try:
        style_by_page: Dict[int, List[StyleAnchor]] = {}
        for page_number in page_numbers:
            style_by_page[page_number] = extract_style_texts_from_page(doc, page_number, style)
        return style_by_page
    finally:
        doc.close()


def extract_bold_text_by_page(pdf_path: str, page_numbers: Sequence[int]) -> Dict[int, List[StyleAnchor]]:
    """Read the requested PDF pages and return bold anchors keyed by 1-based page number."""
    return extract_style_text_by_page(pdf_path, page_numbers, STYLE_BOLD)


def extract_italic_text_by_page(pdf_path: str, page_numbers: Sequence[int]) -> Dict[int, List[StyleAnchor]]:
    """Read the requested PDF pages and return italic anchors keyed by 1-based page number."""
    return extract_style_text_by_page(pdf_path, page_numbers, STYLE_ITALIC)


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
    anchor: StyleAnchor,
    protected_ranges: Sequence[Tuple[int, int]],
    occupied_ranges: Sequence[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """Find the markdown span whose surrounding text best matches a style anchor."""
    needle = normalize_text_for_matching(anchor.text)
    if len(needle) < MIN_STYLE_TEXT_LENGTH:
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
        if any(
            original_start < end and original_end > start and (original_start, original_end) != (start, end)
            for start, end in occupied_ranges
        ):
            found_at = normalized_markdown.find(needle, found_at + 1)
            continue

        candidate_count += 1
        unique_candidate = (original_start, original_end)
        context_window = max(
            len(left_context) + CONTEXT_WINDOW_CHARS,
            len(right_context) + CONTEXT_WINDOW_CHARS,
            CONTEXT_WINDOW_CHARS,
        )
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


def _marker_for_styles(styles: Sequence[str]) -> str:
    style_set = set(styles)
    if STYLE_BOLD in style_set and STYLE_ITALIC in style_set:
        return "***"
    if STYLE_BOLD in style_set:
        return "**"
    if STYLE_ITALIC in style_set:
        return "*"
    return ""


def apply_style_texts_to_markdown(markdown: str, style_texts: Sequence[StyleAnchor]) -> str:
    """Apply style anchors to markdown by matching both the target text and its nearby context."""
    normalized_markdown, index_map = build_normalized_index_map(markdown)
    if not normalized_markdown:
        return markdown

    protected_ranges = _markdown_protected_ranges(markdown)
    occupied_ranges: List[Tuple[int, int]] = []
    range_styles: Dict[Tuple[int, int], set[str]] = {}
    for anchor in style_texts:
        candidate = _best_context_span(
            markdown=markdown,
            normalized_markdown=normalized_markdown,
            index_map=index_map,
            anchor=anchor,
            protected_ranges=protected_ranges,
            occupied_ranges=occupied_ranges,
        )
        if candidate is None:
            continue
        if candidate not in range_styles:
            range_styles[candidate] = set()
            occupied_ranges.append(candidate)
        range_styles[candidate].add(anchor.style)

    if not range_styles:
        return markdown

    enriched = markdown
    for start, end in sorted(range_styles.keys(), key=lambda item: item[0], reverse=True):
        marker = _marker_for_styles(sorted(range_styles[(start, end)]))
        if not marker:
            continue
        enriched = enriched[:start] + marker + enriched[start:end] + marker + enriched[end:]
    return enriched


def apply_bold_texts_to_markdown(markdown: str, bold_texts: Sequence[StyleAnchor]) -> str:
    """Apply bold anchors to markdown by matching both the target text and its nearby context."""
    return apply_style_texts_to_markdown(markdown, bold_texts)


def apply_italic_texts_to_markdown(markdown: str, italic_texts: Sequence[StyleAnchor]) -> str:
    """Apply italic anchors to markdown by matching both the target text and its nearby context."""
    return apply_style_texts_to_markdown(markdown, italic_texts)


def apply_pdf_styles_onepage(doc: fitz.Document, page_number: int, page_markdown: str) -> str:
    """Extract bold and italic anchors for one page and apply them to that page markdown."""
    style_texts: List[StyleAnchor] = []
    for style in (STYLE_BOLD, STYLE_ITALIC):
        try:
            style_texts.extend(extract_style_texts_from_page(doc, page_number, style))
        except Exception:
            continue
    if not style_texts:
        return page_markdown
    return apply_style_texts_to_markdown(page_markdown, style_texts)


def apply_pdf_style_marks(
    pdf_path: str,
    page_numbers: List[int],
    pages: List[str],
) -> List[str]:
    """Open the PDF once and apply bold/italic styling page-by-page to the corresponding markdown pages."""
    doc = fitz.open(pdf_path)
    try:
        return [
            apply_pdf_styles_onepage(doc, page_number, page_markdown)
            for page_number, page_markdown in zip(page_numbers, pages)
        ]
    finally:
        doc.close()
