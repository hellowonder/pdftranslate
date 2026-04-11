#!/usr/bin/env python3
"""
Helpers for resolving paragraph breaks across PDF page boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional, Sequence

from openai import OpenAI


@dataclass
class BoundaryDecision:
    page_number_left: int
    page_number_right: int
    decision: str
    left_block: str
    right_block: str


def split_markdown_blocks(text: str) -> List[str]:
    blocks = [block.strip() for block in (text or "").split("\n\n")]
    return [block for block in blocks if block]


def join_markdown_blocks(blocks: Sequence[str]) -> str:
    return "\n\n".join(block.strip() for block in blocks if block.strip())


SENTENCE_END_PUNCTUATION = set(".!?:;。！？：；")
OPENING_PUNCTUATION = tuple(")]},;:!?%)}]】》」』，。！？：；、")
LOWERCASE_CONNECTOR_RE = re.compile(r"^[a-z]")
TITLE_PREFIX_RE = re.compile(r"^(chapter|section|appendix|part)\b", re.IGNORECASE)
LIST_OR_HEADING_RE = re.compile(
    r"^(#{1,6}\s|[-*+]\s|\d+[.)]\s|\(?[ivxlcdm]+\)?[.)]\s|\(?[A-Za-z]\)?[.)]\s)",
    re.IGNORECASE,
)
FORMULA_LINE_RE = re.compile(r"^(\$\$|\\\[|\\begin\{)")


def _last_line(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _first_line(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[0] if lines else ""


def _last_non_space_char(text: str) -> str:
    stripped = (text or "").rstrip()
    return stripped[-1] if stripped else ""


def _first_non_space_char(text: str) -> str:
    stripped = (text or "").lstrip()
    return stripped[0] if stripped else ""


def _looks_like_markdown_heading(text: str) -> bool:
    first_line = _first_line(text)
    return bool(first_line and (first_line.startswith("#") or TITLE_PREFIX_RE.match(first_line)))


def _looks_like_markdown_heading_line(text: str) -> bool:
    stripped = (text or "").strip()
    return bool(stripped and stripped.startswith("#"))


def _looks_like_list_or_numbered_item(text: str) -> bool:
    first_line = _first_line(text)
    return bool(first_line and LIST_OR_HEADING_RE.match(first_line))


def _looks_like_special_block_start(text: str) -> bool:
    first_line = _first_line(text)
    normalized = first_line.lower()
    if not first_line:
        return False
    if first_line.startswith("![]("):
        return True
    if FORMULA_LINE_RE.match(first_line):
        return True
    if _looks_like_list_or_numbered_item(first_line):
        return True
    if _looks_like_markdown_heading(first_line):
        return True
    if normalized.startswith(("figure ", "fig. ", "fig ", "table ", "caption", "exercise ")):
        return True
    return False


def _looks_like_short_title(text: str) -> bool:
    if "\n" in (text or "").strip():
        return False
    stripped = (text or "").strip()
    if not stripped:
        return False
    words = stripped.split()
    if len(words) > 12:
        return False
    if _looks_like_markdown_heading(stripped):
        return True
    if TITLE_PREFIX_RE.match(stripped):
        return True
    if _last_non_space_char(stripped) in SENTENCE_END_PUNCTUATION:
        return False
    alpha_words = [word for word in words if any(ch.isalpha() for ch in word)]
    if not alpha_words:
        return False
    title_case_words = [word for word in alpha_words if word[:1].isupper()]
    return len(title_case_words) >= max(1, len(alpha_words) - 1)


def _ends_with_block_latex(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.endswith("$$"):
        return True
    if stripped.endswith(r"\]"):
        return True
    return bool(re.search(r"\\end\{[a-zA-Z*]+\}\s*$", stripped))


def _looks_joinable_boundary(left_block: str, right_block: str) -> bool:
    left_last_char = _last_non_space_char(left_block)
    right_first_char = _first_non_space_char(right_block)
    right_stripped = (right_block or "").lstrip()

    if left_last_char == "-":
        return True
    if left_last_char and left_last_char not in SENTENCE_END_PUNCTUATION:
        return True
    if right_first_char in OPENING_PUNCTUATION:
        return True
    if LOWERCASE_CONNECTOR_RE.match(right_stripped):
        return True
    if right_stripped.lower().startswith(
        (
            "and ",
            "or ",
            "but ",
            "nor ",
            "so ",
            "yet ",
            "because ",
            "however ",
            "therefore ",
            "thus ",
            "then ",
            "when ",
            "where ",
            "which ",
            "that ",
        )
    ):
        return True
    return False


def decide_page_boundary_merge(
    left_block: str,
    right_block: str,
) -> str:
    left_stripped = (left_block or "").strip()
    right_stripped = (right_block or "").strip()
    if not left_stripped or not right_stripped:
        return "SPLIT"

    if _looks_like_special_block_start(right_stripped):
        return "SPLIT"
    if _looks_like_markdown_heading_line(_last_line(left_stripped)):
        return "SPLIT"
    if _ends_with_block_latex(left_stripped):
        return "SPLIT"
    if _looks_like_short_title(left_stripped):
        return "SPLIT"

    left_last_char = _last_non_space_char(left_stripped)
    right_first_char = _first_non_space_char(right_stripped)

    if left_last_char in SENTENCE_END_PUNCTUATION and right_first_char and right_first_char.isupper():
        return "SPLIT"
    if left_last_char in SENTENCE_END_PUNCTUATION and right_first_char and right_first_char.isdigit():
        return "SPLIT"
    if _looks_joinable_boundary(left_stripped, right_stripped):
        return "JOIN"
    if left_last_char in SENTENCE_END_PUNCTUATION:
        return "SPLIT"
    return "JOIN"


def merge_cross_page_paragraphs(
    pages: Sequence[str],
    page_numbers: Optional[Sequence[int]] = None,
) -> tuple[List[str], List[BoundaryDecision]]:
    merged_pages = list(pages)
    decisions: List[BoundaryDecision] = []
    effective_page_numbers = list(page_numbers) if page_numbers is not None else list(range(1, len(pages) + 1))

    for idx in range(len(merged_pages) - 1):
        left_blocks = split_markdown_blocks(merged_pages[idx])
        right_blocks = split_markdown_blocks(merged_pages[idx + 1])
        if not left_blocks or not right_blocks:
            continue

        left_block = left_blocks[-1]
        right_block = right_blocks[0]
        decision = decide_page_boundary_merge(
            left_block=left_block,
            right_block=right_block,
        )
        decisions.append(
            BoundaryDecision(
                page_number_left=effective_page_numbers[idx],
                page_number_right=effective_page_numbers[idx + 1],
                decision=decision,
                left_block=left_block,
                right_block=right_block,
            )
        )
        if decision != "JOIN":
            continue

        left_blocks[-1] = f"{left_blocks[-1].rstrip()}\n{right_blocks[0].lstrip()}"
        del right_blocks[0]
        merged_pages[idx] = join_markdown_blocks(left_blocks)
        merged_pages[idx + 1] = join_markdown_blocks(right_blocks)

    return merged_pages, decisions


def format_boundary_decisions(decisions: Sequence[BoundaryDecision]) -> str:
    lines: List[str] = []
    for item in decisions:
        lines.append(f"{item.page_number_left:04d}->{item.page_number_right:04d}: {item.decision}")
    return "\n".join(lines)
