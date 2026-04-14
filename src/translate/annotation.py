#!/usr/bin/env python3
"""
Helpers for generating short mathematical intuition annotations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from llm_util import create_chat_completion_with_retry

ANNOTATION_SYSTEM_PROMPT = """You are a mathematical exposition assistant.

Your task is to explain mathemtaical theorem, proposition, or definition in Simplified Chinese from an English math text.

Requirements:
1. goal is to help readers quickly grasp the core intuition and main idea behind the statement.
2. Focus on mathematical intuition, insight, geometric picture, motivation, or what the statement is really saying.
3. Prefer giving 1 to 2 of the following: what the statement means in plain language; why this concept/result is natural or useful; a typical situation where it is used.
4. Do NOT restate the formal definition or theorem in detail. The book already contains the rigorous statement.
5. Keep it short and sharp. No long derivations. Don't provide detailed proof, only a high-level sketch if necessary to convey the intuition.
6. If the source contains formulas, you may mention them briefly when needed, but prioritize conceptual explanation over symbolic restatement.
7. Use precise but accessible Chinese mathematical language for readers learning the material.
8. Output only the explanation body in Simplified Chinese, with no title and no surrounding Markdown block.
"""

ANNOTATION_HEADING_RE = re.compile(
    r"""
    ^\s*
    (?:[#>*-]+\s*)?
    (?:(?:\*\*|__)\s*)?
    (?:
        (?P<label>definition|theorem|proposition|lemma|corollary)
        \b
        (?:\s+\d+(?:\.\d+)*[a-zA-Z]?)?
        |
        \d+(?:\.\d+)*[a-zA-Z]?
        \s+
        (?P<label_after>definition|theorem|proposition|lemma|corollary)
        \b
    )
    (?:\s*[:.])?
    (?:(?:\*\*|__)\s*)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

ANNOTATION_CONTINUATION_RE = re.compile(
    r"""
    ^
    (?:
        \s{2,}\S
        |
        \t+\S
        |
        \s*(?:[-+*]|\d+[.)]|[a-zA-Z][.)]|\([a-zA-Z0-9]+\))\s+
    )
    """,
    re.VERBOSE,
)

ANNOTATION_STOP_RE = re.compile(
    r"""
    ^\s*
    (?:
        (?:[#>*-]+\s*)?
        (?:(?:\*\*|__)\s*)?
    )?
    (?:
        proof
        |remark
        |example
        |examples
        |note
        |notes
        |exercise
        |exercises
        |question
        |questions
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

ANNOTATION_PLAIN_CONTINUATION_RE = re.compile(
    r"""
    ^\s*
    (?:(?:\*\*|__)\s*)?
    (?:
        then
        |moreover
        |furthermore
        |in\ particular
        |in\ this\ case
        |more\ generally
        |equivalently
        |conversely
        |consequently
        |therefore
        |thus
        |hence
        |where
        |here
        |in\ other\ words
        |namely
        |that\ is
        |i\.e\.
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

ANNOTATION_EMPHATIC_ENUM_RE = re.compile(
    r"""
    ^\s*
    (?:
        >\s*
    )?
    (?:
        \*\*|__
    )
    \s*
    (?:
        [-+*]
        |\d+[.)]
        |[a-zA-Z][.)]
        |\([a-zA-Z0-9]+\)
    )
    \s*
    (?:
        \*\*|__
    )
    \s+
    """,
    re.VERBOSE,
)


@dataclass
class AnnotationService:
    client: Any
    model: str
    reasoning_effort: str = "none"
    enabled: bool = False

    def annotate(self, source_text: str) -> Optional[str]:
        if not self.enabled:
            return None
        content = self._call_annotate(source_text)
        if not content:
            return None
        return f"\n\n> **直观理解：** {content}\n"

    def _normalize_block_for_annotation(self, text: str) -> str:
        """
        去掉 blockquote / list / 粗体等常见 Markdown 包装，便于做 statement 规则判断。
        """
        normalized = text.strip()
        while True:
            updated = re.sub(r"^\s*>\s*", "", normalized)
            updated = re.sub(r"^(?:\*\*|__)(.+?)(?:\*\*|__)(?=\s|$)", r"\1", updated)
            updated = re.sub(r"^\s*(?:[-+*]|\d+[.)]|[a-zA-Z][.)]|\([a-zA-Z0-9]+\))\s+", "", updated)
            updated = re.sub(r"^(?:\*\*|__)\s*", "", updated)
            updated = re.sub(r"\s*(?:\*\*|__)$", "", updated)
            if updated == normalized:
                break
            normalized = updated.strip()
        return normalized

    def _should_annotate(self, current_block: str | None, next_block: str) -> bool:
        """
        if current_block is None, it means next_block is the first block of a section, so we should check if next_block
          starts with a definition/theorem/proposition/lemma/corollary heading.
        if current_block is not None, it means already decided current_block should be annotated, so we should check if
          next_block is a continuation of current_block (e.g. starts with a list item or is indented; or is a display latex formula),
          and if so, we should annotate next_block as well.
        """
        stripped = next_block.strip()
        normalized = self._normalize_block_for_annotation(next_block)
        if not stripped:
            return False

        if current_block is None:
            return bool(ANNOTATION_HEADING_RE.match(stripped) or ANNOTATION_HEADING_RE.match(normalized))

        if ANNOTATION_STOP_RE.match(stripped) or ANNOTATION_STOP_RE.match(normalized):
            return False

        if (
            stripped.startswith("$$")
            or stripped.startswith(r"\[")
            or stripped.startswith(r"\begin{")
            or normalized.startswith("$$")
            or normalized.startswith(r"\[")
            or normalized.startswith(r"\begin{")
        ):
            return True

        if ANNOTATION_EMPHATIC_ENUM_RE.match(stripped):
            return True

        if ANNOTATION_CONTINUATION_RE.match(next_block) or ANNOTATION_CONTINUATION_RE.match(normalized):
            return True

        return bool(ANNOTATION_PLAIN_CONTINUATION_RE.match(normalized))

    def _build_messages(self, source_text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Please explain the following mathematical statement briefly in Simplified Chinese, "
                    "emphasizing intuition, motivation, and practical meaning over formal restatement:\n\n"
                    f"{source_text.strip()}"
                ),
            },
        ]

    def _call_annotate(self, source_text: str) -> Optional[str]:
        result = create_chat_completion_with_retry(
            client=self.client,
            model=self.model,
            messages=self._build_messages(source_text),
            reasoning_effort=self.reasoning_effort,
            max_retries=2,
            error_label="annotate",
        ).strip()
        if not result or result.startswith("annotate error:"):
            return None
        return self._normalize_output(result)

    def _normalize_output(self, text: str) -> str:
        cleaned = re.sub(r"^\s*(?:直观理解|说明|注|注释)\s*[:：]\s*", "", text.strip())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned + "\n\n"
