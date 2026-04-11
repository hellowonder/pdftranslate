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
2. Focus on mathematical insight, intuition, geometric picture, motivation, or what the statement is really saying.
3. Do NOT restate the formal definition in detail. The book already contains the rigorous statement.
4. Keep it simple to understand, no long derivations. Don't provide detailed proof, only a high-level sketch if necessary to convey the intuition.
5. If the source contains formulas, you may mention them briefly when needed, but prioritize conceptual explanation.
6. Use precise but accessible Chinese mathematical language.
7. Output only the explanation body in Simplified Chinese, with no title and no surrounding Markdown block.
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

    def _should_annotate(self, current_block: str | None, next_block: str) -> bool:
        """
        if current_block is None, it means next_block is the first block of a section, so we should check if next_block
          starts with a definition/theorem/proposition/lemma/corollary heading.
        if current_block is not None, it means already decided current_block should be annotated, so we should check if
          next_block is a continuation of current_block (e.g. starts with a list item or is indented; or is a display latex formula),
          and if so, we should annotate next_block as well.
        """
        stripped = next_block.strip()
        if not stripped:
            return False

        if current_block is None:
            return bool(ANNOTATION_HEADING_RE.match(stripped))

        if stripped.startswith("$$") or stripped.startswith(r"\[") or stripped.startswith(r"\begin{"):
            return True

        return bool(ANNOTATION_CONTINUATION_RE.match(next_block))

    def _build_messages(self, source_text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Please explain the following mathematical statement briefly in Simplified Chinese, "
                    "emphasizing intuition over formal restatement:\n\n"
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
