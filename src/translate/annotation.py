#!/usr/bin/env python3
"""
Helpers for generating short mathematical intuition annotations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from llm_util import create_chat_completion_with_retry

ITEM_ANNOTATION_SYSTEM_PROMPT = """You are a mathematical exposition assistant.

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

PAGE_ANNOTATION_SYSTEM_PROMPT = """You are a mathematical exposition assistant.

Your task is to explain one page of an English mathematical text in Simplified Chinese for a reader who wants intuition first.

Requirements:
1. Treat the page as a coherent whole. Explain the page's main theme, what problem or idea it is trying to develop, and how the parts fit together.
2. Strongly emphasize intuition and motivation: what the author is trying to achieve, why the concepts or statements on this page are natural, and what a reader should keep in mind conceptually.
3. If the page moves too quickly or leaves details implicit, fill in the missing explanatory bridge in a concise but helpful way.
4. Prefer plain but precise Chinese mathematical language. You may mention formulas briefly, but only to support conceptual understanding.
5. Do NOT translate the page line by line. Do NOT mechanically restate every sentence. Do NOT give a full proof.
6. Expand on ideas that are under-explained in the source, especially hidden motivation, geometric meaning, standard examples, or why a definition/result matters.
7. Keep the explanation focused and readable. A short multi-sentence note is better than a long exhaustive commentary.
8. Output only the explanation body in Simplified Chinese, with no title and no surrounding Markdown block.
"""

ANNOTATION_SYSTEM_PROMPT = ITEM_ANNOTATION_SYSTEM_PROMPT

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
    mode: str = "item"

    def annotate(self, source_text: str) -> Optional[str]:
        if not self.enabled:
            return None
        content = self._call_annotate(source_text)
        if not content:
            return None

        stripped = content.rstrip("\n")
        lines = stripped.splitlines()
        if not lines:
            return None
        quoted_lines = [f"> **导读：** {lines[0]}"]
        quoted_lines.extend(">" if not line.strip() else f"> {line}" for line in lines[1:])
        return "\n\n" + "\n".join(quoted_lines) + "\n"


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
        if self.mode == "page":
            system_prompt = PAGE_ANNOTATION_SYSTEM_PROMPT
            user_prompt = (
                "Please explain the following page in Simplified Chinese. Focus on intuition, motivation, "
                "and the page's main idea. Expand the under-explained parts instead of restating the page formally:\n\n"
                f"{source_text.strip()}"
            )
        else:
            system_prompt = ITEM_ANNOTATION_SYSTEM_PROMPT
            user_prompt = (
                "Please explain the following mathematical statement briefly in Simplified Chinese, "
                "emphasizing intuition, motivation, and practical meaning over formal restatement:\n\n"
                f"{source_text.strip()}"
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
        if self.mode == "page":
            paragraphs = [re.sub(r"[ \t]+", " ", part).strip() for part in re.split(r"\n\s*\n", cleaned)]
            paragraphs = [part for part in paragraphs if part]
            cleaned = "\n\n".join(paragraphs).strip()
        else:
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned + "\n\n"
