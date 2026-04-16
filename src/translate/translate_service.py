#!/usr/bin/env python3
"""
Shared helpers used by different translation entry points.
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterator, List, Literal, Optional, Sequence, Tuple

from tqdm import tqdm
from llm_util import (
    configure_openai,
    create_chat_completion_with_retry,
    has_low_diversity_or_repetition,
)
from annotation import ANNOTATION_SYSTEM_PROMPT, AnnotationService
from translate_latex import extract_latex_segments, repair_translation_latex

TranslationMode = Literal["markdown", "html", "plain_text"]
LatexFormulaHandlingMode = Literal["placeholder", "direct"]
ReasoningEffort = Literal["none", "low", "medium", "high"]

DEFAULT_MARKDOWN_TRANSLATION_SYSTEM_PROMPT = (
r"""You are a professional English to Chinese translator for mathematical Markdown documents.

Translate the input into Simplified Chinese while strictly preserving structure and formulas.

Rules:
1. Keep all Markdown structure unchanged (headings, lists, bold/italic, links, tables, etc.).
2. Translate only natural language text.
3. Do NOT modify any placeholders or protected content:
   - Tokens like [FORMULA_1], [FORMULA_2] must remain exactly unchanged.
   - Number, order, and position of these tokens must stay the same.
4. Do NOT modify any LaTeX formulas when they appear directly:
   - Content inside $...$, $$...$$, \(...\), \[...\] must remain exactly unchanged.
   - Number, order, and position of formulas must stay the same.
5. Do NOT modify code blocks or inline code.
6. Preserve blank lines, paragraph breaks, and line boundaries exactly.
7. Do NOT add explanations, comments, or extra content.
8. Use accurate and standard Chinese mathematical terminology.

Output only the translated Markdown.
"""
)

DIRECT_MARKDOWN_TRANSLATION_SYSTEM_PROMPT = (
r"""You are a professional English to Chinese translator for mathematical Markdown documents.

Translate the input into Simplified Chinese while strictly preserving structure and formulas.

Rules:
1. Keep all Markdown structure unchanged (headings, lists, bold/italic, links, tables, etc.).
2. Translate only natural language text.
3. Do NOT modify any LaTeX formulas when they appear directly:
   - Content inside $...$, $$...$$, \(...\), \[...\], and \begin{...}...\end{...} must remain exactly unchanged.
   - Number, order, and position of formulas must stay the same.
4. Do NOT modify code blocks or inline code.
5. Preserve blank lines, paragraph breaks, and line boundaries exactly.
6. Do NOT add explanations, comments, or extra content.
7. Use accurate and standard Chinese mathematical terminology.

Output only the translated Markdown.
"""
)

DEFAULT_HTML_TRANSLATION_SYSTEM_PROMPT = (
r"""You are a professional English to Chinese translator for mathematical HTML documents.

Translate the input into Simplified Chinese while strictly preserving structure and formulas.

Rules:
1. Keep all HTML tags, attributes, nesting, and entity structure unchanged.
2. Translate only natural language text content.
3. Do NOT modify any placeholders or protected content:
   - Tokens like [FORMULA_1], [FORMULA_2] must remain exactly unchanged.
   - Number, order, and position of these tokens must stay the same.
4. Do NOT modify any LaTeX formulas when they appear directly:
   - Content inside $...$, $$...$$, \(...\), \[...\] must remain exactly unchanged.
   - Number, order, and position of formulas must stay the same.
5. Do NOT modify code blocks, inline code, URLs, or attribute values.
6. Do NOT add explanations, comments, or extra wrapper elements.
7. Use accurate and standard Chinese mathematical terminology.

Output only the translated HTML.
"""
)

DIRECT_HTML_TRANSLATION_SYSTEM_PROMPT = (
r"""You are a professional English to Chinese translator for mathematical HTML documents.

Translate the input into Simplified Chinese while strictly preserving structure and formulas.

Rules:
1. Keep all HTML tags, attributes, nesting, and entity structure unchanged.
2. Translate only natural language text content.
3. Do NOT modify any LaTeX formulas when they appear directly:
   - Content inside $...$, $$...$$, \(...\), \[...\], and \begin{...}...\end{...} must remain exactly unchanged.
   - Number, order, and position of formulas must stay the same.
4. Do NOT modify code blocks, inline code, URLs, or attribute values.
5. Do NOT add explanations, comments, or extra wrapper elements.
6. Use accurate and standard Chinese mathematical terminology.

Output only the translated HTML.
"""
)

DEFAULT_PLAIN_TEXT_TRANSLATION_SYSTEM_PROMPT = (
r"""You are a professional English to Chinese translator for short plain-text content.

Translate the input into natural, fluent Simplified Chinese.

Rules:
1. Translate only the visible natural language content.
2. Preserve meaning, tone, and proper nouns accurately.
3. Do NOT add explanations, notes, quotes, prefixes, or suffixes.
4. Keep the output concise and faithful to the source.

Output only the translated text.
"""
)
DEFAULT_TRANSLATION_MAX_CHUNK_CHARS = 1200


@dataclass(frozen=True)
class TranslationBlock:
    text: str
    protected: bool
    is_annotation: bool = False

@dataclass
class TranslationService:
    client: Any
    model: str
    temperature: float
    max_chunk_chars: int = DEFAULT_TRANSLATION_MAX_CHUNK_CHARS
    latex_formula_handling: LatexFormulaHandlingMode = "placeholder"
    reasoning_effort: ReasoningEffort = "none"
    _annotation_service: Optional[AnnotationService] = None
    _do_latex_repair: bool = True

    _SHORT_TRANSLATION_USER_CONTENT_MAX_CHARS = 20

    def _build_messages(
        self,
        user_content: str,
        mode: TranslationMode = "markdown",
    ) -> List[dict[str, str]]:
        """
        构造发给聊天模型的消息列表。

        参数:
            user_content: 当前待翻译文本，作为 user 消息内容。
            mode: 翻译模式；内部根据模式选择不同 prompt。
        返回:
            List[dict[str, str]]: 传给聊天补全接口的消息序列。
        """
        system_content = self._get_system_prompt(mode)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": self._format_translation_user_content(user_content, mode)},
        ]

    def _format_translation_user_content(self, user_content: str, mode: TranslationMode) -> str:
        """
        为极短输入补充最小任务说明，避免模型把单词或短语当成普通对话内容。
        """
        stripped = user_content.strip()
        if (
            not stripped
            or "\n" in stripped
            or len(stripped) > self._SHORT_TRANSLATION_USER_CONTENT_MAX_CHARS
        ):
            return user_content

        if mode == "html":
            task = "Translate the following HTML into Simplified Chinese."
            output = "Return only the translated HTML."
        elif mode == "plain_text":
            task = "Translate the following text into Simplified Chinese."
            output = "Return only the translation."
        else:
            task = "Translate the following Markdown into Simplified Chinese."
            output = "Return only the translated Markdown."

        return f"{task}\n{output}\n\n<BEGIN_SOURCE>\n{user_content}\n<END_SOURCE>"

    def _get_system_prompt(self, mode: TranslationMode) -> str:
        if mode == "html":
            if self.latex_formula_handling == "direct":
                return DIRECT_HTML_TRANSLATION_SYSTEM_PROMPT
            return DEFAULT_HTML_TRANSLATION_SYSTEM_PROMPT
        if mode == "plain_text":
            return DEFAULT_PLAIN_TEXT_TRANSLATION_SYSTEM_PROMPT
        if self.latex_formula_handling == "direct":
            return DIRECT_MARKDOWN_TRANSLATION_SYSTEM_PROMPT
        return DEFAULT_MARKDOWN_TRANSLATION_SYSTEM_PROMPT

    def _need_translate(self, text: str) -> bool:
        """
        判断文本是否需要翻译。
        """
        cleaned = self._strip_non_translatable_for_need_translate(text)
        if not cleaned.strip():
            return False
        if all(ch in "\u0000-\u007f" for ch in cleaned):
            return False
        if all(
            ch.isdigit() or ch.isspace() or ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
            for ch in cleaned
        ):
            return False
        return True

    def _strip_non_translatable_for_need_translate(self, text: str) -> str:
        """
        移除不应参与“是否需要翻译”判定的结构性内容。
        """
        cleaned = text
        latex_segments = extract_latex_segments(cleaned)
        if latex_segments:
            parts: List[str] = []
            cursor = 0
            for segment in latex_segments:
                parts.append(cleaned[cursor:segment.start])
                cursor = segment.end
            parts.append(cleaned[cursor:])
            cleaned = "".join(parts)
        cleaned = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', cleaned)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        return cleaned
            
    def _translate_with_retry(self, source_text: str, messages: Sequence[dict[str, str]]) -> str:
        """
        调用翻译模型并在结果可疑或 LaTeX 校验失败时自动重试。

        参数:
            source_text: 原始待翻译文本，用于后续校验和调试输出。
            messages: 已构造好的聊天消息序列。
        返回:
            str: 最终接受的译文；如果多次重试仍失败，则返回最后一次结果。
        """
        leading, core_text, trailing = self._split_outer_whitespace(source_text)
        if not core_text:
            return source_text
        if not self._need_translate(core_text):
            return source_text
        
        if self.latex_formula_handling == "placeholder":
            request_text, formula_map = self._protect_latex(core_text)
        else:
            request_text, formula_map = core_text, []
        request_messages = self._replace_last_user_message(messages, request_text)
        max_retries = 3
        content = ""
        for attempt in range(1, max_retries + 1):
            result = create_chat_completion_with_retry(
                client=self.client,
                model=self.model,
                messages=request_messages,
                reasoning_effort=self.reasoning_effort,
                max_retries=1,
                error_label="translate",
            )
            if self._looks_suspicious_translation(request_text, result):
                print(
                    f"suspicious translation output (attempt {attempt}/{max_retries}); retrying...\n",
                    f"from: {request_text!r}\n",
                    f">>: {result!r}\n",
                    file=sys.stderr,
                )
                continue
            else:
                content = result
                break

        if not content.strip():
            # translation failed with retry, avoid further processing and return original text
            return source_text

        if self.latex_formula_handling == "placeholder":
            content, latex_ok = self._restore_latex(content, formula_map)
        else:
            content, latex_ok = self._repair_translation_latex(core_text, content)

        return self._restore_outer_whitespace(leading, trailing, content)

    def _protect_latex(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        将文本中的 LaTeX 公式替换成占位符，避免模型改写公式内容。

        返回:
            Tuple[str, List[Tuple[str, str]]]:
                替换后的文本，以及按顺序记录的 ``(placeholder, formula)`` 映射。
        """
        segments = extract_latex_segments(text)
        if not segments:
            return text, []

        parts: List[str] = []
        formula_map: List[Tuple[str, str]] = []
        cursor = 0
        for idx, segment in enumerate(segments, start=1):
            placeholder = f"[FORMULA_{idx}]"
            parts.append(text[cursor:segment.start])
            parts.append(placeholder)
            formula_map.append((placeholder, segment.text))
            cursor = segment.end
        parts.append(text[cursor:])
        return "".join(parts), formula_map

    def _restore_latex(self, text: str, formula_map: Sequence[Tuple[str, str]]) -> Tuple[str, bool]:
        """
        将占位符替换回原始 LaTeX 公式。

        若占位符数量或内容不匹配，则返回失败，让上层触发重试。
        """
        if not formula_map:
            return text, True

        restored = text
        missing_placeholders = []
        for placeholder, formula in formula_map:
            if placeholder not in restored:
                missing_placeholders.append((placeholder, formula))
                continue
            restored = restored.replace(placeholder, formula)

        if missing_placeholders:
            print(
                f"missing formula placeholders in translation: {missing_placeholders}",
                file=sys.stderr,
            )

        if re.search(r"\[FORMULA_\d+\]", restored):
            print("unexpected formula placeholders remained after restore", file=sys.stderr)
            return restored, False

        restored_segments = extract_latex_segments(restored)
        if restored_segments:
            source_nospace = {
                item.nospace
                for formula in (formula for _, formula in formula_map)
                for item in extract_latex_segments(formula)
            }
            restored_nospace = [segment.nospace for segment in restored_segments]
            if len(restored_segments) != len(formula_map) or any(
                item not in source_nospace for item in restored_nospace
            ):
                print("translation introduced unexpected latex formulas after placeholder restore", file=sys.stderr)
                return restored, False

        return restored, True

    def _repair_translation_latex(self, source: str, translation: str) -> Tuple[str, bool]:
        if self._do_latex_repair:
            return repair_translation_latex(source, translation, debug_stream=sys.stderr)
        else:
            return translation, True

    def _split_outer_whitespace(self, text: str) -> Tuple[str, str, str]:
        """
        拆分文本两端空白与中间正文，便于将空白留在翻译服务之外处理。

        参数:
            text: 原始待翻译文本。
        返回:
            Tuple[str, str, str]: ``(leading, core, trailing)``。
        """
        leading_match = re.match(r"^\s*", text)
        trailing_match = re.search(r"\s*$", text)
        leading = leading_match.group(0) if leading_match else ""
        trailing = trailing_match.group(0) if trailing_match else ""
        core_end = len(text) - len(trailing)
        core = text[len(leading):core_end]
        return leading, core, trailing

    def _replace_last_user_message(
        self,
        messages: Sequence[dict[str, str]],
        user_content: str,
    ) -> List[dict[str, str]]:
        """
        用去掉首尾空白的正文替换最后一个 user 消息内容。
        """
        copied = [dict(message) for message in messages]
        for idx in range(len(copied) - 1, -1, -1):
            if copied[idx].get("role") == "user":
                copied[idx]["content"] = self._replace_user_content_preserving_wrapper(
                    copied[idx].get("content", ""),
                    user_content,
                )
                return copied
        copied.append({"role": "user", "content": user_content})
        return copied

    def _replace_user_content_preserving_wrapper(self, original_content: str, user_content: str) -> str:
        marker_start = "<BEGIN_SOURCE>\n"
        marker_end = "\n<END_SOURCE>"
        if marker_start in original_content and marker_end in original_content:
            prefix, suffix = original_content.split(marker_start, 1)
            original_source, trailing = suffix.split(marker_end, 1)
            del original_source
            return f"{prefix}{marker_start}{user_content}{marker_end}{trailing}"
        return user_content

    def _restore_outer_whitespace(self, leading: str, trailing: str, translated_text: str) -> str:
        """
        恢复源文本首尾空白，避免模型吞掉段落边界或块间换行。

        参数:
            leading: 原文前导空白。
            trailing: 原文尾随空白。
            translated_text: 模型返回并经过后处理的译文。
        返回:
            str: 恢复了源文本首尾空白后的结果。
        """
        core = translated_text.strip()
        if not core:
            return f"{leading}{trailing}"
        return f"{leading}{core}{trailing}"

    def translate_text_block(self, text: str, mode: TranslationMode = "markdown") -> str:
        """
        翻译单个文本块，并对代码块或数学块做保护处理。

        参数:
            text: 输入的 Markdown 或纯文本内容。
            mode: 翻译模式。
        返回:
            str: 合并后的译文文本。
        """
        if not text:
            return ""
        output_parts: List[str] = []
        for block in self._iter_translation_blocks(text):
            if not block.text:
                continue
            if block.is_annotation:
                if not self._annotation_service:
                    continue
                annotation = self._annotation_service.annotate(block.text)
                if annotation:
                    output_parts.append(annotation)
                continue
            if block.protected:
                output_parts.append(block.text)
                continue
            output_parts.append(
                self._translate_with_retry(
                    block.text,
                    self._build_messages(block.text, mode=mode),
                )
            )
        return "".join(output_parts)

    def _looks_suspicious_translation(self, source: str, translation: str) -> bool:
        """
        用启发式规则判断译文是否明显异常。

        参数:
            source: 原文文本。
            translation: 模型返回的译文文本。
        返回:
            bool: 如果结果为空、过长、重复严重或疑似未翻译，则返回 ``True``。
        """
        if not translation.strip():
            return True

        max_reasonable_len = len(source) * 6
        if len(translation) > max_reasonable_len:
            return True

        if has_low_diversity_or_repetition(translation):
            return True

        if max(len(source), len(translation)) < 100:
            return False

        if self._looks_untranslated(source, translation):
            return True

        return False

    def _looks_untranslated(self, source: str, translation: str) -> bool:
        """
        判断译文是否可能基本没有被翻译。

        参数:
            source: 原文文本。
            translation: 译文文本。
        返回:
            bool: 若中文含量过低且与原文高度相似，则返回 ``True``。
        """
        chinese_chars = sum(1 for ch in translation if "\u4e00" <= ch <= "\u9fff")
        chinese_threshold = max(10, int(len(translation) * 0.1))
        if chinese_chars >= chinese_threshold:
            return False

        source_clean = self._strip_noise_for_similarity(source)
        translation_clean = self._strip_noise_for_similarity(translation)
        if not source_clean or not translation_clean or len(translation_clean) < 20:
            return False

        if self._looks_like_bibliography_block(source_clean) and self._looks_like_bibliography_block(
            translation_clean
        ):
            return False

        matcher = difflib.SequenceMatcher(
            None, source_clean.lower(), translation_clean.lower()
        )
        return matcher.quick_ratio() > 0.9

    def _strip_noise_for_similarity(self, text: str) -> str:
        """
        去掉代码块、LaTeX、HTML 标签和多余空白，便于相似度比较。

        这里按 LaTeX span 一次性重建文本，避免对长文本做多次整串 replace。
        """
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        segments = extract_latex_segments(text)
        if segments:
            parts: List[str] = []
            cursor = 0
            for segment in segments:
                parts.append(text[cursor:segment.start])
                cursor = segment.end
            parts.append(text[cursor:])
            text = "".join(parts)
        text = re.sub(r"<[^>]+>", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _looks_like_bibliography_block(self, text: str) -> bool:
        """
        判断文本是否整体上像参考文献/引用条目块。

        参数:
            text: 已清洗的文本。
        返回:
            bool: 若文本看起来像参考文献列表中的条目，则返回 ``True``。
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False

        matching_lines = sum(1 for line in lines if self._looks_like_bibliography_line(line))
        if matching_lines == len(lines):
            return True

        return matching_lines >= max(1, len(lines) - 1)

    def _looks_like_bibliography_line(self, text: str) -> bool:
        """
        判断单行文本是否像参考文献条目。

        参数:
            text: 单行文本。
        返回:
            bool: 若文本具备参考文献常见特征，则返回 ``True``。
        """
        line = re.sub(r"\s+", " ", text).strip()
        if len(line) < 40:
            return False

        starts_with_marker = bool(re.match(r"^(?:\[\d+\]|\(\d+\)|\d+\.\s+)", line))
        has_year = bool(re.search(r"\b(?:1[89]\d{2}|20\d{2})\b", line))
        has_author_pattern = bool(
            re.search(r"(?:^|[\]\s])(?:[A-Z][A-Z'`\-]+,\s*(?:[A-Z]\.|[A-Z][a-z]+))", line)
        )
        has_reference_separator = bool(re.search(r"[A-Z][A-Z'`\-]+,\s*(?:[A-Z]\.|[A-Z][a-z]+)\s*:", line))
        has_journalish_marker = bool(
            re.search(r"\b(?:vol\.?|no\.?|pp\.?|eds?\.?|trans\.?|proc\.?|journal|press)\b", line, flags=re.IGNORECASE)
        )
        has_doi_or_url = bool(re.search(r"\bdoi\b|https?://|www\.", line, flags=re.IGNORECASE))
        has_page_range = bool(re.search(r"\bpp?\.?\s*\d+(?:\s*[-–]\s*\d+)?\b|\b\d+\s*[-–]\s*\d+\b", line, flags=re.IGNORECASE))
        has_title_quotes = bool(re.search(r"[\"'“”‘’].{8,}[\"'“”‘’]", line))
        punctuation_count = sum(1 for ch in line if ch in ",;:.")
        score = sum(
            [
                starts_with_marker,
                has_year,
                has_author_pattern,
                has_reference_separator,
                has_journalish_marker,
                has_doi_or_url,
                has_page_range,
                has_title_quotes,
                punctuation_count >= 3,
            ]
        )

        if starts_with_marker and (has_year or has_author_pattern or has_doi_or_url):
            return True

        if has_year and (has_author_pattern or has_reference_separator):
            return True

        if has_doi_or_url and (has_year or has_author_pattern or has_journalish_marker):
            return True

        return score >= 4

    def _iter_translation_blocks(self, text: str) -> Iterator[TranslationBlock]:
        """
        生成翻译流水线消费的块流。

        普通内容先按段级基础块切分；如果普通块过长，再做二次拆分。
        annotation 块不是原文中的基础块，而是在识别出完整 statement 后额外插入。

        参数:
            text: 原始输入文本。
        返回:
            Iterator[TranslationBlock]: 按输出顺序生成的块序列。
        """
        raw_blocks = list(self._iter_base_markdown_blocks(text))
        if not raw_blocks:
            yield TranslationBlock(text=text, protected=False)
            return

        annotation_parts: List[str] = []
        annotation_active = False
        annotation_service = self._annotation_service

        for index, block in enumerate(raw_blocks):
            if block.protected or len(block.text) <= self.max_chunk_chars:
                yield block
            else:
                yield from self._iter_plain_translation_chunks(block.text)

            if not annotation_service:
                continue

            should_extend = annotation_service._should_annotate(
                "".join(annotation_parts) if annotation_active else None,
                block.text,
            )
            if should_extend:
                annotation_parts.append(block.text)
                annotation_active = True

            next_block = raw_blocks[index + 1] if index + 1 < len(raw_blocks) else None
            if (
                annotation_active
                and (
                    next_block is None
                    or not annotation_service._should_annotate("".join(annotation_parts), next_block.text)
                )
            ):
                yield TranslationBlock(
                    text="".join(annotation_parts),
                    protected=False,
                    is_annotation=True,
                )
                annotation_parts = []
                annotation_active = False

    def _iter_base_markdown_blocks(self, text: str) -> Iterator[TranslationBlock]:
        """
        按段级边界切分 Markdown 基础块。

        普通文本默认按空行分段；受保护块（代码块、display math、LaTeX 环境）
        会被整体保留，不会在内部继续拆开。
        """
        lines = text.splitlines(keepends=True)
        if not lines:
            yield TranslationBlock(text=text, protected=False)
            return

        current: List[str] = []
        current_protected = False
        block_mode: Optional[Tuple[str, Optional[str]]] = None

        def flush() -> Optional[TranslationBlock]:
            """
            将当前缓冲区写入结果列表。

            参数:
                无。
            返回:
                None: 仅更新闭包中的状态。
            """
            nonlocal current, current_protected
            if current:
                yield_block = TranslationBlock(text="".join(current), protected=current_protected)
                current = []
                current_protected = False
                return yield_block
            return None

        for line in lines:
            stripped = line.strip()

            if block_mode is None:
                block_mode = self._detect_protected_block_mode(stripped) if not current else None
                if block_mode is not None:
                    current = [line]
                    current_protected = True
                    if self._is_single_line_protected_block(stripped, block_mode):
                        block = flush()
                        if block:
                            yield block
                        block_mode = None
                    continue

            current.append(line)
            if block_mode is not None:
                mode_kind, env_name = block_mode
                if (
                    (mode_kind == "fence" and stripped.startswith("```"))
                    or (mode_kind == "display_dollar" and "$$" in stripped)
                    or (mode_kind == "display_bracket" and stripped.endswith(r"\]"))
                    or (mode_kind == "environment" and env_name and stripped.startswith(f"\\end{{{env_name}}}"))
                ):
                    block = flush()
                    if block:
                        yield block
                    block_mode = None
                continue

            if not stripped:
                block = flush()
                if block:
                    yield block

        block = flush()
        if block:
            yield block

    def _iter_plain_translation_chunks(self, text: str) -> Iterator[TranslationBlock]:
        """
        将过长的普通文本基础块继续切成多个翻译请求片段。

        参数:
            text: 单个段级普通文本块。
        返回:
            Iterator[TranslationBlock]: 切分后的普通翻译块。
        """
        if len(text) <= self.max_chunk_chars:
            yield TranslationBlock(text=text, protected=False)
            return

        remaining = text
        while len(remaining) > self.max_chunk_chars:
            split_at = self._find_split_index(remaining)
            if split_at <= 0:
                yield TranslationBlock(text=text, protected=False)
                return
            yield TranslationBlock(text=remaining[:split_at], protected=False)
            remaining = remaining[split_at:]
        if remaining:
            yield TranslationBlock(text=remaining, protected=False)

    def _find_split_index(self, text: str) -> int:
        """
        为长文本寻找一个合适的切分位置。

        参数:
            text: 待切分文本。
        返回:
            int: 推荐的切分下标；找不到时返回 ``0``。
        """
        limit = min(len(text), self.max_chunk_chars)
        for separator in ("\n\n", "\n"):
            idx = text.rfind(separator, 0, limit)
            if idx >= 0:
                return idx + len(separator)
        return 0

    def _detect_protected_block_mode(self, stripped: str) -> Optional[Tuple[str, Optional[str]]]:
        """
        判断当前行是否开启了一个受保护块。

        参数:
            stripped: 去除首尾空白后的当前行文本。
        返回:
            Optional[Tuple[str, Optional[str]]]:
                若识别到受保护块，返回块类型及可选环境名；否则返回 ``None``。
        """
        if stripped.startswith("```"):
            return ("fence", None)
        if stripped.startswith("$$"):
            return ("display_dollar", None)
        if stripped.startswith(r"\["):
            return ("display_bracket", None)
        env_match = re.match(r"\\begin\{([a-zA-Z*]+)\}", stripped)
        if env_match:
            return ("environment", env_match.group(1))
        return None

    def _is_single_line_protected_block(self, stripped: str, mode: Tuple[str, Optional[str]]) -> bool:
        """
        判断受保护块是否在当前行内就已经闭合。

        参数:
            stripped: 去除首尾空白后的当前行文本。
            mode: 当前块模式，包含块类型和可选环境名。
        返回:
            bool: 若该行同时是开始和结束，则返回 ``True``。
        """
        mode_kind, env_name = mode
        if mode_kind == "fence":
            return stripped.count("```") >= 2
        if mode_kind == "display_dollar":
            return stripped.count("$$") >= 2
        if mode_kind == "display_bracket":
            return stripped.startswith(r"\[") and stripped.endswith(r"\]") and len(stripped) > 2
        if mode_kind == "environment" and env_name:
            return f"\\end{{{env_name}}}" in stripped
        return False

    def translate_pages(
        self,
        pages: Sequence[str],
        max_workers: int = 1,
        mode: TranslationMode = "markdown",
    ) -> List[str]:
        """
        批量翻译多个页面，并支持按页并发处理。

        参数:
            pages: 页面文本序列。
            max_workers: 最大并发工作线程数。
            mode: 翻译模式。
        返回:
            List[str]: 与输入页顺序一致的译文页列表。
        """
        # if True:
        #     return ["<p>test</p>"] * len(pages)
        total_pages = len(pages)
        results: List[str] = [""] * total_pages

        def _translate(idx_page: Tuple[int, str]) -> Tuple[int, str]:
            """
            翻译单个页面并保留原始页索引。

            参数:
                idx_page: ``(页索引, 页面文本)`` 元组。
            返回:
                Tuple[int, str]: ``(页索引, 译文文本)``。
            """
            idx, page_text = idx_page
            translated = self.translate_text_block(page_text, mode=mode)
            return idx, translated

        max_workers = min(max_workers, len(pages))
        if max_workers <= 1:
            for idx, page_text in enumerate(pages):
                translated = self.translate_text_block(page_text, mode=mode)
                results[idx] = translated
                print(f"Translated page {idx + 1}/{total_pages}", file=sys.stderr)
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            print(f"parallel translate: {max_workers}...")
            for idx_out, translated in tqdm(executor.map(_translate, enumerate(pages))):
                results[idx_out] = translated
        return results


def add_translation_arguments(parser: argparse.ArgumentParser) -> None:
    """
    为命令行解析器注册翻译相关的公共参数。

    参数:
        parser: 需要追加参数的 ``ArgumentParser`` 实例。
    返回:
        None: 直接在传入的解析器对象上添加参数。
    """
    parser.add_argument(
        "--translation-base-url",
        default="http://localhost:11434/v1",
        help="OpenAI-compatible translation base URL. Defaults to the local Ollama endpoint.",
    )
    parser.add_argument(
        "--translation-api-key",
        default=os.environ.get("OLLAMA_API_KEY", "ollama"),
        help="API key for the translation endpoint. Defaults to the local Ollama convention.",
    )
    parser.add_argument(
        "--translation-model",
        default="gemma4:26b",
        help="Model name exposed by the translation endpoint. Defaults to the local Ollama model gemma4:26b.",
    )
    parser.add_argument(
        "--translation-reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="none",
        help="Best-effort reasoning/thinking level requested from the translation backend.",
    )
    parser.add_argument(
        "--translation-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for translation.",
    )
    parser.add_argument(
        "--translation-workers",
        type=int,
        default=16,
        help="Number of parallel translation workers.",
    )
    parser.add_argument(
        "--translation-max-chunk-chars",
        type=int,
        default=DEFAULT_TRANSLATION_MAX_CHUNK_CHARS,
        help="Maximum source characters per translation request.",
    )
    parser.add_argument(
        "--translation-latex-formula-handling",
        choices=["placeholder", "direct"],
        default="direct",
        help="How to handle LaTeX formulas during translation: replace with placeholders or send directly to the model.",
    )
    parser.add_argument(
        "--enable-annotation",
        action="store_true",
        help="Generate short intuition annotations for theorem/proposition/definition blocks and insert them into translated output.",
    )
    parser.add_argument(
        "--annotation-base-url",
        help="Annotation OpenAI-compatible base URL. Defaults to --translation-base-url.",
    )
    parser.add_argument(
        "--annotation-api-key",
        help="API key for the annotation endpoint. Defaults to --translation-api-key.",
    )
    parser.add_argument(
        "--annotation-model",
        help="Model name used for annotation generation. Defaults to --translation-model.",
    )
    parser.add_argument(
        "--annotation-reasoning-effort",
        choices=["none", "low", "medium", "high"],
        help="Best-effort reasoning/thinking level requested from the annotation backend. Defaults to --translation-reasoning-effort.",
    )


def _resolve_annotation_args(args: argparse.Namespace) -> tuple[str, str, str, str]:
    base_url = args.annotation_base_url or args.translation_base_url
    api_key = args.annotation_api_key or args.translation_api_key
    model = args.annotation_model or args.translation_model
    reasoning_effort = args.annotation_reasoning_effort or args.translation_reasoning_effort
    return base_url, api_key, model, reasoning_effort


def init_translation_service(
    args: argparse.Namespace,
) -> TranslationService:
    """
    根据命令行参数创建翻译服务实例。

    参数:
        args: 解析后的命令行参数对象。
    返回:
        TranslationService: 已配置好的翻译服务对象。
    """
    annotation_service = None
    if getattr(args, "enable_annotation", False):
        annotation_base_url, annotation_api_key, annotation_model, annotation_reasoning_effort = (
            _resolve_annotation_args(args)
        )
        annotation_client = configure_openai(
            base_url=annotation_base_url,
            api_key=annotation_api_key,
        )
        annotation_service = AnnotationService(
            client=annotation_client,
            model=annotation_model,
            reasoning_effort=annotation_reasoning_effort,
            enabled=True,
        )
    client = configure_openai(
        base_url=args.translation_base_url,
        api_key=args.translation_api_key,
    )
    return TranslationService(
        client=client,
        model=args.translation_model,
        temperature=args.translation_temperature,
        max_chunk_chars=args.translation_max_chunk_chars,
        latex_formula_handling=args.translation_latex_formula_handling,
        reasoning_effort=args.translation_reasoning_effort,
        _annotation_service=annotation_service,
    )
