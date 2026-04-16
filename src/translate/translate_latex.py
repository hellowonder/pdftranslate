#!/usr/bin/env python3
"""
LaTeX 片段提取、校验与修复辅助函数。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import sys
from typing import Dict, List, Optional, Sequence, TextIO, Tuple
from Levenshtein import distance as edit_distance

LATEX_SEGMENT_PATTERN = re.compile(
    r"""
    (?P<display_dollar>(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$)
    |(?P<inline_dollar>(?<!\\)\$(?:\\.|[^$\\])*(?<!\\)\$)
    |(?P<display_bracket>\\\[[\s\S]*?\\\])
    |(?P<inline_paren>\\\([\s\S]*?\\\))
    |(?P<environment>\\begin\{(?P<env>[a-zA-Z*]+)\}[\s\S]*?\\end\{(?P=env)\})
    """,
    re.VERBOSE,
)


def _normalize_latex_segment(kind: str, text: str) -> str:
    """
    提取用于比较的公式主体，并移除全部空白字符。
    """
    if kind == "display_dollar":
        content = text[2:-2]
    elif kind == "inline_dollar":
        content = text[1:-1]
    elif kind in {"display_bracket", "inline_paren"}:
        content = text[2:-2]
    else:
        content = text
    return re.sub(r"\s+", "", content)


@dataclass(frozen=True)
class LatexSegment:
    start: int
    end: int
    text: str
    kind: str
    nospace: str  # 去掉两边的标记，以及内部的所有空白字符后的内容，用于比较相似度


def extract_latex_segments(text: str) -> List[LatexSegment]:
    """
    从文本中提取所有 LaTeX 公式片段。

    参数:
        text: 待扫描的文本。
    返回:
        List[LatexSegment]: 按出现顺序记录的公式片段信息。
    """
    segments: List[LatexSegment] = []
    for match in LATEX_SEGMENT_PATTERN.finditer(text):
        kind = next(name for name, value in match.groupdict().items() if name != "env" and value is not None)
        segments.append(
            LatexSegment(
                start=match.start(),
                end=match.end(),
                text=match.group(0),
                kind=kind,
                nospace=_normalize_latex_segment(kind, match.group(0)),
            )
        )
    return segments

def debug_dump_translation_pair(source: str, translation: str, debug_stream: TextIO) -> None:
    """
    将原文和译文成对打印到标准错误，便于排查翻译问题。

    参数:
        source: 原文文本。
        translation: 译文文本。
    返回:
        None: 仅输出调试日志。
    """
    print("source text:", file=debug_stream)
    print(source, file=debug_stream)
    print("translated text:", file=debug_stream)
    print(translation, file=debug_stream)
    print("-" * 40, file=debug_stream)


def append_latex_failure_marker(segment: LatexSegment) -> str:
    """
    在公式内容末尾追加失败标记，保留原有定界符形式。
    """
    marker = r"\textcircled{?}"
    if segment.kind == "display_dollar":
        return f"$${segment.text[2:-2]}{marker}$$"
    if segment.kind == "inline_dollar":
        return f"${segment.text[1:-1]}{marker}$"
    if segment.kind == "display_bracket":
        return f"\\[{segment.text[2:-2]}{marker}\\]"
    if segment.kind == "inline_paren":
        return f"\\({segment.text[2:-2]}{marker}\\)"
    if segment.kind == "environment":
        match = re.match(r"(\\begin\{[a-zA-Z*]+\})([\s\S]*)(\\end\{[a-zA-Z*]+\})", segment.text)
        if match:
            begin, content, end = match.groups()
            return f"{begin}{content}{marker}{end}"
    return f"{segment.text}{marker}"

def repair_translation_latex(
    source: str,
    translation: str,
    debug_stream: Optional[TextIO] = None,
) -> Tuple[str, bool]:
    """
    校验并尽量修复译文中的 LaTeX 公式。

    参数:
        source: 原文文本。
        translation: 模型输出的译文文本。
        debug_stream: 调试输出目标；为空时不打印 LaTeX 细节日志。
    返回:
        Tuple[str, bool]:
            第一个值是修复后的文本或原始译文，
            第二个值表示该结果是否可直接用于后续流程。
    """
    source_latex = extract_latex_segments(source)
    if not source_latex:
        return translation, True
    
    translation = translation.replace("$\\(", "$")
    translation = translation.replace("\\)$", "$")
    translation = translation.replace("$\\[", "$")
    translation = translation.replace("\\]$", "$")

    translation_latex = extract_latex_segments(translation)
    source_set = {item.nospace for item in source_latex}
    source_by_nospace = {item.nospace: item for item in source_latex}
    source_lengths = [(item, len(item.nospace)) for item in source_latex]
    fix_pairs = []
    failed_items = []
    fail = False
    for item in translation_latex:
        if item.nospace in source_set:
            source_item = source_by_nospace[item.nospace]
            if item.kind != source_item.kind:
                fix_pairs.append((item, source_item))
            continue
        if item.nospace not in source_set:
            # this is a sign of potential mistranslation or omission of LaTeX content; attempt to repair
            # we try to find a source LaTeX segment with minimal edit distance to this one, and replace it in the translation
            best_match = None
            best_distance = float("inf")
            item_len = len(item.nospace)
            for source_item, source_len in source_lengths:
                # Edit distance is always at least the absolute length difference.
                # If that lower bound already exceeds the current acceptance threshold,
                # this candidate can never be selected as a valid repair.
                max_len = max(item_len, source_len)
                acceptance_threshold = max(2.0, max_len * 0.2)
                if abs(item_len - source_len) > acceptance_threshold:
                    continue
                distance = edit_distance(item.nospace, source_item.nospace)
                if distance < best_distance:
                    best_distance = distance
                    best_match = source_item
            
            if item_len >= 3:  # only attempt repair for segments of reasonable length, to avoid false positives on very short segments
                if best_match and (best_distance == 1 or best_distance <= min(item_len*0.2, 5)):  # only consider it a match if it's reasonably close
                    if debug_stream is not None:
                        print(
                            f"Fix Latex: {item.nospace!r} ======> {best_match.nospace!r}, edit distance: {best_distance}",
                            file=debug_stream,
                        )
                    fix_pairs.append((item, best_match))
                else:
                    if debug_stream is not None:
                        print(
                            f"Fix Latex Fail: can't fix {item.text!r}",
                            file=debug_stream,
                        )
                    failed_items.append(item)
                    fail = True

    repaired = translation
    for item, best_match in fix_pairs:
        repaired = repaired.replace(item.text, best_match.text)
    #for item in failed_items:
    #    repaired = repaired.replace(item.text, append_latex_failure_marker(item))

    if fail and debug_stream is not None:
        debug_dump_translation_pair(source, repaired, debug_stream)  # 打印修复前的译文以便对比

    return repaired, not fail
