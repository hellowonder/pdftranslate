#!/usr/bin/env python3
"""
Shared utilities for calling OpenAI-compatible LLM backends.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from typing import Any, Sequence

from openai import OpenAI

CODE_FENCE_PATTERN = re.compile(
    r"^\s*```(?:[\w-]+)?\s*\n(?P<body>[\s\S]*?)\n```?\s*$",
    re.DOTALL,
)
THINK_TAG_PATTERN = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_OPEN_TAG_PATTERN = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
THINK_CLOSE_TAG_PATTERN = re.compile(r"</think\s*>", re.IGNORECASE)

def has_low_diversity_or_repetition(text: str) -> bool:
    tokens = (text or "").split()
    if len(tokens) >= 20:
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < 0.1:
            return True

    if len(tokens) < 6:
        return False

    n = 6
    ngrams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counter = Counter(ngrams)
    return any(count > 10 for count in counter.values())


def configure_openai(base_url: str, api_key: str) -> OpenAI:
    """
    Instantiate an OpenAI client pointed at an OpenAI-compatible endpoint.
    """
    return OpenAI(base_url=base_url, api_key=api_key)


def strip_code_fences(text: str) -> str:
    """
    Remove triple backtick fences wrapped around the whole string.
    """
    if not text:
        return text
    match = CODE_FENCE_PATTERN.match(text.strip())
    if match:
        return match.group("body").strip()
    return text


def clean_model_output(text: str) -> str:
    """
    Normalize LLM responses by removing <think> tags and stray code fences.
    """
    if not text:
        return ""
    cleaned = THINK_TAG_PATTERN.sub("", text)
    closing_only = THINK_CLOSE_TAG_PATTERN.search(cleaned) and not THINK_OPEN_TAG_PATTERN.search(cleaned)
    if closing_only:
        closing_match = THINK_CLOSE_TAG_PATTERN.search(cleaned)
        if closing_match:
            cleaned = cleaned[closing_match.end():]
    cleaned = THINK_OPEN_TAG_PATTERN.sub("", cleaned)
    cleaned = THINK_CLOSE_TAG_PATTERN.sub("", cleaned)
    cleaned = strip_code_fences(cleaned.strip())
    return cleaned.strip()


def create_chat_completion_with_retry(
    client: Any,
    model: str,
    messages: Sequence[dict[str, str]],
    reasoning_effort: str = "none",
    max_retries: int = 3,
    error_label: str = "llm request",
) -> str:
    last_content = ""
    for attempt in range(1, max_retries + 1):
        try:
            if hasattr(client, "create_chat_completion"):
                return clean_model_output(
                    client.create_chat_completion(
                        model=model,
                        messages=list(messages),
                    )
                )
            request_kwargs = {
                "model": model,
                "messages": list(messages),
            }
            if reasoning_effort:
                request_kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
            response = client.chat.completions.create(**request_kwargs)
            return clean_model_output(response.choices[0].message.content or "")
        except Exception as exc:  # pragma: no cover - network failures hard to mock per call
            last_content = f"{error_label} error: {exc}"
            print(
                f"{error_label} error (attempt {attempt}/{max_retries}): {exc}",
                file=sys.stderr,
            )
    return last_content
