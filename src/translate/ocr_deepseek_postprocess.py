#!/usr/bin/env python3
"""
DeepSeek OCR 原始输出的解析与单页 Markdown 生成。
"""
from __future__ import annotations

import ast
import os
import re
from typing import List, Optional, Tuple

from PIL import Image

REF_PATTERN2 = r"^((.*?)\[\[(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*)\]\])\s*$"
DISPLAY_FORMULA_PATTERN = re.compile(
    r"^\s*(?P<formula>(?:\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$|\\begin\{[a-zA-Z*]+\}[\s\S]*?\\end\{[a-zA-Z*]+\}))(?P<tail>[\s\S]*)$",
    re.DOTALL,
)
LAYOUT_PATTERN = re.compile(
    r"<\|ref\|>(?P<label>.*?)<\|/ref\|><\|det\|>\[\[(?P<coords>\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+)\]\]<\|/det\|>(?P<body>.*?)(?=(?:<\|ref\|>)|\Z)",
    re.DOTALL,
)

TEXTUAL_REGION_LABELS = {
    "text",
    "title",
    "sub_title",
    "list",
    "index",
    "table",
    "image_caption",
    "table_caption",
    "equation",
    "interline_equation",
}


def re_match2(text: str) -> Tuple[List[Tuple[str, str, str]], List[str], List[str]]:
    matches = re.findall(REF_PATTERN2, text, re.MULTILINE)
    matches_image: List[str] = []
    matches_other: List[str] = []
    for raw, ref_value, _ in matches:
        if ref_value == "image":
            matches_image.append(raw)
        else:
            matches_other.append(raw)
    return matches, matches_image, matches_other


def extract_coordinates_and_label(ref_text: Tuple[str, str, str]) -> Optional[Tuple[str, List]]:
    try:
        label_type = ref_text[1]
        cor_list = [ast.literal_eval(ref_text[2])]
    except (ValueError, SyntaxError):
        return None
    return label_type, cor_list


def save_referenced_images(image: Image.Image, refs, output_dir: str, page_idx: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    image_width, image_height = image.size
    img_idx = 0
    for ref in refs:
        parsed = extract_coordinates_and_label(ref)
        if not parsed:
            continue
        label_type, points_list = parsed
        if label_type != "image":
            continue
        for points in points_list:
            try:
                x1, y1, x2, y2 = points
            except ValueError:
                continue
            x1 = int(x1 / 999 * image_width)
            y1 = int(y1 / 999 * image_height)
            x2 = int(x2 / 999 * image_width)
            y2 = int(y2 / 999 * image_height)
            try:
                cropped = image.crop((x1, y1, x2, y2)).convert("RGB")
                cropped.save(os.path.join(output_dir, f"{page_idx}_{img_idx}.jpg"))
                img_idx += 1
            except Exception:
                continue


def parse_layout_regions(layout_content: str) -> List[dict]:
    regions: List[dict] = []
    for match in LAYOUT_PATTERN.finditer(layout_content or ""):
        coords_raw = match.group("coords")
        try:
            coords = [int(part.strip()) for part in coords_raw.split(",")]
        except ValueError:
            continue
        body = match.group("body").strip()
        regions.append(
            {
                "label": match.group("label").strip(),
                "coords": coords,
                "body": body,
            }
        )
    return regions


def crop_layout_images(
    image: Image.Image,
    regions: List[dict],
    output_dir: str,
    page_idx: int,
) -> List[dict]:
    os.makedirs(output_dir, exist_ok=True)
    image_width, image_height = image.size
    cropped_regions: List[dict] = []
    image_idx = 0
    for region in regions:
        if region["label"] != "image":
            continue
        x1, y1, x2, y2 = region["coords"]
        px1 = int(x1 / 999 * image_width)
        py1 = int(y1 / 999 * image_height)
        px2 = int(x2 / 999 * image_width)
        py2 = int(y2 / 999 * image_height)
        if px2 <= px1 or py2 <= py1:
            continue
        filename = f"{page_idx}_{image_idx}.jpg"
        image.crop((px1, py1, px2, py2)).convert("RGB").save(os.path.join(output_dir, filename))
        cropped_regions.append(
            {
                **region,
                "filename": filename,
                "center_y": (y1 + y2) / 2,
            }
        )
        image_idx += 1
    return cropped_regions


def normalize_layout_text(text: str) -> str:
    normalized = re.sub(r"</?center>", "", text or "")
    normalized = normalized.replace("**", "").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _append_plain_text_blocks(blocks: List[str], text: str) -> None:
    for part in re.split(r"\n\s*\n", text or ""):
        cleaned = part.strip()
        if cleaned:
            blocks.append(cleaned)


def split_region_body(label: str, body: str) -> Tuple[str, str]:
    if label not in {"equation", "interline_equation"}:
        return body, ""
    match = DISPLAY_FORMULA_PATTERN.match(body or "")
    if not match:
        return body, ""
    return match.group("formula").strip(), match.group("tail")


def grounded_content_to_markdown(
    content: str,
    image: Image.Image,
    output_dir: str,
    page_idx: int,
) -> str:
    matches = list(LAYOUT_PATTERN.finditer(content or ""))
    if not matches:
        return ""

    regions = parse_layout_regions(content)
    cropped_regions = crop_layout_images(image, regions, output_dir, page_idx)
    image_by_coords = {tuple(region["coords"]): region for region in cropped_regions}
    blocks: List[str] = []

    for idx, match in enumerate(matches):
        _append_plain_text_blocks(blocks, content[matches[idx - 1].end() if idx > 0 else 0 : match.start()])

        label = match.group("label").strip()
        body, trailing_plain_text = split_region_body(label, match.group("body"))
        if label == "image":
            coords = [int(part.strip()) for part in match.group("coords").split(",")]
            cropped = image_by_coords.get(tuple(coords))
            if cropped:
                blocks.append(f"![](images/{cropped['filename']})")
            continue
        if label in TEXTUAL_REGION_LABELS:
            normalized = normalize_layout_text(body)
            if normalized:
                blocks.append(normalized)
        _append_plain_text_blocks(blocks, trailing_plain_text)

    _append_plain_text_blocks(blocks, content[matches[-1].end() :])
    return "\n\n".join(blocks).strip()


def inject_images_from_layout(markdown: str, regions: List[dict], page_idx: int) -> str:
    image_regions = [region for region in regions if region["label"] == "image"]
    caption_regions = [region for region in regions if region["label"] == "image_caption"]
    if not image_regions:
        return markdown

    injected = markdown
    used_filenames: set[str] = set()
    for image_region in image_regions:
        marker = f"![](images/{image_region['filename']})\n\n"
        target_caption = None
        for caption_region in caption_regions:
            if caption_region.get("center_y", 0) > image_region.get("center_y", 0):
                target_caption = caption_region
                break
        if target_caption:
            caption_text = normalize_layout_text(target_caption["body"])
            if caption_text and caption_text in injected:
                injected = injected.replace(caption_text, f"{marker}{caption_text}", 1)
                used_filenames.add(image_region["filename"])
                continue
        injected = f"{marker}{injected}"
        used_filenames.add(image_region["filename"])

    remaining = [region for region in image_regions if region["filename"] not in used_filenames]
    if remaining:
        prefix = "".join(f"![](images/{region['filename']})\n\n" for region in remaining)
        injected = prefix + injected
    return injected


def build_deepseek_page_markdown(
    raw_text: str,
    image: Image.Image,
    image_output_dir: str,
    page_number: int,
) -> str:
    content = (raw_text or "").replace("<｜end▁of▁sentence｜>", "").strip()
    grounded_regions = parse_layout_regions(content)
    if grounded_regions:
        return grounded_content_to_markdown(
            content,
            image,
            image_output_dir,
            page_number,
        )

    matches_ref, matches_images, matches_other = re_match2(content)
    save_referenced_images(image, matches_ref, image_output_dir, page_number)
    for idx, image_match in enumerate(matches_images):
        content = content.replace(
            image_match,
            f"![](images/{page_number}_{idx}.jpg)\n",
        )

    for other_match in matches_other:
        content = content.replace(other_match, "")

    content = (
        content.replace("\\coloneqq", ":=")
        .replace("\\eqqcolon", "=:")
        .replace("\n\n\n\n", "\n\n")
        .replace("\n\n\n", "\n\n")
    )
    return content.strip()
