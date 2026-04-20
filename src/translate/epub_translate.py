#!/usr/bin/env python3
"""
Translate EPUB files by interleaving original paragraphs with their translations.
"""
from __future__ import annotations

import argparse
import os
import sys
import itertools
import io
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup, Tag
from ebooklib import epub, ITEM_IMAGE
from PIL import Image, ImageDraw, ImageFont

from translate_service import (
    TranslationService,
    add_translation_arguments,
    init_translation_service,
    validate_translation_args,
)

SUPPORTED_BLOCK_TAGS: Tuple[str, ...] = (
    "p",
    "ul",
    "ol",
    "li",
    "dl",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "figcaption",
    "table",
)
DEFAULT_TRANSLATION_CLASS = "translation-text"
ORIGINAL_DETAILS_CLASS = "original-text-details"
ORIGINAL_DETAILS_SUMMARY = "_"
UNWANTED_BLOCK_TEXTS = {"OceanOfPDF.com".lower()}


@dataclass
class HtmlSegment:
    soup: BeautifulSoup
    tag: Tag
    text: str
    raw_html: str


def resolve_spine_range(spine_arg: Optional[str], total_docs: int) -> List[int]:
    """
    Parse the --spine-range argument into a sorted list of 1-based spine indexes.
    """
    if total_docs <= 0:
        return []
    if not spine_arg:
        return list(range(1, total_docs + 1))
    selected = set()
    for part in spine_arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                raise ValueError(f"Invalid spine range: '{part}'")
            if start > end:
                start, end = end, start
            for idx in range(start, end + 1):
                selected.add(idx)
        else:
            try:
                selected.add(int(part))
            except ValueError:
                raise ValueError(f"Invalid spine index: '{part}'")
    filtered = sorted(idx for idx in selected if 1 <= idx <= total_docs)
    if not filtered:
        raise ValueError("No valid spine indexes selected after filtering.")
    return filtered


def extract_translatable_segments(
    soup: BeautifulSoup,
    block_tags: Sequence[str] = SUPPORTED_BLOCK_TAGS,
    translation_class: str = DEFAULT_TRANSLATION_CLASS,
) -> List[HtmlSegment]:
    """
    Collect textual blocks that should be translated from a soup document.
    """
    segments: List[HtmlSegment] = []
    tag_names: Tuple[str, ...] = tuple(block_tags)
    for tag in soup.find_all(tag_names):
        if tag.find_parent(tag_names):
            continue
        classes = tag.get("class") or []
        if translation_class and translation_class in classes:
            continue
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
        if _should_remove_block(text):
            tag.decompose()
            continue
        segments.append(
            HtmlSegment(
                soup=soup,
                tag=tag,
                text=text,
                raw_html=str(tag),
            )
        )
    return segments


def apply_translations(
    segments: Sequence[HtmlSegment],
    translations: Sequence[str],
    translation_class: str = DEFAULT_TRANSLATION_CLASS,
    mode: str = "interleaved",
) -> None:
    """
    Merge translations back into the soup according to the requested mode.
    """
    if len(segments) != len(translations):
        raise ValueError("Segment/translation length mismatch.")
    index = 0
    if mode == "translated-only":
        the_soup = segments[0].soup
        footnotes_div = the_soup.new_tag("div", attrs={"data-type": "footnotes"})
        the_soup.body.append(footnotes_div)
        skip_original = the_soup.new_tag("a", href="#__end__end")
        skip_original.string = "skip original"
        footnotes_div.insert_before(skip_original)
        the_soup.body.append(the_soup.new_tag("div", id="__end__end"))
    for segment, translated in zip(segments, translations):
        if not translated:
            continue
        new_tag = _build_translated_tag(segment, translated, translation_class)
        if new_tag:
            if mode == "translated-only":
                ref = segment.soup.new_tag("a", 
                                           attrs = {
                                               "href" : f"#__ref__{index}", 
                                               "id" : f"_ref{index}",
                                               "data-type": "noteref",
                                               "epub:type" : "noteref"
                                           })
                ref.string = '-'
                new_tag.append(ref)
                segment.tag.insert_before(new_tag)

                # section = segment.soup.new_tag("p", attrs = {"epub:type": "footnote", "id": f"__ref__{index}", "data-type": "footnote"})
                original = segment.tag.extract()
                original["id"] = f"__ref__{index}"
                # section.append(original)
                footnotes_div.append(original)
                index += 1
            elif mode == "interleaved":
                segment.tag.insert_before(new_tag)
            else:
                raise ValueError(f"Unsupported translation merge mode: {mode}")

def _build_translated_tag(
    segment: HtmlSegment,
    translated_html: str,
    translation_class: str,
) -> Optional[Tag]:
    parsed = BeautifulSoup(translated_html, "html.parser")
    candidate = parsed.find(segment.tag.name)
    if candidate:
        candidate = candidate.extract()
    else:
        candidate = segment.soup.new_tag(segment.tag.name)
        candidate.string = translated_html

    combined_classes: List[str] = []
    existing_from_candidate = candidate.get("class") or []
    combined_classes.extend(existing_from_candidate)
    base_classes = segment.tag.get("class") or []
    combined_classes.extend(base_classes)
    if translation_class:
        combined_classes.append(translation_class)
    if combined_classes:
        candidate["class"] = list(dict.fromkeys(combined_classes))
    return candidate


def _should_remove_block(text: str) -> bool:
    """
    Return True if the block should be removed entirely (both source and translation).
    """
    normalized = text.strip().lower()
    return normalized in UNWANTED_BLOCK_TEXTS


def _wrap_original_in_details(segment: HtmlSegment) -> Tag:
    """
    Wrap the original tag in a <details> element so it can be toggled in translated-only mode.
    """
    wrapper = segment.soup.new_tag("details")
    if ORIGINAL_DETAILS_CLASS:
        wrapper["class"] = [ORIGINAL_DETAILS_CLASS]
    summary = segment.soup.new_tag("summary")
    summary.string = ORIGINAL_DETAILS_SUMMARY
    segment.tag.wrap(wrapper)
    wrapper.insert(0, summary)
    return wrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate EPUB spine documents and interleave original + translated paragraphs."
    )
    parser.add_argument("--input", required=True, help="Input EPUB file.")
    parser.add_argument("--output", required=True, help="Output EPUB file with interleaved text.")
    parser.add_argument(
        "--output-cn",
        help="Optional EPUB path to write a Chinese-only translation (original text removed).",
    )
    parser.add_argument(
        "--spine-range",
        help="Comma separated 1-based spine indexes or ranges (e.g. 1-3,5). Defaults to all spine items.",
    )
    parser.add_argument(
        "--translation-class",
        default=DEFAULT_TRANSLATION_CLASS,
        help="CSS class added to inserted translation blocks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output EPUB if it already exists.",
    )
    add_translation_arguments(parser)
    args = parser.parse_args()
    try:
        validate_translation_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


class EpubProcessor:
    """
    Drive the end-to-end EPUB translation workflow for a single book.
    """

    def __init__(
        self,
        translation_service: TranslationService,
        *,
        translation_class: str = DEFAULT_TRANSLATION_CLASS,
        block_tags: Iterable[str] = SUPPORTED_BLOCK_TAGS,
        translation_workers: int = 1,
    ) -> None:
        self.translation_service = translation_service
        self.translation_class = translation_class
        self.block_tags: Tuple[str, ...] = tuple(block_tags)
        self.translation_workers = max(1, translation_workers)

    def translate_epub(
        self,
        input_path: str,
        output_path: str,
        *,
        output_cn_path: Optional[str] = None,
        spine_range: Optional[str] = None,
        overwrite: bool = False,
    ) -> None:
        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)
        output_cn_path = os.path.abspath(output_cn_path) if output_cn_path else None
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input EPUB not found: {input_path}")
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(output_path) and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}. Use --overwrite to replace it."
            )
        if output_cn_path:
            cn_dir = os.path.dirname(output_cn_path) or "."
            os.makedirs(cn_dir, exist_ok=True)
            if os.path.exists(output_cn_path) and not overwrite:
                raise FileExistsError(
                    f"Output file already exists: {output_cn_path}. Use --overwrite to replace it."
                )

        print(f"Loading EPUB from {input_path}...", file=sys.stderr)
        book = epub.read_epub(input_path)
        total_spine_items = len(book.spine)
        selected_spine = resolve_spine_range(spine_range, total_spine_items)
        selected_order = sorted(selected_spine)
        items = self._collect_epubitems(book, selected_order)

        book_cn: Optional[epub.EpubBook] = None
        processed_docs_cn: List[Tuple[epub.EpubItem, BeautifulSoup]] = []

        if output_cn_path:
            book_cn = epub.read_epub(input_path)
            items_cn = self._collect_epubitems(book_cn, selected_order)
            assert len(items) == len(items_cn)

        self._translate_metadata(book, mirror_book=book_cn)

        if not items:
            print("No eligible blocks found; copying EPUB without changes.", file=sys.stderr)
            epub.write_epub(output_path, book)
            print(f"Done. Output written to {output_path}")
            if output_cn_path and book_cn:
                epub.write_epub(output_cn_path, book_cn)
                print(f"Done. Output written to {output_cn_path}")
            return

        print(f"Collected {len(items)} pages for translation.", file=sys.stderr)
        modes = ["interleaved"] if not output_cn_path else ["interleaved", "translated-only"]
        for idx, item in enumerate(items):
            results = self.translate_html(item.get_content(), modes)
            if not results:
                continue
            if results[0]:
                item.set_content(results[0].encode("utf-8"))
            if output_cn_path and results[1]:
                items_cn[idx].set_content(results[1].encode("utf-8"))
        self._add_cover_badge(book, "CN")
        if output_cn_path and book_cn:
            self._add_cover_badge(book_cn, "CN2")
        self._repair_toc(book)
        epub.write_epub(output_path, book)
        print(f"Done. Output written to {output_path}")

        if output_cn_path and book_cn:
            self._repair_toc(book_cn)
            epub.write_epub(output_cn_path, book_cn)
            print(f"Done. Output written to {output_cn_path}")

    def _translate_metadata(
        self,
        book: epub.EpubBook,
        *,
        mirror_book: Optional[epub.EpubBook] = None,
    ) -> None:
        """
        Translate metadata fields that are shown to readers (e.g., title/description).
        The translated values are applied to the primary book and an optional mirror copy.
        """
        translations = self._collect_metadata_translations(book)
        if not translations:
            return
        self._apply_metadata_translations(book, translations)
        if mirror_book:
            self._apply_metadata_translations(mirror_book, translations)

    def _collect_metadata_translations(
        self, book: epub.EpubBook
    ) -> dict[str, list[tuple[str, Optional[dict]]]]:
        fields = ("title", "description")
        translated: dict[str, list[tuple[str, Optional[dict]]]] = {}
        for field in fields:
            entries = book.get_metadata("DC", field)
            if not entries:
                continue
            translated_entries: list[tuple[str, Optional[dict]]] = []
            for value, attrs in entries:
                translated_value = self.translation_service.translate_text_block(
                    str(value),
                    mode="plain_text",
                )
                translated_entries.append((translated_value or value, attrs))
            translated[field] = translated_entries
        return translated

    @staticmethod
    def _apply_metadata_translations(
        book: epub.EpubBook, translations: dict[str, list[tuple[str, Optional[dict]]]]
    ) -> None:
        dc_namespace = epub.NAMESPACES.get("DC", "DC")
        book.metadata.setdefault(dc_namespace, {})
        for name, entries in translations.items():
            book.metadata[dc_namespace][name] = entries
            if name == "title" and entries:
                book.title = entries[0][0]

    def translate_segments(self, segments: Sequence[HtmlSegment]) -> List[str]:
        """
        Translate a batch of HtmlSegments via the configured translation service.
        """
        if not segments:
            return []
        payload = [
            segment.raw_html
            for segment in segments
        ]
        return self.translation_service.translate_pages(
            payload,
            max_workers=self.translation_workers,
            mode="html",
        )

    def translate_html(
        self, content, modes: list[str]
    ) -> list[str] | None:
        if not content:
            return None

        if not modes:
            return []
        result = []
        translations = None
        for mode in modes:
            soup = BeautifulSoup(content, "html.parser")
            segments = extract_translatable_segments(
                soup, block_tags=self.block_tags, translation_class=self.translation_class)
            if not segments:
                return None
            if not translations:  # 不同mode的translations应该是一样的
                translations = self.translate_segments(segments)
            apply_translations(segments, translations, mode=mode)
            result.append(str(soup))
        return result

    def _add_cover_badge(self, book: epub.EpubBook, badge_text: str) -> None:
        if not badge_text:
            return
        item = self._find_cover_item(book)
        if not item:
            print("No cover image found; skipping badge.", file=sys.stderr)
            return
        try:
            raw = item.get_content()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to read cover content: {exc}", file=sys.stderr)
            return
        try:
            with Image.open(io.BytesIO(raw)) as img:
                original_format = self._detect_image_format(item, img)
                canvas = img.convert("RGBA")
                draw = ImageDraw.Draw(canvas)
                font_size = max(12, int(min(canvas.size) * 0.1))
                font = self._load_badge_font(font_size)
                stroke_width = max(1, font_size // 15)
                text_bbox = draw.textbbox((0, 0), badge_text, font=font, stroke_width=stroke_width)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                padding = max(4, int(min(canvas.size) * 0.02))
                x = max(0, canvas.width - text_width - padding)
                y = max(0, canvas.height - text_height - padding)
                draw.text(
                    (x, y),
                    badge_text,
                    font=font,
                    fill=(255, 0, 0, 255),
                    stroke_width=stroke_width,
                    stroke_fill=(255, 255, 255, 255),
                )
                if original_format == "JPEG":
                    canvas = canvas.convert("RGB")
                buffer = io.BytesIO()
                canvas.save(buffer, format=original_format or "PNG")
                item.set_content(buffer.getvalue())
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to add badge to cover: {exc}", file=sys.stderr)

    def _find_cover_item(self, book: epub.EpubBook) -> Optional[epub.EpubItem]:
        cover_id = None
        for value, _attrs in book.get_metadata("OPF", "cover"):
            if value:
                cover_id = str(value)
                break
        if not cover_id:
            for value, attrs in book.get_metadata("OPF", "meta"):
                if attrs and attrs.get("name") == "cover":
                    cover_id = attrs.get("content")
                    break
        if cover_id:
            item = book.get_item_with_id(cover_id)
            if item:
                return item
        for item in book.get_items():
            if getattr(item, "media_type", "").startswith("image"):
                if isinstance(item, epub.EpubCover):
                    return item
                if getattr(item, "get_id", None) and item.get_id() == "cover-image":
                    return item
                props = getattr(item, "properties", []) or []
                if "cover-image" in props:
                    return item
        for item in book.get_items_of_type(ITEM_IMAGE):
            name = getattr(item, "file_name", "") or ""
            if "cover" in name.lower():
                return item
        images = list(book.get_items_of_type(ITEM_IMAGE))
        return images[0] if images else None

    def _detect_image_format(self, item: epub.EpubItem, img: Image.Image) -> str:
        media_type = getattr(item, "media_type", "").lower()
        if "jpeg" in media_type or "jpg" in media_type:
            return "JPEG"
        if "png" in media_type:
            return "PNG"
        if "gif" in media_type:
            return "GIF"
        if img.format:
            return img.format
        ext = os.path.splitext(getattr(item, "file_name", ""))[1].lower()
        if ext in {".jpg", ".jpeg"}:
            return "JPEG"
        if ext == ".png":
            return "PNG"
        if ext == ".gif":
            return "GIF"
        return "PNG"

    def _load_badge_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

    def _collect_epubitems(
        self,
        book: epub.EpubBook,
        selected_spine: Sequence[int],
    ) -> list[epub.EpubItem]:
        """
        Parse only the selected spine documents and gather both their soup trees
        (for later content replacement) and the HtmlSegments that need translation.
        """
        selected_set = set(selected_spine)
        result = []
        for idx, (idref, _) in enumerate(book.spine, start=1):
            if idx not in selected_set:
                continue
            item = book.get_item_with_id(idref)
            if not isinstance(item, epub.EpubItem):
                print("not isinstance of EpubHtml: " + str(type(item)))
                continue
            result.append(item)
        return result

    def _repair_toc(self, book: epub.EpubBook) -> None:
        """
        Ensure book.toc only contains supported node types with valid identifiers.
        """
        counter = itertools.count(1)
        book.toc = self._repair_toc_entries(book.toc, counter)

    def _repair_toc_entries(self, entries, counter) -> List:
        fixed: List = []
        if not entries:
            return fixed
        for entry in entries:
            repaired = self._repair_toc_entry(entry, counter)
            if repaired is None:
                continue
            fixed.append(repaired)
        return fixed

    def _repair_toc_entry(self, entry, counter):
        if isinstance(entry, (list, tuple)):
            if not entry:
                return None
            header = self._ensure_section_header(entry[0], counter)
            children_raw = entry[1] if len(entry) > 1 else []
            if not isinstance(children_raw, (list, tuple)):
                children_raw = []
            children = self._repair_toc_entries(children_raw, counter)
            return (header, tuple(children))
        if isinstance(entry, epub.Link):
            self._ensure_link_uid(entry, counter)
            return entry
        if isinstance(entry, epub.EpubHtml):
            self._ensure_html_id(entry, counter)
            return entry
        if isinstance(entry, epub.Section):
            return entry
        if isinstance(entry, str):
            return epub.Link(entry, entry, self._generate_uid("auto", counter))
        return None

    def _ensure_section_header(self, header, counter):
        if isinstance(header, epub.Link):
            self._ensure_link_uid(header, counter)
            return header
        if isinstance(header, epub.EpubHtml):
            self._ensure_html_id(header, counter)
            return header
        if isinstance(header, epub.Section):
            return header
        title = ""
        href = ""
        if isinstance(header, str):
            title = header
        else:
            title = getattr(header, "title", "") or ""
            href = getattr(header, "href", "") or ""
        if not title:
            title = f"Section {self._generate_uid('sec', counter)}"
        return epub.Section(title, href)

    def _ensure_link_uid(self, link: epub.Link, counter) -> None:
        if not getattr(link, "uid", None):
            link.uid = self._generate_uid("toc", counter)
        if not getattr(link, "title", ""):
            link.title = link.href or "Section"
        if not getattr(link, "href", ""):
            link.href = "#"

    def _ensure_html_id(self, doc: epub.EpubHtml, counter) -> None:
        if not doc.get_id():
            doc.id = self._generate_uid("doc", counter)
        if not getattr(doc, "title", ""):
            base = os.path.splitext(os.path.basename(doc.file_name))[0]
            doc.title = base or doc.get_id()

    @staticmethod
    def _generate_uid(prefix: str, counter) -> str:
        return f"{prefix}_{next(counter)}"


def _wrap_html_translation_prompt(html_snippet: str) -> str:
    instruction = (
        "Translate the following HTML snippet into natural, fluent Simplified Chinese. "
        "Preserve all HTML tags and attributes exactly as provided; only translate text nodes. "
        "Return only the translated HTML snippet.\n\n"
    )
    return f"{instruction}{html_snippet}"


def main() -> None:
    args = parse_args()
    print(
        f"Translating segments via {args.translation_base_url} using model {args.translation_model}...",
        file=sys.stderr,
    )
    translation_service: TranslationService = init_translation_service(args)
    processor = EpubProcessor(
        translation_service=translation_service,
        translation_class=args.translation_class,
        block_tags=SUPPORTED_BLOCK_TAGS,
        translation_workers=max(1, args.translation_workers),
    )
    # print(processor.translate_html('''<html><body><h1>hello</h1><p>it is a great day today.</p></body></html>'''))
    processor.translate_epub(
        input_path=args.input,
        output_path=args.output,
        output_cn_path=args.output_cn,
        spine_range=args.spine_range,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
    
