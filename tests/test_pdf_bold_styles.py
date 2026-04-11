import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from ocr_pdf_bold_styles import apply_bold_texts_to_markdown, extract_bold_text_by_page, extract_bold_texts_from_page_dict  # noqa: E402


class PdfBoldStylesTest(unittest.TestCase):
    def test_apply_bold_texts_matches_repeated_strings_in_order(self) -> None:
        markdown = "Alpha. About figures. Beta. About figures. Gamma."
        result = apply_bold_texts_to_markdown(markdown, ["About figures."])
        self.assertEqual(result, "Alpha. **About figures.** Beta. About figures. Gamma.")

    def test_extract_bold_text_by_page_reads_fixture_pdf(self) -> None:
        pdf_path = PROJECT_ROOT / "tests" / "data" / "test_page_14.pdf"
        if not pdf_path.exists():
            self.skipTest(f"missing fixture: {pdf_path}")
        bold_by_page = extract_bold_text_by_page(str(pdf_path), [1])
        self.assertIn("Figure 1.1.2.", bold_by_page[1])
        self.assertIn("About figures.", bold_by_page[1])

    def test_extract_bold_texts_from_page_dict_merges_adjacent_bold_spans(self) -> None:
        page_dict = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {"text": "Deep", "font": "Some-Bold", "size": 12, "bbox": (0, 0, 30, 12)},
                                {"text": "Learning", "font": "Some-Bold", "size": 12, "bbox": (33, 0, 90, 12)},
                            ]
                        }
                    ]
                }
            ]
        }

        self.assertEqual(extract_bold_texts_from_page_dict(page_dict), ["Deep Learning"])

    def test_extract_bold_texts_from_page_dict_keeps_separated_bold_spans_apart(self) -> None:
        page_dict = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {"text": "Deep", "font": "Some-Bold", "size": 12, "bbox": (0, 0, 30, 12)},
                                {"text": "Learning", "font": "Some-Bold", "size": 12, "bbox": (50, 0, 110, 12)},
                            ]
                        }
                    ]
                }
            ]
        }

        self.assertEqual(extract_bold_texts_from_page_dict(page_dict), ["Deep", "Learning"])

    def test_extract_bold_texts_from_page_dict_stops_merge_at_non_bold_span(self) -> None:
        page_dict = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {"text": "Deep", "font": "Some-Bold", "size": 12, "bbox": (0, 0, 30, 12)},
                                {"text": " ", "font": "Some-Regular", "size": 12, "bbox": (30, 0, 34, 12)},
                                {"text": "Learning", "font": "Some-Bold", "size": 12, "bbox": (34, 0, 91, 12)},
                            ]
                        }
                    ]
                }
            ]
        }

        self.assertEqual(extract_bold_texts_from_page_dict(page_dict), ["Deep", "Learning"])
