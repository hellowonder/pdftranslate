import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from ocr_pdf_bold_styles import BoldAnchor, apply_bold_texts_to_markdown, extract_bold_text_by_page, extract_bold_texts_from_page_dict  # noqa: E402


class PdfBoldStylesTest(unittest.TestCase):
    def test_apply_bold_texts_matches_repeated_strings_using_context(self) -> None:
        markdown = "Alpha. About figures. Beta. About figures. Gamma."
        result = apply_bold_texts_to_markdown(
            markdown,
            [BoldAnchor(text="About figures.", left_context="Beta.", right_context="Gamma.")],
        )
        self.assertEqual(result, "Alpha. About figures. Beta. **About figures.** Gamma.")

    def test_apply_bold_texts_skips_markdown_image_destinations(self) -> None:
        markdown = "![](images/39_0.jpg)\n\nj starts paragraph."
        result = apply_bold_texts_to_markdown(
            markdown,
            [BoldAnchor(text="j", left_context="", right_context="starts paragraph.")],
        )
        self.assertEqual(result, "![](images/39_0.jpg)\n\nj starts paragraph.")

    def test_apply_bold_texts_normalizes_ligatures_and_uses_unique_fallback(self) -> None:
        markdown = "Definition 3.1 A manifold is smooth."
        result = apply_bold_texts_to_markdown(
            markdown,
            [BoldAnchor(text="Deﬁnition 3.1", left_context="", right_context="A map i : X → M")],
        )
        self.assertEqual(result, "**Definition 3.1** A manifold is smooth.")

    def test_apply_bold_texts_accepts_single_sided_context_when_best_match_is_clear(self) -> None:
        markdown = "A closed embedding is proper. Another closed embedding appears later."
        result = apply_bold_texts_to_markdown(
            markdown,
            [BoldAnchor(text="closed embedding", left_context="A", right_context="is a proper4 injective")],
        )
        self.assertEqual(result, "A **closed embedding** is proper. Another closed embedding appears later.")

    def test_apply_bold_texts_inserts_ranges_by_position_not_anchor_order(self) -> None:
        markdown = "Alpha Beta Gamma Delta"
        result = apply_bold_texts_to_markdown(
            markdown,
            [
                BoldAnchor(text="Gamma", left_context="Beta", right_context="Delta"),
                BoldAnchor(text="Alpha", left_context="", right_context="Beta"),
                BoldAnchor(text="Delta", left_context="Gamma", right_context=""),
            ],
        )
        self.assertEqual(result, "**Alpha** Beta **Gamma** **Delta**")

    def test_apply_bold_texts_skips_markdown_heading_lines(self) -> None:
        markdown = "## 3 Lagrangian Submanifolds\n\n### 3.1 Submanifolds"
        result = apply_bold_texts_to_markdown(
            markdown,
            [
                BoldAnchor(text="Lagrangian Submanifolds", left_context="", right_context=""),
                BoldAnchor(text="Submanifolds", left_context="3.1", right_context=""),
            ],
        )
        self.assertEqual(result, markdown)

    def test_extract_bold_text_by_page_reads_fixture_pdf(self) -> None:
        pdf_path = PROJECT_ROOT / "tests" / "data" / "test_page_14.pdf"
        if not pdf_path.exists():
            self.skipTest(f"missing fixture: {pdf_path}")
        bold_by_page = extract_bold_text_by_page(str(pdf_path), [1])
        page_items = bold_by_page[1]
        self.assertTrue(any(item.text == "Figure 1.1.2." for item in page_items))
        self.assertTrue(any(item.text == "About figures." for item in page_items))

    def test_extract_bold_texts_from_page_dict_merges_adjacent_bold_spans(self) -> None:
        page_dict = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {"text": "An ", "font": "Some-Regular", "size": 12, "bbox": (0, 0, 18, 12)},
                                {"text": "Important", "font": "Some-Bold", "size": 12, "bbox": (18, 0, 75, 12)},
                                {"text": "Term", "font": "Some-Bold", "size": 12, "bbox": (78, 0, 115, 12)},
                                {"text": " appears.", "font": "Some-Regular", "size": 12, "bbox": (118, 0, 170, 12)},
                            ]
                        }
                    ]
                }
            ]
        }

        self.assertEqual(
            extract_bold_texts_from_page_dict(page_dict),
            [BoldAnchor(text="Important Term", left_context="An", right_context="appears.")],
        )

    def test_extract_bold_texts_from_page_dict_keeps_bold_space_inside_same_anchor(self) -> None:
        page_dict = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {"text": "Part", "font": "Some-Bold", "size": 12, "bbox": (0, 0, 28, 12)},
                                {"text": " ", "font": "Some-Bold", "size": 12, "bbox": (28, 0, 32, 12)},
                                {"text": "II", "font": "Some-Bold", "size": 12, "bbox": (32, 0, 45, 12)},
                                {"text": " title", "font": "Some-Regular", "size": 12, "bbox": (45, 0, 80, 12)},
                            ]
                        }
                    ]
                }
            ]
        }

        self.assertEqual(
            extract_bold_texts_from_page_dict(page_dict),
            [BoldAnchor(text="Part II", left_context="", right_context="title")],
        )

    def test_extract_bold_texts_from_page_dict_keeps_separated_bold_spans_apart(self) -> None:
        page_dict = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {"text": "Alpha ", "font": "Some-Regular", "size": 12, "bbox": (0, 0, 35, 12)},
                                {"text": "Deep", "font": "Some-Bold", "size": 12, "bbox": (35, 0, 65, 12)},
                                {"text": " ", "font": "Some-Regular", "size": 12, "bbox": (65, 0, 69, 12)},
                                {"text": "Learning", "font": "Some-Bold", "size": 12, "bbox": (90, 0, 147, 12)},
                                {"text": " Omega", "font": "Some-Regular", "size": 12, "bbox": (147, 0, 190, 12)},
                            ]
                        }
                    ]
                }
            ]
        }

        self.assertEqual(
            extract_bold_texts_from_page_dict(page_dict),
            [
                BoldAnchor(text="Deep", left_context="Alpha", right_context="Learning Omega"),
                BoldAnchor(text="Learning", left_context="Alpha Deep", right_context="Omega"),
            ],
        )

    def test_extract_bold_texts_from_page_dict_ignores_short_bold_noise(self) -> None:
        page_dict = {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {"text": "x", "font": "Some-Bold", "size": 12, "bbox": (0, 0, 10, 12)},
                                {"text": " normal", "font": "Some-Regular", "size": 12, "bbox": (10, 0, 60, 12)},
                            ]
                        }
                    ]
                }
            ]
        }

        self.assertEqual(extract_bold_texts_from_page_dict(page_dict), [])


if __name__ == "__main__":
    unittest.main()
