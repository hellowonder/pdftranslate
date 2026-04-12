import argparse
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from ocr_markdown import init_ocr_client, is_nearly_blank_page  # noqa: E402
from ocr_client import resize_image_for_ocr  # noqa: E402
from ocr_deepseek_postprocess import build_deepseek_page_markdown  # noqa: E402


class OCRMarkdownBlankPageTest(unittest.TestCase):
    def test_blank_page_is_detected(self) -> None:
        image = Image.new("RGB", (1200, 1600), "white")
        self.assertTrue(is_nearly_blank_page(image))

    def test_text_page_is_not_detected_as_blank(self) -> None:
        image = Image.new("RGB", (1200, 1600), "white")
        draw = ImageDraw.Draw(image)
        for idx in range(30):
            y = 80 + idx * 40
            draw.rectangle((80, y, 950, y + 10), fill="black")
        self.assertFalse(is_nearly_blank_page(image))

    def test_resize_image_for_ocr_limits_max_side(self) -> None:
        image = Image.new("RGB", (1239, 1776), "white")
        resized = resize_image_for_ocr(image, max_side=640)
        self.assertEqual(resized.size, (446, 640))

    def test_process_grounded_markdown_preserves_image_position(self) -> None:
        image = Image.new("RGB", (1000, 1600), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((200, 450, 800, 950), fill="black")
        grounded = """<|ref|>text<|/ref|><|det|>[[100, 100, 900, 180]]<|/det|>
Top paragraph.

<|ref|>image<|/ref|><|det|>[[200, 300, 800, 600]]<|/det|>
<|ref|>image_caption<|/ref|><|det|>[[220, 620, 780, 680]]<|/det|>
<center>Figure 1.2.10. Caption text.</center>

<|ref|>text<|/ref|><|det|>[[100, 720, 900, 820]]<|/det|>
Bottom paragraph.
"""
        output_dir = PROJECT_ROOT / "tests" / "data" / "output" / "grounded_images"
        markdown = build_deepseek_page_markdown(
            raw_text=grounded,
            image=image,
            image_output_dir=str(output_dir),
            page_number=1,
        )
        self.assertIn("Top paragraph.\n\n![](images/1_0.jpg)\n\nFigure 1.2.10. Caption text.\n\nBottom paragraph.", markdown)
        self.assertTrue((output_dir / "1_0.jpg").exists())

    def test_process_grounded_markdown_preserves_equation_blocks(self) -> None:
        image = Image.new("RGB", (1000, 1600), "white")
        grounded = r"""<|ref|>text<|/ref|><|det|>[[100, 100, 900, 180]]<|/det|>
The equation is

<|ref|>equation<|/ref|><|det|>[[150, 200, 850, 260]]<|/det|>
\[x^2 + y^2 = 1\]

<|ref|>text<|/ref|><|det|>[[100, 280, 900, 360]]<|/det|>
Done.
"""
        markdown = build_deepseek_page_markdown(
            raw_text=grounded,
            image=image,
            image_output_dir=str(PROJECT_ROOT / "tests" / "data" / "output" / "grounded_images"),
            page_number=2,
        )
        self.assertIn("The equation is", markdown)
        self.assertIn(r"\[x^2 + y^2 = 1\]", markdown)
        self.assertIn("Done.", markdown)

    def test_process_grounded_markdown_preserves_plain_text_around_mixed_equation_blocks(self) -> None:
        image = Image.new("RGB", (1000, 1600), "white")
        grounded = r"""Leading paragraph before grounded block.

<|ref|>equation<|/ref|><|det|>[[150, 200, 850, 260]]<|/det|>
\[x^2 + y^2 = 1\]

Trailing paragraph after grounded block.
"""
        markdown = build_deepseek_page_markdown(
            raw_text=grounded,
            image=image,
            image_output_dir=str(PROJECT_ROOT / "tests" / "data" / "output" / "grounded_images"),
            page_number=3,
        )
        self.assertIn("Leading paragraph before grounded block.", markdown)
        self.assertIn(r"\[x^2 + y^2 = 1\]", markdown)
        self.assertIn("Trailing paragraph after grounded block.", markdown)

    def test_process_grounded_markdown_splits_equation_tail_text(self) -> None:
        image = Image.new("RGB", (1000, 1600), "white")
        grounded = r"""Before.

<|ref|>equation<|/ref|><|det|>[[150, 200, 850, 260]]<|/det|>
\[x^2 + y^2 = 1\]

After.
"""
        markdown = build_deepseek_page_markdown(
            raw_text=grounded,
            image=image,
            image_output_dir=str(PROJECT_ROOT / "tests" / "data" / "output" / "grounded_images"),
            page_number=4,
        )
        self.assertEqual(markdown, "Before.\n\n\\[x^2 + y^2 = 1\\]\n\nAfter.")


class OCRMarkdownClientInitTest(unittest.TestCase):
    def test_init_ocr_client_uses_openai_configuration(self) -> None:
        args = argparse.Namespace(
            ocr_base_url="http://localhost:11434/v1",
            ocr_api_key="secret",
            ocr_model="deepseek-ocr:3b",
        )

        with patch("ocr_client.configure_openai", return_value="client") as mocked_configure, patch(
            "ocr_client_deepseek.DeepseekOCRClient"
        ) as mocked_client:
            init_ocr_client(args)

        mocked_configure.assert_called_once_with("http://localhost:11434/v1", "secret")
        mocked_client.assert_called_once_with(
            client="client",
            model="deepseek-ocr:3b",
        )


if __name__ == "__main__":
    unittest.main()
