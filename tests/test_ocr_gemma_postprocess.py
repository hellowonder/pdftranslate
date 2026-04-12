import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from ocr_gemma_postprocess import build_gemma_page_markdown  # noqa: E402


class GemmaPostprocessTest(unittest.TestCase):
    def test_build_gemma_page_markdown_rewrites_image_placeholders(self) -> None:
        image = Image.new("RGB", (1000, 1600), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((200, 480, 800, 960), fill="black")
        raw_text = "Intro.\n\n![](image_200_300_800_600.png)\n\nCaption."
        output_dir = PROJECT_ROOT / "tests" / "data" / "output" / "gemma_postprocess"

        markdown = build_gemma_page_markdown(
            raw_text=raw_text,
            image=image,
            image_output_dir=str(output_dir),
            page_number=7,
        )

        self.assertEqual(markdown, "Intro.\n\n![](images/7_0.jpg)\n\nCaption.")
        self.assertTrue((output_dir / "7_0.jpg").exists())


if __name__ == "__main__":
    unittest.main()
