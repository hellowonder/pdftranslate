import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from ocr_client import OCRPageRequest  # noqa: E402
from ocr_gemma_client import GemmaOCRClient  # noqa: E402


class GemmaOCRClientTest(unittest.TestCase):
    def test_recognize_page_returns_markdown_and_crops_grounded_images(self) -> None:
        grounded_output = """<|ref|>text<|/ref|><|det|>[[100, 100, 900, 180]]<|/det|>
Top paragraph.

<|ref|>image<|/ref|><|det|>[[200, 300, 800, 600]]<|/det|>
<|ref|>image_caption<|/ref|><|det|>[[220, 620, 780, 680]]<|/det|>
<center>Figure 1. Caption text.</center>
"""
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=grounded_output))]
        )
        chat_client = MagicMock()
        chat_client.chat.completions.create.return_value = mock_response
        client = GemmaOCRClient(
            client=chat_client,
            model="gemma4:26b",
        )

        image = Image.new("RGB", (1000, 1600), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((200, 480, 800, 960), fill="black")
        output_dir = PROJECT_ROOT / "tests" / "data" / "output" / "gemma_images"

        result = client.recognize_page(
            OCRPageRequest(
                page_number=1,
                image=image,
                image_output_dir=str(output_dir),
            )
        )

        self.assertEqual(result.raw_text, grounded_output)
        self.assertIn("![](images/1_0.jpg)", result.markdown)
        self.assertTrue((output_dir / "1_0.jpg").exists())


if __name__ == "__main__":
    unittest.main()
