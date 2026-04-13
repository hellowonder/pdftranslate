import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from ocr_client import OCRPageRequest  # noqa: E402
from ocr_client_chandra import ChandraOCRClient  # noqa: E402


class ChandraOCRClientTest(unittest.TestCase):
    def test_infer_image_uses_openai_chat_completions(self) -> None:
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="<div data-label='Text'>ok</div>"))]
        )
        chat_client = MagicMock()
        chat_client.chat.completions.create.return_value = mock_response
        client = ChandraOCRClient(client=chat_client, model="chandra-ocr")

        raw = client.infer_image(Image.new("RGB", (256, 256), "white"))

        self.assertIn("data-label", raw)
        chat_client.chat.completions.create.assert_called_once()

    def test_build_markdown_from_raw_converts_html_and_saves_images(self) -> None:
        raw_html = (
            '<div data-label="Text" data-bbox="0 0 1000 200"><p>Hello <strong>world</strong>.</p></div>'
            '<div data-label="Figure" data-bbox="100 200 900 900"><img alt="diagram"/></div>'
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            request = OCRPageRequest(
                page_number=1,
                image=Image.new("RGB", (1000, 1000), "white"),
                image_output_dir=tmpdir,
            )
            client = ChandraOCRClient(client=MagicMock(), model="chandra-ocr")

            markdown = client.build_markdown_from_raw(request, raw_html)

            self.assertIn("Hello **world**.", markdown)
            self.assertIn("![diagram](", markdown)
            self.assertTrue(any(Path(tmpdir).iterdir()))

    def test_recognize_page_returns_empty_markdown_for_invalid_output(self) -> None:
        client = ChandraOCRClient(client=MagicMock(), model="chandra-ocr")
        with patch.object(client, "infer_image", return_value="<div></div>"), patch.object(
            client, "build_markdown_from_raw", return_value=""
        ):
            result = client.recognize_page(
                OCRPageRequest(
                    page_number=1,
                    image=Image.new("RGB", (100, 100), "white"),
                    image_output_dir="unused",
                )
            )

        self.assertEqual(result.raw_text, "<div></div>")
        self.assertEqual(result.markdown, "")


if __name__ == "__main__":
    unittest.main()
