import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from translate_service import configure_openai  # noqa: E402
from ocr_client import DeepseekOCRClient, looks_invalid_ocr_output  # noqa: E402
from ocr_postprocess import process_ocr_page_content  # noqa: E402


class OCRClientIntegrationTest(unittest.TestCase):
    def test_looks_invalid_ocr_output_rejects_low_unique_token_ratio(self) -> None:
        repeated = "word " * 20
        self.assertTrue(looks_invalid_ocr_output(repeated))

    def test_looks_invalid_ocr_output_rejects_long_repeating_segments(self) -> None:
        repeated_segment = "alpha beta gamma delta epsilon zeta"
        text = " ".join([repeated_segment] * 11)
        self.assertTrue(looks_invalid_ocr_output(text))

    def test_infer_uses_openai_compatible_chat_completions(self) -> None:
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="recognized markdown"))]
        )
        chat_client = MagicMock()
        chat_client.chat.completions.create.return_value = mock_response
        client = DeepseekOCRClient(
            client=chat_client,
            model="model",
        )

        client.infer(data_url="data:image/png;base64,abc")

        chat_client.chat.completions.create.assert_called_once()

    def test_infer_retries_until_valid_content_is_returned(self) -> None:
        chat_client = MagicMock()
        chat_client.chat.completions.create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))]),
            RuntimeError("temporary OCR failure"),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="recognized markdown"))]),
        ]
        client = DeepseekOCRClient(
            client=chat_client,
            model="model",
        )

        content = client.infer(data_url="data:image/png;base64,abc")

        self.assertEqual(content, "recognized markdown")
        self.assertEqual(chat_client.chat.completions.create.call_count, 3)

    def test_ocr_fixture_image_to_markdown(self) -> None:
        image_path = PROJECT_ROOT / "tests" / "data" / "test_page_14_page_0001.png"
        if not image_path.exists():
            self.skipTest(f"Missing OCR fixture image: {image_path}")
        output_dir = PROJECT_ROOT / "tests" / "data" / "output"
        output_path = output_dir / "test_page_14_page_0001.md"
        layout_output_path = output_dir / "test_page_14_page_0001.layout.md"
        image_assets_dir = output_dir / "images"

        ocr_base_url = os.environ.get("OCR_BASE_URL", "http://localhost:11434/v1")
        ocr_api_key = os.environ.get("OCR_API_KEY", "ollama")
        client = DeepseekOCRClient(
            client=configure_openai(ocr_base_url, ocr_api_key),
            model=os.environ.get("OCR_MODEL", "deepseek-ocr:3b"),
        )

        with Image.open(image_path) as image:
            from ocr_client import encode_image_data_url_for_ocr  # noqa: E402

            raw_markdown = client.infer(encode_image_data_url_for_ocr(image))
            markdown = process_ocr_page_content(
                raw_markdown,
                image,
                str(image_assets_dir),
                page_number=1,
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        layout_output_path.write_text(raw_markdown, encoding="utf-8")

        self.assertTrue(markdown.strip())
        self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
