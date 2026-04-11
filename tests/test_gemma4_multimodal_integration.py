import base64
import os
import sys
import unittest
from pathlib import Path

from PIL import Image
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from ocr_postprocess import process_ocr_page_content  # noqa: E402


IMAGE_PATH = PROJECT_ROOT / "tests" / "data" / "one_page.png"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "data" / "output"
OUTPUT_PATH = OUTPUT_DIR / "one_page.gemma4.md"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "one_page.gemma4.images"


def encode_image_as_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class Gemma4MultimodalIntegrationTest(unittest.TestCase):
    def test_convert_one_page_png_to_markdown(self) -> None:
        if not IMAGE_PATH.exists():
            self.skipTest(f"Missing test image: {IMAGE_PATH}")

        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OCR_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OCR_API_KEY")
        model = os.environ.get("OPENAI_MODEL") or os.environ.get("OCR_MODEL") or "gemma4:26b"

        if not base_url:
            self.skipTest("Set OPENAI_BASE_URL or OCR_BASE_URL to run this integration test.")
        if not api_key:
            self.skipTest("Set OPENAI_API_KEY or OCR_API_KEY to run this integration test.")

        client = OpenAI(base_url=base_url, api_key=api_key)
        image_data_url = encode_image_as_data_url(IMAGE_PATH)

        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "<image>\n"
                                "<|grounding|>Convert the document page to markdown. "
                                "Preserve reading order, headings, lists, tables, formulas, "
                                "image regions, and image captions. Output image regions as markdown image links with alt text describing the image content. "
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                    ],
                }
            ],
        )

        grounded_output = (response.choices[0].message.content or "").strip()
        self.assertTrue(grounded_output, "Expected non-empty grounded OCR output from multimodal model.")

        with Image.open(IMAGE_PATH) as image:
            markdown = process_ocr_page_content(
                grounded_output,
                image,
                str(OUTPUT_IMAGE_DIR),
                page_number=1,
            )

        self.assertTrue(markdown, "Expected non-empty Markdown from multimodal model.")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(markdown + "\n", encoding="utf-8")

        self.assertTrue(OUTPUT_PATH.exists())


if __name__ == "__main__":
    unittest.main()
