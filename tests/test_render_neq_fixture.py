import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from render_weasyprint import RenderSettings, load_katex_css, render_markdown_page, resolve_font_path  # noqa: E402


class RenderNeqFixtureTest(unittest.TestCase):
    def test_render_neq_sample_pdf(self) -> None:
        fixture_path = PROJECT_ROOT / "tests" / "data" / "neq_sample.md"
        output_dir = PROJECT_ROOT / "tests" / "data" / "output"
        output_path = output_dir / "neq_sample.pdf"

        output_dir.mkdir(parents=True, exist_ok=True)
        markdown_text = fixture_path.read_text(encoding="utf-8")
        settings = RenderSettings(
            font_path=resolve_font_path(None),
            font_size="10.5pt",
            page_width=1240,
            page_height=1754,
            margin="0.5in",
            image_root=str(fixture_path.parent),
            image_spacing=16,
            katex_css=load_katex_css(None),
            base_url=str(fixture_path.parent),
        )

        result = render_markdown_page(
            markdown_text=markdown_text,
            settings=settings,
            output_path=str(output_path),
            return_mode="pdf_bytes",
        )

        self.assertIsNone(result)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
