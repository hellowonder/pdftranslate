import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from render_stage import collect_original_pdf_metadata, load_render_page_images_step, render_interleaved_pdf, render_stage_outputs_exist  # noqa: E402


class RenderStageDocumentSupportTest(unittest.TestCase):
    def test_collect_original_pdf_metadata_returns_page_objects_for_pdf_and_missing_for_out_of_range(self) -> None:
        pdf_reader = PdfReader(str(PROJECT_ROOT / "tests" / "data" / "one_page.pdf"))
        pages, sizes = collect_original_pdf_metadata(pdf_reader, [1, 2])

        self.assertIsNotNone(pages[0])
        self.assertEqual(sizes[1], (0.0, 0.0))

    def test_render_interleaved_pdf_uses_weasyprint_builder(self) -> None:
        args = SimpleNamespace(
            font_path=None,
            font_size="10.5pt",
            page_width=1240,
            min_page_height=1754,
            margin="0.5in",
            image_spacing=16,
            layout_workers=8,
            katex_css_path=None,
        )
        output_paths = {
            "interleaved_pdf": "output.pdf",
            "translate_images_dir": "translate-images",
        }

        mocked_writer = unittest.mock.Mock()
        with patch("render_stage.build_interleaved_pdf_document", return_value=mocked_writer) as mocked_build, patch(
            "render_stage.resolve_font_path",
            return_value="/tmp/font.ttf",
        ), patch("render_stage.load_katex_css", return_value="katex-css"), patch(
            "builtins.open",
            unittest.mock.mock_open(),
        ):
            render_interleaved_pdf(
                args,
                output_paths,
                translated_pages=["page 1"],
                selected_pdf_pages=[None],
                selected_page_sizes=[(0.0, 0.0)],
            )

        mocked_build.assert_called_once()
        self.assertEqual(mocked_build.call_args.kwargs["translations"], ["page 1"])
        self.assertEqual(mocked_build.call_args.kwargs["original_pdf_pages"], [None])
        self.assertEqual(mocked_build.call_args.kwargs["original_page_sizes"], [(0.0, 0.0)])
        self.assertEqual(mocked_build.call_args.kwargs["max_workers"], 8)
        settings = mocked_build.call_args.kwargs["settings"]
        self.assertEqual(settings.font_path, "/tmp/font.ttf")
        self.assertEqual(settings.font_size, "10.5pt")
        self.assertEqual(settings.page_width, 1240)
        self.assertEqual(settings.page_height, 1754)
        self.assertEqual(settings.margin, "0.5in")
        self.assertEqual(settings.image_root, "translate-images")
        self.assertEqual(settings.image_spacing, 16)
        self.assertEqual(settings.katex_css, "katex-css")
        mocked_writer.write.assert_called_once()

    def test_render_interleaved_pdf_uses_pdf_pages_when_available(self) -> None:
        args = SimpleNamespace(
            font_path=None,
            font_size="10.5pt",
            page_width=1240,
            min_page_height=1754,
            margin="0.5in",
            image_spacing=16,
            layout_workers=8,
            katex_css_path=None,
        )
        output_paths = {
            "interleaved_pdf": "output.pdf",
            "translate_images_dir": "translate-images",
        }

        mocked_writer = unittest.mock.Mock()
        with patch("render_stage.build_interleaved_pdf_document", return_value=mocked_writer) as mocked_build, patch(
            "render_stage.resolve_font_path",
            return_value="/tmp/font.ttf",
        ), patch("render_stage.load_katex_css", return_value="katex-css"), patch(
            "builtins.open",
            unittest.mock.mock_open(),
        ):
            render_interleaved_pdf(
                args,
                output_paths,
                translated_pages=["page 1"],
                selected_pdf_pages=["pdf-page-1"],
                selected_page_sizes=[(612.0, 792.0)],
            )

        self.assertEqual(mocked_build.call_args.kwargs["original_pdf_pages"], ["pdf-page-1"])

    def test_render_stage_outputs_exist_only_requires_translated_pdf(self) -> None:
        args = SimpleNamespace(generate_interleave_pdf=False)
        output_paths = {
            "translated_pdf": str(PROJECT_ROOT / "tests" / "data" / "output" / "translated-only.pdf"),
            "interleaved_pdf": str(PROJECT_ROOT / "tests" / "data" / "output" / "interleaved.pdf"),
        }
        Path(output_paths["translated_pdf"]).parent.mkdir(parents=True, exist_ok=True)
        Path(output_paths["translated_pdf"]).write_bytes(b"%PDF-1.4\n")
        try:
            self.assertTrue(render_stage_outputs_exist(args, output_paths))
        finally:
            Path(output_paths["translated_pdf"]).unlink(missing_ok=True)

    def test_load_render_page_images_step_skips_pdf_images_for_weasyprint(self) -> None:
        args = SimpleNamespace(generate_interleave_pdf=True)
        output_paths = {
            "input_pdf": str(PROJECT_ROOT / "tests" / "data" / "one_page.pdf"),
            "translated_pdf": str(PROJECT_ROOT / "tests" / "data" / "output" / "translated-only.pdf"),
            "interleaved_pdf": str(PROJECT_ROOT / "tests" / "data" / "output" / "interleaved.pdf"),
        }

        with patch("render_stage.pdf_to_images_high_quality") as mocked_loader:
            result = load_render_page_images_step(args, output_paths, [1])

        self.assertEqual(result, [])
        mocked_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
