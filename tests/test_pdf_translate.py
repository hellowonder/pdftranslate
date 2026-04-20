import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from pdf_translate import (  # noqa: E402
    build_output_paths,
    main,
    parse_args,
    parse_stage_selection,
    prepare_output_paths,
    resolve_target_page_numbers,
)


class PdfTranslateStageArtifactsTest(unittest.TestCase):
    def test_build_output_paths_places_final_pdfs_in_output_root(self) -> None:
        output_paths = build_output_paths("tests/data/one_page.pdf", "/tmp/pdftranslate-output")

        self.assertEqual(output_paths["original_pdf"], "/tmp/pdftranslate-output/one_page_original.pdf")
        self.assertEqual(output_paths["translated_pdf"], "/tmp/pdftranslate-output/one_page_cn.pdf")
        self.assertEqual(output_paths["interleaved_pdf"], "/tmp/pdftranslate-output/one_page_interleaved.pdf")
        self.assertEqual(output_paths["render_dir"], "/tmp/pdftranslate-output/render")

    def test_prepare_output_paths_normalizes_djvu_input_to_render_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = str(Path(tmpdir) / "book.djvu")
            Path(input_path).write_bytes(b"djvu")

            with patch("pdf_translate.normalize_document_input", return_value="/tmp/render/book.source.pdf") as mocked_normalize:
                output_paths = prepare_output_paths(input_path, tmpdir)

            mocked_normalize.assert_called_once_with(str(Path(input_path).resolve()), output_paths["render_dir"])
            self.assertEqual(output_paths["input_pdf"], "/tmp/render/book.source.pdf")

    def test_parse_args_leaves_margin_unset_by_default(self) -> None:
        argv = [
            "pdf_translate.py",
            "--input",
            "tests/data/one_page.pdf",
            "--output-dir",
            "tests/data/output",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertIsNone(args.margin)
        self.assertEqual(args.font_size, "9.5pt")
        self.assertTrue(args.vllm_sleep)
        self.assertTrue(args.generate_interleave_pdf)
        self.assertFalse(args.generate_translation_only_pdf)
        self.assertEqual(args.translation_base_url, "http://localhost:11434/v1")
        self.assertEqual(args.translation_reasoning_effort, "none")

    def test_parse_args_accepts_translation_only_pdf_output(self) -> None:
        argv = [
            "pdf_translate.py",
            "--input",
            "tests/data/one_page.pdf",
            "--output-dir",
            "tests/data/output",
            "--generate-translation-only-pdf",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertTrue(args.generate_interleave_pdf)
        self.assertTrue(args.generate_translation_only_pdf)

    def test_parse_args_rejects_render_with_no_pdf_outputs(self) -> None:
        argv = [
            "pdf_translate.py",
            "--input",
            "tests/data/one_page.pdf",
            "--output-dir",
            "tests/data/output",
            "--no-generate-interleave-pdf",
        ]

        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            parse_args()

    def test_parse_args_accepts_vllm_sleep(self) -> None:
        argv = [
            "pdf_translate.py",
            "--input",
            "tests/data/one_page.pdf",
            "--output-dir",
            "tests/data/output",
            "--vllm-sleep",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertTrue(args.vllm_sleep)

    def test_parse_args_accepts_translation_base_url(self) -> None:
        argv = [
            "pdf_translate.py",
            "--input",
            "tests/data/one_page.pdf",
            "--output-dir",
            "tests/data/output",
            "--translation-base-url",
            "http://localhost:9999/v1",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.translation_base_url, "http://localhost:9999/v1")

    def test_parse_args_accepts_translation_scope(self) -> None:
        argv = [
            "pdf_translate.py",
            "--input",
            "tests/data/one_page.pdf",
            "--output-dir",
            "tests/data/output",
            "--translation-scope",
            "page",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.translation_scope, "page")

    def test_parse_args_rejects_item_annotation_with_page_scope(self) -> None:
        argv = [
            "pdf_translate.py",
            "--input",
            "tests/data/one_page.pdf",
            "--output-dir",
            "tests/data/output",
            "--annotation-mode",
            "item",
            "--translation-scope",
            "page",
        ]

        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            parse_args()

    def test_resolve_target_page_numbers_uses_generic_document_page_count(self) -> None:
        args = SimpleNamespace(pages="1-2")
        pdf_reader = MagicMock()
        pdf_reader.pages = [object(), object(), object(), object(), object()]
        page_numbers = resolve_target_page_numbers(args, pdf_reader)
        self.assertEqual(page_numbers, [1, 2])

    def test_parse_stage_selection_validates_and_orders(self) -> None:
        self.assertEqual(parse_stage_selection("render,ocr"), ["ocr", "render"])

    def test_parse_stage_selection_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            parse_stage_selection("ocr,invalid")

    def test_main_loads_translated_pages_from_files_for_render_stage(self) -> None:
        args = SimpleNamespace(
            input="in.pdf",
            output_dir="out",
            clear=False,
            selected_stages=["ocr", "translate", "render"],
        )
        output_paths = {"input_pdf": "normalized.pdf", "ocr_dir": "ocr", "translate_dir": "translate"}
        pdf_reader = MagicMock()

        with patch("pdf_translate.parse_args", return_value=args), patch(
            "pdf_translate.prepare_output_paths", return_value=output_paths
        ), patch("pdf_translate.PdfReader", return_value=pdf_reader), patch(
            "pdf_translate.resolve_target_page_numbers", return_value=[1, 2]
        ), patch(
            "pdf_translate.run_ocr_stage"
        ) as mocked_ocr_stage, patch("pdf_translate.run_translate_stage") as mocked_translate_stage, patch(
            "pdf_translate.run_render_stage"
        ) as mocked_render:
            main()

        mocked_ocr_stage.assert_called_once_with(args, output_paths, [1, 2])
        mocked_translate_stage.assert_called_once_with(args, output_paths, [1, 2])
        mocked_render.assert_called_once_with(args, pdf_reader, output_paths, [1, 2])


if __name__ == "__main__":
    unittest.main()
