import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from document_io import convert_djvu_to_pdf, normalize_document_input  # noqa: E402


class DocumentIOTest(unittest.TestCase):
    def test_convert_djvu_to_pdf_invokes_ddjvu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "book.djvu"
            output_dir = Path(tmpdir) / "render"
            source.write_bytes(b"djvu")

            def _run(cmd, capture_output, text):
                output_path = Path(cmd[-1])
                output_path.write_bytes(b"%PDF-1.4\n")
                result = MagicMock()
                result.returncode = 0
                result.stderr = ""
                result.stdout = ""
                return result

            with patch("document_io.shutil.which", return_value="/usr/bin/ddjvu"), patch(
                "document_io.subprocess.run",
                side_effect=_run,
            ) as mocked_run:
                output = convert_djvu_to_pdf(str(source), str(output_dir))

            self.assertEqual(output, str((output_dir / "book.source.pdf").resolve()))
            mocked_run.assert_called_once()

    def test_normalize_document_input_returns_pdf_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "book.pdf"
            source.write_bytes(b"%PDF-1.4\n")
            output = normalize_document_input(str(source), str(Path(tmpdir) / "render"))
            self.assertEqual(output, str(source.resolve()))


if __name__ == "__main__":
    unittest.main()
