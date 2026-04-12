import sys
import tempfile
import unittest
from pathlib import Path

import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from pdf_crop import (  # noqa: E402
    CropMargins,
    canvas_rect_to_pdf_rect,
    margins_to_selection,
    normalize_selection,
    pdf_rect_to_canvas_rect,
    remap_selection,
    save_cropped_pdf,
    selection_from_normalized,
    selection_to_margins,
    selection_to_normalized,
)


class PdfCropTest(unittest.TestCase):
    def test_selection_round_trip_via_normalized_margins(self) -> None:
        base_rect = fitz.Rect(0, 0, 600, 800)
        selection = fitz.Rect(40, 50, 560, 760)
        normalized = selection_to_normalized(selection, base_rect)
        restored = selection_from_normalized(normalized, base_rect)
        self.assertEqual(restored, selection)

    def test_margins_round_trip(self) -> None:
        base_rect = fitz.Rect(0, 0, 400, 500)
        margins = CropMargins(left=20, top=30, right=40, bottom=50)
        selection = margins_to_selection(margins, base_rect)
        self.assertEqual(selection_to_margins(selection, base_rect), margins)

    def test_canvas_pdf_coordinate_conversion(self) -> None:
        pdf_rect = fitz.Rect(10, 20, 110, 220)
        zoom = 1.5
        canvas_rect = pdf_rect_to_canvas_rect(pdf_rect, zoom)
        restored = canvas_rect_to_pdf_rect(canvas_rect, zoom)
        self.assertEqual(restored, pdf_rect)

    def test_remap_selection_preserves_relative_margins(self) -> None:
        source_rect = fitz.Rect(0, 0, 400, 600)
        target_rect = fitz.Rect(0, 0, 800, 900)
        source_selection = fitz.Rect(40, 60, 360, 540)
        remapped = remap_selection(source_selection, source_rect, target_rect)
        self.assertEqual(remapped, fitz.Rect(80, 90, 720, 810))

    def test_normalize_selection_rejects_too_small_selection(self) -> None:
        base_rect = fitz.Rect(0, 0, 100, 100)
        with self.assertRaises(ValueError):
            normalize_selection(fitz.Rect(10, 10, 11, 11), base_rect)

    def test_save_cropped_pdf_updates_cropbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.pdf"
            output_path = Path(tmpdir) / "output.pdf"

            doc = fitz.open()
            page = doc.new_page(width=200, height=300)
            page.insert_text((20, 40), "crop test")
            doc.save(input_path)
            doc.close()

            save_cropped_pdf(str(input_path), str(output_path), {0: fitz.Rect(10, 20, 180, 260)})

            result = fitz.open(output_path)
            try:
                cropbox = result[0].cropbox
                self.assertAlmostEqual(cropbox.x0, 10.0)
                self.assertAlmostEqual(cropbox.y0, 20.0)
                self.assertAlmostEqual(cropbox.x1, 180.0)
                self.assertAlmostEqual(cropbox.y1, 260.0)
            finally:
                result.close()


if __name__ == "__main__":
    unittest.main()
