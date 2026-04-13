import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from render_units import page_size_to_margin, resolve_page_margin_value  # noqa: E402


class RenderUnitsMarginResolutionTest(unittest.TestCase):
    def test_page_size_to_margin_matches_small_reference_page(self) -> None:
        self.assertEqual(page_size_to_margin(431.0 * 96.0 / 72.0, 649.0 * 96.0 / 72.0), "0.3in")

    def test_page_size_to_margin_matches_large_reference_page(self) -> None:
        self.assertEqual(page_size_to_margin(612.0 * 96.0 / 72.0, 792.0 * 96.0 / 72.0), "0.8in")

    def test_resolve_page_margin_value_prefers_explicit_margin(self) -> None:
        self.assertEqual(resolve_page_margin_value("12mm", 100, 200), "12mm")


if __name__ == "__main__":
    unittest.main()
