import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from render_runner import LAYOUT_JOB_PAGE_CHUNK_SIZE, PDFRenderRunner  # noqa: E402
from render_weasyprint import RenderSettings  # noqa: E402


class RenderPdfLayoutJobSplitTest(unittest.TestCase):
    def test_build_render_jobs_chunks_same_size_pages(self) -> None:
        builder = PDFRenderRunner(
            settings=RenderSettings(
                font_path=None,
                font_size=14,
                page_width=600,
                page_height=800,
                margin="12px",
                image_root=None,
                image_spacing=8,
                katex_css="body{}",
            ),
            max_workers=8,
        )
        translations = [f"page {i}" for i in range(10)]
        builder.original_page_sizes = [(600.0, 800.0)] * 10

        jobs = builder._build_render_jobs(translations)

        self.assertEqual(len(jobs), 2)
        self.assertEqual(list(jobs[0].indices), list(range(LAYOUT_JOB_PAGE_CHUNK_SIZE)))
        self.assertEqual(list(jobs[1].indices), [8, 9])
        self.assertEqual(jobs[0].settings.page_width, 600)
        self.assertEqual(jobs[0].settings.page_height, 800)

    def test_build_render_jobs_uses_original_page_sizes_without_images(self) -> None:
        builder = PDFRenderRunner(
            settings=RenderSettings(
                font_path=None,
                font_size=14,
                page_width=600,
                page_height=800,
                margin="12px",
                image_root=None,
                image_spacing=8,
                katex_css="body{}",
            ),
            max_workers=8,
            original_page_sizes=[(612.0, 792.0), (700.0, 900.0)],
        )

        jobs = builder._build_render_jobs(["page 1", "page 2"])

        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0].settings.page_width, 612)
        self.assertEqual(jobs[0].settings.page_height, 792)


if __name__ == "__main__":
    unittest.main()
