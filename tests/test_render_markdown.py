import sys
import unittest
import re
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

import render_weasyprint  # noqa: E402


class RenderMarkdownMathOptimizationTest(unittest.TestCase):
    def setUp(self) -> None:
        render_weasyprint._katex_render_cached.cache_clear()
        render_weasyprint._KATEX_BIN_CMD = None
        render_weasyprint._NODE_KATEX_RENDERER = None

    def tearDown(self) -> None:
        render_weasyprint._close_node_katex_renderer()

    def test_extract_math_placeholders_renders_duplicate_formula_once_per_page(self) -> None:
        calls = []

        def fake_render(expr, display_mode=True):
            calls.append((expr, display_mode))
            return f"<span>{expr}:{display_mode}</span>"

        markdown = "Inline $x^2$ again $x^2$ and block $$x^2$$"

        with patch("render_weasyprint.katex_render", side_effect=fake_render):
            _, placeholders = render_weasyprint.extract_math_placeholders(markdown)

        self.assertEqual(calls, [("x^2", True), ("x^2", False)])
        self.assertEqual(len(placeholders), 3)

    def test_katex_render_prefers_node_worker(self) -> None:
        fake_renderer = Mock()
        fake_renderer.render.return_value = "<span>ok</span>"

        with patch("render_weasyprint._get_node_katex_renderer", return_value=fake_renderer), patch(
            "render_weasyprint.tex2html"
        ) as mocked_tex2html:
            html = render_weasyprint.katex_render("x^2", display_mode=False)

        self.assertEqual(html, "<span>ok</span>")
        fake_renderer.render.assert_called_once_with("x^2", False)
        mocked_tex2html.assert_not_called()

    def test_katex_render_falls_back_to_markdown_katex_cli(self) -> None:
        fake_renderer = Mock()
        fake_renderer.render.side_effect = RuntimeError("worker failed")

        with patch("render_weasyprint._get_node_katex_renderer", return_value=fake_renderer), patch(
            "render_weasyprint.tex2html", return_value="<span>cli</span>"
        ) as mocked_tex2html:
            html = render_weasyprint.katex_render("x^2", display_mode=False)

        self.assertEqual(html, "<span>cli</span>")
        mocked_tex2html.assert_called_once_with(
            "x^2",
            {
                "display-mode": False,
                "insert_fonts_css": False,
            },
        )

    def test_katex_render_rewrites_neq_private_use_glyph_for_weasyprint(self) -> None:
        html = render_weasyprint.katex_render(r"x \neq y", display_mode=False)

        self.assertIn("katex-neq-fallback", html)
        self.assertIn("≠", html)
        self.assertNotIn("", html)

    def test_prefer_packaged_katex_binary_over_default_lookup(self) -> None:
        packaged_bin = "/tmp/fake-katex-binary"

        with (
            patch("render_weasyprint.katex_wrapper._get_pkg_bin_path", return_value=packaged_bin),
            patch("render_weasyprint.katex_wrapper.get_bin_cmd", return_value=["npx", "--no-install", "katex"]),
        ):
            render_weasyprint._prefer_packaged_katex_binary()
            resolved = render_weasyprint.katex_wrapper.get_bin_cmd()

        self.assertEqual(render_weasyprint._KATEX_BIN_CMD, (packaged_bin,))
        self.assertEqual(resolved, [packaged_bin])

    def test_resolve_font_path_prefers_local_simsun_by_default(self) -> None:
        local_simsun = str(Path("~/.local/share/fonts/simsun.ttc").expanduser())
        original_exists = render_weasyprint.os.path.exists

        def fake_exists(path):
            if path == local_simsun:
                return True
            return original_exists(path)

        with patch("render_weasyprint.os.path.exists", side_effect=fake_exists):
            resolved = render_weasyprint.resolve_font_path(None)

        self.assertEqual(resolved, local_simsun)

    def test_render_translation_pdf_batch_page_bytes_splits_large_batches(self) -> None:
        render_calls = []
        write_pdf_calls = []

        class FakeRenderedDocument:
            def __init__(self, html_text):
                self._html_text = html_text
                indices = [int(value) for value in re.findall(r'id="page-start-(\d+)"', html_text)]
                self.pages = [SimpleNamespace(anchors={f"page-start-{i}": None, f"page-end-{i}": None}) for i in indices]

            def write_pdf(self):
                write_pdf_calls.append(self._html_text)
                return b"%PDF-1.4\n"

        class FakeHTML:
            def __init__(self, string, base_url):
                self.string = string
                self.base_url = base_url

            def render(self):
                render_calls.append(self.string)
                return FakeRenderedDocument(self.string)

        indices = list(range(10))
        markdown_pages = [f"page {i}" for i in indices]
        fake_reader = SimpleNamespace(pages=[object()] * 8)
        fake_writer_instance = Mock()

        with patch("render_weasyprint._render_weasyprint_html", side_effect=lambda text, _: (f"<p>{text}</p>", False)), patch(
            "render_weasyprint.HTML", FakeHTML
        ), patch("render_weasyprint.PdfReader", return_value=fake_reader), patch(
            "render_weasyprint.PdfWriter", return_value=fake_writer_instance
        ):
            result = render_weasyprint.render_translation_pdf_batch_page_bytes(
                indices=indices,
                markdown_pages=markdown_pages,
                width=600,
                height=800,
                font_path=None,
                font_size=14,
                margin="12px",
                image_root=None,
                image_spacing=8,
                katex_css="",
                base_url="/tmp",
            )

        self.assertEqual(len(result), 10)
        self.assertEqual(len(render_calls), 3)
        self.assertEqual(len(write_pdf_calls), 3)

    def test_render_translation_pdf_batch_page_bytes_falls_back_when_anchor_is_missing(self) -> None:
        class FakeRenderedDocument:
            def __init__(self):
                self.pages = [
                    SimpleNamespace(
                        anchors={
                            "page-start-0": None,
                            "page-end-0": None,
                            "page-start-1": None,
                        }
                    )
                ]

            def write_pdf(self):
                return b"%PDF-batch\n"

        class FakeHTML:
            def __init__(self, string, base_url):
                self.string = string
                self.base_url = base_url

            def render(self):
                return FakeRenderedDocument()

        fake_reader = SimpleNamespace(pages=[object()])

        class FakeWriter:
            def __init__(self):
                self.added_pages = []

            def add_page(self, page):
                self.added_pages.append(page)

            def write(self, buffer):
                buffer.write(b"%PDF-split\n")

        with patch("render_weasyprint._render_weasyprint_html", side_effect=lambda text, _: (f"<p>{text}</p>", False)), patch(
            "render_weasyprint.HTML", FakeHTML
        ), patch("render_weasyprint.PdfReader", return_value=fake_reader), patch(
            "render_weasyprint.PdfWriter", side_effect=lambda: FakeWriter()
        ), patch(
            "render_weasyprint.render_markdown_page", return_value=b"%PDF-fallback\n"
        ) as fallback_render:
            result = render_weasyprint.render_translation_pdf_batch_page_bytes(
                indices=[0, 1],
                markdown_pages=["page 0", "page 1"],
                width=600,
                height=800,
                font_path=None,
                font_size=14,
                margin="12px",
                image_root=None,
                image_spacing=8,
                katex_css="",
                base_url="/tmp",
            )

        self.assertEqual(result, [(0, b"%PDF-split\n"), (1, b"%PDF-fallback\n")])
        fallback_render.assert_called_once()
        self.assertEqual(fallback_render.call_args.kwargs["markdown_text"], "page 1")
        settings = fallback_render.call_args.kwargs["settings"]
        self.assertEqual(settings.page_width, 600)
        self.assertEqual(settings.page_height, 800)

    def test_render_markdown_to_pdf_bytes_returns_pdf_bytes(self) -> None:
        captured = {}

        class FakeHTML:
            def __init__(self, string, base_url):
                captured["html"] = string
                self.string = string
                self.base_url = base_url

            def write_pdf(self):
                return b"%PDF-1.4\n"

        with patch("render_weasyprint._render_weasyprint_html", return_value=("<p>hello</p>", False)), patch(
            "render_weasyprint.HTML", FakeHTML
        ):
            result = render_weasyprint.render_markdown_to_pdf_bytes(
                markdown_text="hello",
                width=600,
                height=800,
                font_path=None,
                font_size="10.5pt",
                margin="12px",
                image_root=None,
                image_spacing=8,
                katex_css="",
                base_url="/tmp",
            )

        self.assertEqual(result, b"%PDF-1.4\n")
        self.assertIn("font-size: 10.5pt;", captured["html"])

    def test_render_markdowns_pdf_continuous_bytes_does_not_force_page_sections(self) -> None:
        captured = {}

        class FakeHTML:
            def __init__(self, string, base_url):
                captured["html"] = string
                self.string = string
                self.base_url = base_url

            def write_pdf(self):
                return b"%PDF-1.4\n"

        with patch("render_weasyprint._render_weasyprint_html", side_effect=lambda text, _: (f"<p>{text}</p>", False)), patch(
            "render_weasyprint.HTML", FakeHTML
        ):
            result = render_weasyprint.render_markdowns_pdf_continuous_bytes(
                markdown_pages=["page 1", "page 2"],
                width=600,
                height=800,
                font_path=None,
                font_size="10.5pt",
                margin="12px",
                image_root=None,
                image_spacing=8,
                katex_css="",
                base_url="/tmp",
            )

        self.assertEqual(result, b"%PDF-1.4\n")
        self.assertIn("<p>page 1</p>\n<p>page 2</p>", captured["html"])
        self.assertNotIn('class="translated-page"', captured["html"])

    def test_prepare_katex_css_rewrites_relative_font_urls_to_absolute_file_urls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            css_dir = Path(tmpdir)
            fonts_dir = css_dir / "fonts"
            fonts_dir.mkdir()
            font_path = fonts_dir / "KaTeX_Main-Regular.woff2"
            font_path.write_bytes(b"font")

            prepared = render_weasyprint.prepare_katex_css_for_weasyprint(
                "@font-face{src:url(fonts/KaTeX_Main-Regular.woff2) format('woff2')}",
                css_dir,
            )

        self.assertIn(font_path.resolve().as_uri(), prepared)
        self.assertNotIn("url(fonts/KaTeX_Main-Regular.woff2)", prepared)

    def test_prepare_katex_css_injects_local_tex_font_aliases_when_katex_fonts_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prepared = render_weasyprint.prepare_katex_css_for_weasyprint(
                ".katex{font:normal 1.21em KaTeX_Main,Times New Roman,serif;}",
                Path(tmpdir),
            )

        self.assertIn("@font-face{font-family:KaTeX_Main", prepared)
        self.assertIn("@font-face{font-family:KaTeX_Math", prepared)
        self.assertIn(".katex{font:normal 1.21em KaTeX_Main,Times New Roman,serif;}", prepared)

    def test_prepare_katex_css_injects_local_aliases_when_only_some_katex_fonts_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            css_dir = Path(tmpdir)
            fonts_dir = css_dir / "fonts"
            fonts_dir.mkdir()
            (fonts_dir / "KaTeX_Main-Regular.woff2").write_bytes(b"font")

            prepared = render_weasyprint.prepare_katex_css_for_weasyprint(
                (
                    "@font-face{src:url(fonts/KaTeX_Main-Regular.woff2) format('woff2')}"
                    "@font-face{src:url(fonts/KaTeX_Math-Italic.woff2) format('woff2')}"
                ),
                css_dir,
            )

        self.assertIn("@font-face{font-family:KaTeX_Main", prepared)
        self.assertIn((fonts_dir / "KaTeX_Main-Regular.woff2").resolve().as_uri(), prepared)
        self.assertIn("url(fonts/KaTeX_Math-Italic.woff2)", prepared)

    def test_normalize_markdown_image_links_keeps_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_root = Path(tmpdir)
            (image_root / "images").mkdir()
            (image_root / "images" / "3_0.jpg").write_bytes(b"img")

            normalized = render_weasyprint.normalize_markdown_image_links("![](./3_0.jpg)", str(image_root))

        self.assertEqual(normalized, "![](images/3_0.jpg)")


if __name__ == "__main__":
    unittest.main()
