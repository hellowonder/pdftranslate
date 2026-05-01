import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from ebooklib import epub


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_SRC = PROJECT_ROOT / "src" / "translate"
if str(TRANSLATE_SRC) not in sys.path:
    sys.path.insert(0, str(TRANSLATE_SRC))

from epub_translate import write_epub_preserving_raw_html  # noqa: E402


class EpubTranslateWriteTest(unittest.TestCase):
    def test_preserving_writer_keeps_original_head_links(self) -> None:
        original_xhtml = b"""<?xml version='1.0' encoding='utf-8'?>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">
<head>
<link rel="stylesheet" type="text/css" href="stylesheet.css"/>
<title>part0000</title>
</head>
<body>
<p>Hello</p>
</body>
</html>
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            default_path = Path(tmpdir) / "default.epub"
            preserved_path = Path(tmpdir) / "preserved.epub"

            book_default = self._build_book_with_raw_xhtml(original_xhtml)
            book_preserved = self._build_book_with_raw_xhtml(original_xhtml)

            epub.write_epub(str(default_path), book_default)
            write_epub_preserving_raw_html(str(preserved_path), book_preserved)

            default_xhtml = self._read_chapter(default_path, "Text/part0000.xhtml")
            preserved_xhtml = self._read_chapter(preserved_path, "Text/part0000.xhtml")

            self.assertNotIn('href="stylesheet.css"', default_xhtml)
            self.assertIn('href="stylesheet.css"', preserved_xhtml)

    def _build_book_with_raw_xhtml(self, raw_xhtml: bytes) -> epub.EpubBook:
        book = epub.EpubBook()
        book.set_identifier("test-book")
        book.set_title("Test Book")
        book.set_language("en")

        chapter = epub.EpubHtml(uid="part0000", file_name="Text/part0000.xhtml", title="part0000", lang="en")
        # Intentionally populate raw content directly to mirror read_epub(): the XHTML
        # contains a stylesheet link, but EbookLib does not hydrate that back into
        # chapter.links when loading an existing book.
        chapter.set_content(raw_xhtml)
        book.add_item(chapter)
        book.spine = [chapter]
        return book

    def _read_chapter(self, epub_path: Path, chapter_name: str) -> str:
        with zipfile.ZipFile(epub_path) as zf:
            member = next(name for name in zf.namelist() if name.endswith(chapter_name))
            return zf.read(member).decode("utf-8")


if __name__ == "__main__":
    unittest.main()
