Place KaTeX font files from the official `dist/fonts/` directory here.

Expected filenames include entries such as:

- `KaTeX_Main-Regular.woff2`
- `KaTeX_Main-Italic.woff2`
- `KaTeX_Math-Italic.woff2`
- `KaTeX_Size1-Regular.woff2`

The bundled [katex.min.css](/home/wonder/dev/pdftranslate/src/translate/katex.min.css) expects a sibling `fonts/` directory by default, matching the KaTeX documentation:
https://katex.org/docs/font

When these files are present, `render_weasyprint.py` rewrites the relative `fonts/...` URLs to absolute `file://` paths for WeasyPrint. If they are missing, the renderer falls back to local TeX-style system fonts.
