#!/usr/bin/env python3
"""
Markdown、数学公式与 PDF 渲染相关的辅助函数。
"""
from __future__ import annotations

import atexit
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

from markdown_it import MarkdownIt
from markdown_katex.extension import tex2html
from markdown_katex import wrapper as katex_wrapper
from pypdf import PdfReader, PdfWriter
from weasyprint import HTML

from render_units import font_size_to_css_value, margin_to_css_value


DEFAULT_FONT_CANDIDATES = [
    str(Path("~/.local/share/fonts/simsun.ttc").expanduser()),
    "/usr/share/fonts/truetype/microsoft/simsun.ttc",
    "/usr/share/fonts/truetype/microsoft/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
]

DEFAULT_KATEX_CSS_PATH = Path(__file__).with_name("katex.min.css")
DEFAULT_KATEX_JS_PATH = Path(__file__).with_name("katex.min.js")
DEFAULT_KATEX_WORKER_PATH = Path(__file__).with_name("katex_worker.js")
DEFAULT_KATEX_FONTS_DIR = Path(__file__).with_name("fonts")
KATEX_FONT_URL_RE = re.compile(r"url\((?P<quote>['\"]?)(?P<path>fonts/[^)\"']+)(?P=quote)\)")
KATEX_NEQ_HTML_RE = re.compile(
    r'<span class="mrel"><span class="mrel"><span class="mord vbox"><span class="thinbox"><span class="rlap">'
    r'<span class="strut"[^>]*></span><span class="inner"><span class="mord"><span class="mrel"></span></span></span>'
    r'<span class="fix"></span></span></span></span></span><span class="mrel">=</span></span>'
)
MATH_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
MATH_BLOCK_BRACKET_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
MATH_INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", re.DOTALL)
MATH_INLINE_PARENS_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
IMAGE_MARKDOWN_PATTERN = re.compile(
    r"!\s*\[(?P<alt>[^\]]*)\]\s*\(\s*(?P<src>[^)]+?)\s*\)",
    re.IGNORECASE,
)
_MARKDOWN_RENDERER: Optional[MarkdownIt] = None
MATH_PATTERN = re.compile(r"(\$\$.+?\$\$|\\\(.+?\\\)|\\\[.+?\\\]|\$(?!\$).+?(?<!\$)\$)", re.DOTALL)
SLOW_MARKDOWN_RENDER_THRESHOLD_SECONDS = 0.5
SLOW_WEASYPRINT_RENDER_THRESHOLD_SECONDS = 1.0
WEASYPRINT_SPLIT_BATCH_PAGE_COUNT = 4
_KATEX_BIN_CMD: Optional[Tuple[str, ...]] = None
_NODE_KATEX_RENDERER: Optional["NodeKatexRenderer"] = None


@dataclass(frozen=True)
class RenderSettings:
    font_path: Optional[str]
    font_size: Union[str, int, float]
    page_width: int
    page_height: int
    margin: str
    image_root: Optional[str]
    image_spacing: int
    katex_css: str = ""
    base_url: Optional[str] = None


class NodeKatexRenderer:
    """
    通过常驻 Node 进程调用仓库内 ``katex.min.js``，避免每个公式启动一次 CLI。
    """

    def __init__(self, node_path: str, worker_path: Path, katex_js_path: Path) -> None:
        self.node_path = node_path
        self.worker_path = worker_path
        self.katex_js_path = katex_js_path
        self._proc: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()
        self._request_id = 0

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()

    def render(self, expr: str, display_mode: bool) -> str:
        with self._lock:
            proc = self._ensure_process()
            request = {
                "id": self._request_id,
                "expr": expr,
                "options": {
                    "displayMode": display_mode,
                    "output": "htmlAndMathml",
                    "throwOnError": False,
                },
            }
            self._request_id += 1

            assert proc.stdin is not None
            assert proc.stdout is not None
            proc.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
            proc.stdin.flush()
            response_line = proc.stdout.readline()
            if not response_line:
                stderr_output = ""
                if proc.stderr is not None:
                    stderr_output = proc.stderr.read().strip()
                self.close()
                raise RuntimeError(f"KaTeX worker exited unexpectedly. {stderr_output}".strip())

            response = json.loads(response_line)
            if response.get("id") != request["id"]:
                raise RuntimeError("KaTeX worker response id mismatch.")
            if "error" in response:
                raise RuntimeError(response["error"])
            return response["html"]

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        self.close()
        self._proc = subprocess.Popen(
            [self.node_path, str(self.worker_path), str(self.katex_js_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=str(self.worker_path.parent),
        )
        return self._proc


def _get_node_katex_renderer() -> Optional[NodeKatexRenderer]:
    global _NODE_KATEX_RENDERER
    if _NODE_KATEX_RENDERER is not None:
        return _NODE_KATEX_RENDERER

    node_path = shutil.which("node")
    if not node_path or not DEFAULT_KATEX_WORKER_PATH.exists() or not DEFAULT_KATEX_JS_PATH.exists():
        return None

    _NODE_KATEX_RENDERER = NodeKatexRenderer(
        node_path=node_path,
        worker_path=DEFAULT_KATEX_WORKER_PATH,
        katex_js_path=DEFAULT_KATEX_JS_PATH,
    )
    return _NODE_KATEX_RENDERER


def _prefer_packaged_katex_binary() -> None:
    """
    优先固定使用 ``markdown_katex`` 自带的 KaTeX 可执行文件，绕开 PATH/npx 探测。
    """
    global _KATEX_BIN_CMD
    if _KATEX_BIN_CMD is not None:
        return

    try:
        pkg_bin_path = katex_wrapper._get_pkg_bin_path()
    except Exception:
        _KATEX_BIN_CMD = tuple(katex_wrapper.get_bin_cmd())
    else:
        _KATEX_BIN_CMD = (str(pkg_bin_path),)

    katex_wrapper.get_bin_cmd = lambda: list(_KATEX_BIN_CMD)


_prefer_packaged_katex_binary()


def _close_node_katex_renderer() -> None:
    renderer = _NODE_KATEX_RENDERER
    if renderer is not None:
        renderer.close()


atexit.register(_close_node_katex_renderer)


def get_markdown_renderer() -> MarkdownIt:
    """
    懒加载一个启用常用功能的 MarkdownIt 渲染器。

    返回:
        MarkdownIt: 可复用的 Markdown 渲染实例。
    """
    global _MARKDOWN_RENDERER
    if _MARKDOWN_RENDERER is None:
        renderer = MarkdownIt("commonmark", {"typographer": True})
        renderer.enable("table")
        renderer.enable("strikethrough")
        renderer.enable("linkify")
        _MARKDOWN_RENDERER = renderer
    return _MARKDOWN_RENDERER


def resolve_font_path(font_path: Optional[str]) -> Optional[str]:
    """
    在用户指定路径和默认候选列表中寻找可用字体。

    参数:
        font_path: 用户传入的字体路径，可为空。
    返回:
        Optional[str]: 找到时返回字体路径，否则返回 ``None``。
    """
    candidates: List[str] = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(DEFAULT_FONT_CANDIDATES)
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def resolve_bold_font_path(font_path: Optional[str]) -> Optional[str]:
    """
    尝试为正文主字体寻找对应的粗体文件。
    """
    if not font_path:
        return None
    path = Path(font_path)
    if not path.exists():
        return None

    replacement_candidates = [
        ("Regular", "Bold"),
        ("regular", "bold"),
        ("-Reg", "-Bold"),
        ("-Roman", "-Bold"),
        ("-Light", "-Bold"),
        ("-Book", "-Bold"),
        ("_Regular", "_Bold"),
    ]
    for source, target in replacement_candidates:
        if source not in path.name:
            continue
        candidate = path.with_name(path.name.replace(source, target, 1))
        if candidate.exists():
            return str(candidate)

    fixed_candidates = {
        "msyh.ttc": path.with_name("msyhbd.ttc"),
        "MSYH.TTC": path.with_name("MSYHBD.TTC"),
        "simsun.ttc": path.with_name("simhei.ttf"),
        "SIMSUN.TTC": path.with_name("SIMHEI.TTF"),
    }
    fixed = fixed_candidates.get(path.name)
    if fixed is not None and fixed.exists():
        return str(fixed)
    return None


def normalize_markdown_image_links(text: str, image_root: Optional[str]) -> str:
    """
    规范化 Markdown 图片链接，但尽量保留相对路径，交由 ``base_url`` / ``image_root`` 解析。

    参数:
        text: 原始 Markdown 文本。
        image_root: 图片根目录；为空时直接返回原文。
    返回:
        str: 替换后的 Markdown 文本。
    """
    if not image_root:
        return text

    def _replace(match: re.Match) -> str:
        alt = match.group("alt")
        src = match.group("src").strip().strip('"').strip("'")
        if os.path.isabs(src):
            return f"![{alt}]({src})"

        normalized_src = src
        if normalized_src.startswith("./"):
            normalized_src = normalized_src[2:]
        if normalized_src.startswith("images/"):
            relative_candidate = normalized_src
            relative_fallback = os.path.basename(normalized_src)
        else:
            relative_candidate = normalized_src
            relative_fallback = os.path.join("images", os.path.basename(normalized_src))

        candidate_path = os.path.join(image_root, relative_candidate)
        fallback_path = os.path.join(image_root, relative_fallback)
        if os.path.exists(candidate_path):
            return f"![{alt}]({relative_candidate})"
        if os.path.exists(fallback_path):
            return f"![{alt}]({relative_fallback})"
        return f"![{alt}]({normalized_src})"

    return IMAGE_MARKDOWN_PATTERN.sub(_replace, text)


@lru_cache(maxsize=1024)
def _katex_render_cached(expr: str, display_mode: bool) -> str:
    """
    缓存 KaTeX 渲染结果，优先走常驻 Node worker，失败后回退到 ``markdown_katex``。
    """
    node_renderer = _get_node_katex_renderer()
    if node_renderer is not None:
        try:
            return node_renderer.render(expr, display_mode)
        except Exception as exc:
            print(f"node katex worker failed, falling back to markdown_katex CLI: {exc}", file=sys.stderr)

    options = {
        "display-mode": display_mode,
        "insert_fonts_css": False,
    }
    return tex2html(expr, options)


def katex_render(expr, display_mode=True):
    """
    使用 ``markdown_katex`` 将 LaTeX 表达式渲染为 HTML。

    参数:
        expr: 公式内容。
        display_mode: 是否按块级公式渲染。
    返回:
        str: 渲染后的 HTML 片段。
    """
    html = _katex_render_cached(expr, display_mode)
    return _normalize_katex_html_for_weasyprint(html)


def _normalize_katex_html_for_weasyprint(html: str) -> str:
    """
    WeasyPrint 对 KaTeX 里某些依赖私有区字形叠加的关系符支持不稳定，
    这里把 ``\\neq`` / ``\\ne`` 这类结构替换成稳定的 Unicode 符号。
    """
    return KATEX_NEQ_HTML_RE.sub('<span class="mrel katex-neq-fallback">≠</span>', html)


def extract_math_placeholders(text: str) -> Tuple[str, Dict[str, str]]:
    """
    将 Markdown 中的数学公式替换为占位符，并返回对应的渲染 HTML。

    参数:
        text: 原始 Markdown 文本。
    返回:
        Tuple[str, Dict[str, str]]:
            第一个值是插入占位符后的文本，第二个值是 ``占位符 -> HTML`` 的映射。
    """
    placeholders: Dict[str, str] = {}
    rendered_html_cache: Dict[Tuple[str, bool], str] = {}
    counter = 0

    def _replace(match: re.Match, display_mode: bool) -> str:
        nonlocal counter
        expr = match.group(1).strip()
        key = f"@@KATEX_{counter}@@"
        counter += 1
        cache_key = (expr, display_mode)
        html = rendered_html_cache.get(cache_key)
        if html is None:
            if katex_render:
                try:
                    html = katex_render(expr, display_mode=display_mode)
                except Exception as exc:
                    html = f"<span class=\"katex-error\">{escape(expr)}<!-- {exc} --></span>"
            else:
                tag = "div" if display_mode else "span"
                cls = "math-display" if display_mode else "math-inline"
                html = f"<{tag} class=\"{cls}\">{escape(expr)}</{tag}>"
            rendered_html_cache[cache_key] = html
        placeholders[key] = html
        return key

    def repl_display(match: re.Match) -> str:
        return _replace(match, True)

    def repl_inline(match: re.Match) -> str:
        return _replace(match, False)

    without_blocks = MATH_BLOCK_RE.sub(repl_display, text)
    without_blocks = MATH_BLOCK_BRACKET_RE.sub(repl_display, without_blocks)
    without_inline = MATH_INLINE_RE.sub(repl_inline, without_blocks)
    without_inline = MATH_INLINE_PARENS_RE.sub(repl_inline, without_inline)
    return without_inline, placeholders


def load_katex_css(css_path: Optional[str]) -> str:
    """
    从显式路径或默认位置读取 ``katex.min.css``。

    参数:
        css_path: 用户指定的 CSS 路径，可为空。
    返回:
        str: CSS 文件内容。
    """
    candidates: List[Path] = []
    if css_path:
        candidates.append(Path(css_path).expanduser())
    candidates.append(DEFAULT_KATEX_CSS_PATH)

    for candidate in candidates:
        if candidate.exists():
            try:
                css_text = candidate.read_text(encoding="utf-8")
                return prepare_katex_css_for_weasyprint(css_text, candidate.parent)
            except OSError as exc:
                raise RuntimeError(f"Failed to read KaTeX CSS at {candidate}: {exc}") from exc

    raise FileNotFoundError(
        "KaTeX CSS was not found. Pass --katex-css-path to a local katex.min.css file "
        f"or place one at {DEFAULT_KATEX_CSS_PATH}."
    )


def prepare_katex_css_for_weasyprint(css_text: str, css_dir: Path) -> str:
    """
    将 KaTeX CSS 调整为更适合 WeasyPrint 的形式：
    1. 将可用的相对字体路径重写为绝对 file URL。
    2. 当 KaTeX 原生字体文件缺失或不完整时，注入本机 TeX 风格字体别名。
    """
    rewritten_css = _rewrite_katex_font_urls(css_text, css_dir)
    if "url(fonts/" not in css_text or "url(fonts/" in rewritten_css:
        return _build_weasyprint_katex_fallback_font_faces() + rewritten_css
    return rewritten_css


def _rewrite_katex_font_urls(css_text: str, css_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        font_path = css_dir / match.group("path")
        if not font_path.exists():
            return match.group(0)
        return f"url('{font_path.resolve().as_uri()}')"

    return KATEX_FONT_URL_RE.sub(replace, css_text)


def _font_face_rule(
    *,
    family: str,
    path: Path,
    style: str = "normal",
    weight: str = "400",
) -> str:
    font_format = {
        ".otf": "opentype",
        ".ttf": "truetype",
        ".woff": "woff",
        ".woff2": "woff2",
    }.get(path.suffix.lower(), "opentype")
    return (
        f"@font-face{{font-family:{family};font-style:{style};font-weight:{weight};"
        f"src:url('{path.resolve().as_uri()}') format('{font_format}');"
        f"font-display:block;}}"
    )


def _existing_font_path(*candidates: str) -> Optional[Path]:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return None


@lru_cache(maxsize=1)
def _build_weasyprint_katex_fallback_font_faces() -> str:
    """
    当原生 KaTeX webfont 不可用时，用系统内可用的 TeX 风格字体为其提供别名。
    """
    rules: List[str] = []

    alias_specs = [
        ("KaTeX_Main", "normal", "400", _existing_font_path("/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf")),
        ("KaTeX_Main", "italic", "400", _existing_font_path("/usr/share/texmf/fonts/opentype/public/lm/lmroman10-italic.otf")),
        ("KaTeX_Main", "normal", "700", _existing_font_path("/usr/share/texmf/fonts/opentype/public/lm/lmroman10-bold.otf")),
        ("KaTeX_Main", "italic", "700", _existing_font_path("/usr/share/texmf/fonts/opentype/public/lm/lmroman10-bolditalic.otf")),
        (
            "KaTeX_Math",
            "italic",
            "400",
            _existing_font_path(
                "/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf",
                "/usr/share/fonts/truetype/dejavu/DejaVuMathTeXGyre.ttf",
            ),
        ),
        (
            "KaTeX_Math",
            "italic",
            "700",
            _existing_font_path(
                "/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf",
                "/usr/share/fonts/truetype/dejavu/DejaVuMathTeXGyre.ttf",
            ),
        ),
        ("KaTeX_SansSerif", "normal", "400", _existing_font_path("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-regular.otf")),
        ("KaTeX_SansSerif", "italic", "400", _existing_font_path("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-oblique.otf")),
        ("KaTeX_SansSerif", "normal", "700", _existing_font_path("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-bold.otf")),
        ("KaTeX_Typewriter", "normal", "400", _existing_font_path("/usr/share/texmf/fonts/opentype/public/lm/lmmono10-regular.otf")),
    ]

    symbol_font = _existing_font_path(
        "/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuMathTeXGyre.ttf",
    )
    if symbol_font is not None:
        for family in ("KaTeX_AMS", "KaTeX_Size1", "KaTeX_Size2", "KaTeX_Size3", "KaTeX_Size4"):
            alias_specs.append((family, "normal", "400", symbol_font))

    for family, style, weight, path in alias_specs:
        if path is None:
            continue
        rules.append(_font_face_rule(family=family, path=path, style=style, weight=weight))

    return "".join(rules)


def markdown_contains_math(text: str) -> bool:
    """
    粗略判断 Markdown 中是否包含数学公式标记。

    参数:
        text: 待检查的 Markdown 文本。
    返回:
        bool: 检测到常见公式定界符时返回 ``True``。
    """
    return bool(text and MATH_PATTERN.search(text))


def _render_weasyprint_html(markdown_text: str, image_root: Optional[str]) -> Tuple[str, bool]:
    """
    将 Markdown 转为 HTML 片段，并返回是否包含数学公式。

    参数:
        markdown_text: 输入 Markdown 文本。
        image_root: 图片根目录。
    返回:
        Tuple[str, bool]: HTML 片段，以及该页是否包含数学公式。
    """
    total_start = perf_counter()
    renderer = get_markdown_renderer()
    normalized_markdown = normalize_markdown_image_links(markdown_text, image_root)
    has_math = markdown_contains_math(normalized_markdown or "")
    math_start = perf_counter()
    markdown_with_math, placeholders = extract_math_placeholders(normalized_markdown or "")
    math_elapsed = perf_counter() - math_start
    markdown_start = perf_counter()
    html_body = renderer.render(markdown_with_math)
    markdown_elapsed = perf_counter() - markdown_start
    replace_start = perf_counter()
    for placeholder, snippet in placeholders.items():
        html_body = html_body.replace(placeholder, snippet)
    replace_elapsed = perf_counter() - replace_start
    total_elapsed = perf_counter() - total_start
    if total_elapsed >= SLOW_MARKDOWN_RENDER_THRESHOLD_SECONDS:
        print(
            (
                "slow markdown render: "
                f"total={total_elapsed:.3f}s "
                f"math={math_elapsed:.3f}s "
                f"markdown={markdown_elapsed:.3f}s "
                f"replace={replace_elapsed:.3f}s "
                f"math_count={len(placeholders)} "
                f"unique_math={len(set(placeholders.values()))}"
            ),
            file=sys.stderr,
        )
    return html_body, has_math


def _build_style_css(
    width: int,
    height: int,
    font_path: Optional[str],
    font_size: Union[str, int, float],
    margin: str,
    image_spacing: int,
    katex_css: str,
    include_katex_css: bool,
) -> str:
    """
    生成 WeasyPrint 渲染所需的公共 CSS。

    参数:
        width: 页面宽度，像素。
        height: 页面高度，像素。
        font_path: 已解析的字体路径。
        font_size: 最终正文字号。
        margin: 页面边距。
        image_spacing: 图片间距。
        katex_css: KaTeX CSS 内容。
        include_katex_css: 是否注入 KaTeX CSS。
    返回:
        str: 内联 CSS 文本。
    """
    font_stack = "'Noto Sans CJK SC', 'PingFang SC', 'Microsoft YaHei', sans-serif"
    font_face_css = ""
    if font_path:
        font_uri = Path(font_path).resolve().as_uri()
        font_face_rules = [
            f"@font-face {{ font-family: 'DocCustomCJK'; src: url('{font_uri}'); font-weight: 400; font-style: normal; }}",
        ]
        bold_font_path = resolve_bold_font_path(font_path)
        if bold_font_path:
            bold_uri = Path(bold_font_path).resolve().as_uri()
            font_face_rules.append(
                f"@font-face {{ font-family: 'DocCustomCJK'; src: url('{bold_uri}'); font-weight: 700; font-style: normal; }}"
            )
        font_face_css = "\n".join(font_face_rules) + "\n"
        font_stack = "'DocCustomCJK', " + font_stack

    image_spacing_px = max(0, image_spacing)
    font_size_css = font_size_to_css_value(font_size)
    katex_block = katex_css if include_katex_css else ""
    margin_css = margin_to_css_value(margin)

    return f"""
    {font_face_css}
    {katex_block}
    @page {{
        size: {width}px {height}px;
        margin: {margin_css};
    }}
    body {{
        font-family: {font_stack};
        font-size: {font_size_css};
        line-height: 1.5;
        color: #111;
    }}
    strong, b {{
        font-weight: 700;
    }}
    img {{
        max-width: 100%;
        height: auto;
        display: block;
        margin: {image_spacing_px}px 0;
    }}
    pre {{
        background: #f6f8fa;
        padding: 8px;
        border-radius: 4px;
        overflow-x: auto;
    }}
    code {{
        font-family: 'JetBrains Mono', 'SFMono-Regular', 'Menlo', monospace;
        font-size: {font_size_css};
    }}
    .katex .katex-neq-fallback {{
        font-family: {font_stack};
        font-style: normal;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 12px;
    }}
    table, th, td {{
        border: 1px solid #ddd;
        padding: 6px;
    }}
    .translated-page {{
        break-after: page;
    }}
    .translated-page:last-child {{
        break-after: auto;
    }}
    blockquote {{
        background-color: #f9f9f9;
        border-left: 4px solid #ddd;
        border-radius: 4px;
    }}
    """


def _log_slow_weasyprint_render(
    *,
    label: str,
    html_build_elapsed: float,
    html_init_elapsed: float,
    render_elapsed: Optional[float] = None,
    write_pdf_elapsed: Optional[float] = None,
    page_count: int = 0,
) -> None:
    total_elapsed = html_build_elapsed + html_init_elapsed
    if render_elapsed is not None:
        total_elapsed += render_elapsed
    if write_pdf_elapsed is not None:
        total_elapsed += write_pdf_elapsed
    if total_elapsed < SLOW_WEASYPRINT_RENDER_THRESHOLD_SECONDS:
        return

    parts = [
        f"slow weasyprint render [{label}]:",
        f"total={total_elapsed:.3f}s",
        f"html_build={html_build_elapsed:.3f}s",
        f"html_init={html_init_elapsed:.3f}s",
    ]
    if render_elapsed is not None:
        parts.append(f"render={render_elapsed:.3f}s")
    if write_pdf_elapsed is not None:
        parts.append(f"write_pdf={write_pdf_elapsed:.3f}s")
    if page_count:
        parts.append(f"pages={page_count}")
    print(" ".join(parts), file=sys.stderr)


def _build_weasyprint_html_document(
    *,
    page_fragments: Sequence[str],
    width: int,
    height: int,
    font_path: Optional[str],
    font_size: Union[str, int, float],
    margin: str,
    image_spacing: int,
    katex_css: str,
    include_katex_css: bool,
) -> str:
    style_css = _build_style_css(
        width=width,
        height=height,
        font_path=font_path,
        font_size=font_size,
        margin=margin,
        image_spacing=image_spacing,
        katex_css=katex_css,
        include_katex_css=include_katex_css,
    )
    return f"""
    <html>
      <head>
        <meta charset="utf-8">
        <style>{style_css}</style>
      </head>
      <body>
        {''.join(page_fragments)}
      </body>
    </html>
    """


def _resolve_render_settings(settings: RenderSettings) -> Tuple[RenderSettings, str]:
    resolved_font = settings.font_path
    if not (resolved_font and os.path.exists(resolved_font)):
        resolved_font = resolve_font_path(resolved_font)
    resolved_settings = RenderSettings(
        font_path=resolved_font,
        font_size=settings.font_size,
        page_width=settings.page_width,
        page_height=settings.page_height,
        margin=settings.margin,
        image_root=settings.image_root,
        image_spacing=settings.image_spacing,
        katex_css=settings.katex_css,
        base_url=settings.base_url,
    )
    return resolved_settings, resolved_settings.base_url or resolved_settings.image_root or os.getcwd()


def _render_markdown_to_html_body(markdown_text: str, settings: RenderSettings) -> Tuple[str, bool]:
    return _render_weasyprint_html(markdown_text, settings.image_root)


def _write_pdf_output(
    pdf_bytes: bytes,
    output_path: Optional[str],
    return_mode: Literal["pdf_bytes", "pages"],
):
    if output_path:
        Path(output_path).write_bytes(pdf_bytes)
        if return_mode == "pdf_bytes":
            return None
    if return_mode == "pages":
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return list(reader.pages)
    return pdf_bytes


def render_markdown_page(
    markdown_text: str,
    settings: RenderSettings,
    output_path: Optional[str] = None,
    return_mode: Literal["pdf_bytes", "pages"] = "pdf_bytes",
):
    """
    渲染单个 Markdown 页面。
    """
    resolved_settings, base_url_value = _resolve_render_settings(settings)
    html_build_start = perf_counter()
    html_body, has_math = _render_markdown_to_html_body(markdown_text, resolved_settings)
    html = _build_weasyprint_html_document(
        page_fragments=[html_body],
        width=resolved_settings.page_width,
        height=resolved_settings.page_height,
        font_path=resolved_settings.font_path,
        font_size=resolved_settings.font_size,
        margin=resolved_settings.margin,
        image_spacing=resolved_settings.image_spacing,
        katex_css=resolved_settings.katex_css,
        include_katex_css=has_math,
    )
    html_build_elapsed = perf_counter() - html_build_start

    html_init_start = perf_counter()
    html_obj = HTML(string=html, base_url=base_url_value)
    html_init_elapsed = perf_counter() - html_init_start

    write_pdf_start = perf_counter()
    pdf_bytes = html_obj.write_pdf()
    write_pdf_elapsed = perf_counter() - write_pdf_start
    _log_slow_weasyprint_render(
        label="single",
        html_build_elapsed=html_build_elapsed,
        html_init_elapsed=html_init_elapsed,
        write_pdf_elapsed=write_pdf_elapsed,
        page_count=1,
    )
    return _write_pdf_output(pdf_bytes, output_path, return_mode)


def render_markdown_pages(
    markdown_pages: Sequence[str],
    settings: RenderSettings,
    *,
    mode: Literal["continuous", "per_markdown"] = "continuous",
    output_path: Optional[str] = None,
    return_mode: Literal["pdf_bytes", "pages"] = "pdf_bytes",
    page_indices: Optional[Sequence[int]] = None,
):
    """
    渲染多个 Markdown 页面。
    """
    resolved_settings, base_url_value = _resolve_render_settings(settings)

    if mode == "continuous":
        html_build_start = perf_counter()
        page_html: List[str] = []
        include_katex_css = False
        for markdown_text in markdown_pages:
            html_body, has_math = _render_markdown_to_html_body(markdown_text, resolved_settings)
            include_katex_css = include_katex_css or has_math
            if html_body.strip():
                page_html.append(html_body)
        html = _build_weasyprint_html_document(
            page_fragments=["\n".join(page_html)],
            width=resolved_settings.page_width,
            height=resolved_settings.page_height,
            font_path=resolved_settings.font_path,
            font_size=resolved_settings.font_size,
            margin=resolved_settings.margin,
            image_spacing=resolved_settings.image_spacing,
            katex_css=resolved_settings.katex_css,
            include_katex_css=include_katex_css,
        )
        html_build_elapsed = perf_counter() - html_build_start

        html_init_start = perf_counter()
        html_obj = HTML(string=html, base_url=base_url_value)
        html_init_elapsed = perf_counter() - html_init_start

        write_pdf_start = perf_counter()
        pdf_bytes = html_obj.write_pdf()
        write_pdf_elapsed = perf_counter() - write_pdf_start
        _log_slow_weasyprint_render(
            label="continuous",
            html_build_elapsed=html_build_elapsed,
            html_init_elapsed=html_init_elapsed,
            write_pdf_elapsed=write_pdf_elapsed,
            page_count=len(markdown_pages),
        )
        return _write_pdf_output(pdf_bytes, output_path, return_mode)

    indices = list(page_indices) if page_indices is not None else list(range(len(markdown_pages)))
    if len(indices) != len(markdown_pages):
        raise ValueError("page_indices length must match markdown_pages length.")

    split_pdfs: List[Tuple[int, bytes]] = []
    for start in range(0, len(indices), WEASYPRINT_SPLIT_BATCH_PAGE_COUNT):
        batch_indices = list(indices[start : start + WEASYPRINT_SPLIT_BATCH_PAGE_COUNT])
        batch_pages = list(markdown_pages[start : start + WEASYPRINT_SPLIT_BATCH_PAGE_COUNT])
        page_fragments: List[str] = []
        include_katex_css = False

        html_build_start = perf_counter()
        for idx, markdown_text in zip(batch_indices, batch_pages):
            html_body, has_math = _render_markdown_to_html_body(markdown_text, resolved_settings)
            include_katex_css = include_katex_css or has_math
            page_fragments.append(
                (
                    f'<section class="translated-page">'
                    f'<div id="page-start-{idx}"></div>{html_body}<div id="page-end-{idx}"></div>'
                    f"</section>"
                )
            )

        html = _build_weasyprint_html_document(
            page_fragments=page_fragments,
            width=resolved_settings.page_width,
            height=resolved_settings.page_height,
            font_path=resolved_settings.font_path,
            font_size=resolved_settings.font_size,
            margin=resolved_settings.margin,
            image_spacing=resolved_settings.image_spacing,
            katex_css=resolved_settings.katex_css,
            include_katex_css=include_katex_css,
        )
        html_build_elapsed = perf_counter() - html_build_start

        html_init_start = perf_counter()
        html_obj = HTML(string=html, base_url=base_url_value)
        html_init_elapsed = perf_counter() - html_init_start

        render_start = perf_counter()
        document = html_obj.render()
        render_elapsed = perf_counter() - render_start

        anchor_pages: Dict[str, int] = {}
        for page_idx, page in enumerate(document.pages):
            for anchor_name in page.anchors:
                anchor_pages[anchor_name] = page_idx

        batch_write_start = perf_counter()
        batch_pdf_bytes = document.write_pdf()
        batch_reader = PdfReader(io.BytesIO(batch_pdf_bytes))
        page_markdown_by_idx = dict(zip(batch_indices, batch_pages))
        for idx in batch_indices:
            start_page = anchor_pages.get(f"page-start-{idx}")
            end_page = anchor_pages.get(f"page-end-{idx}")
            if start_page is None or end_page is None:
                available_anchors = ", ".join(sorted(anchor_pages.keys())) or "<none>"
                print(
                    (
                        "missing rendered page anchors, falling back to single-page render: "
                        f"idx={idx} "
                        f"start_found={start_page is not None} "
                        f"end_found={end_page is not None} "
                        f"batch_indices={batch_indices} "
                        f"available_anchors={available_anchors}"
                    ),
                    file=sys.stderr,
                )
                fallback_pdf_bytes = render_markdown_page(
                    markdown_text=page_markdown_by_idx.get(idx, ""),
                    settings=resolved_settings,
                    return_mode="pdf_bytes",
                )
                split_pdfs.append((idx, fallback_pdf_bytes))
                continue
            writer = PdfWriter()
            for page_number in range(start_page, end_page + 1):
                writer.add_page(batch_reader.pages[page_number])
            buffer = io.BytesIO()
            writer.write(buffer)
            split_pdfs.append((idx, buffer.getvalue()))
        batch_write_elapsed = perf_counter() - batch_write_start
        _log_slow_weasyprint_render(
            label=(
                f"split-batch[{start // WEASYPRINT_SPLIT_BATCH_PAGE_COUNT}]"
                f"[pages {batch_indices[0]}-{batch_indices[-1]}]"
            ),
            html_build_elapsed=html_build_elapsed,
            html_init_elapsed=html_init_elapsed,
            render_elapsed=render_elapsed,
            write_pdf_elapsed=batch_write_elapsed,
            page_count=len(batch_indices),
        )

    if output_path:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, pdf_bytes in split_pdfs:
            output_dir.joinpath(f"page_{idx:04d}.pdf").write_bytes(pdf_bytes)
        if return_mode == "pdf_bytes":
            return None

    if return_mode == "pages":
        return [(idx, list(PdfReader(io.BytesIO(pdf_bytes)).pages)) for idx, pdf_bytes in split_pdfs]
    return split_pdfs


def render_markdown_to_pdf_bytes(
    markdown_text: str,
    width: int,
    height: int,
    font_path: Optional[str],
    font_size: Union[str, int, float],
    margin: str,
    image_root: Optional[str],
    image_spacing: int,
    katex_css: str,
    base_url: Optional[str] = None,
) -> bytes | None:
    settings = RenderSettings(
        font_path=font_path,
        font_size=font_size,
        page_width=width,
        page_height=height,
        margin=margin,
        image_root=image_root,
        image_spacing=image_spacing,
        katex_css=katex_css,
        base_url=base_url,
    )
    return render_markdown_page(markdown_text, settings, return_mode="pdf_bytes")


def render_markdowns_pdf_continuous_bytes(
    markdown_pages: Sequence[str],
    width: int,
    height: int,
    font_path: Optional[str],
    font_size: Union[str, int, float],
    margin: str,
    image_root: Optional[str],
    image_spacing: int,
    katex_css: str,
    base_url: Optional[str] = None,
) -> bytes | None:
    settings = RenderSettings(
        font_path=font_path,
        font_size=font_size,
        page_width=width,
        page_height=height,
        margin=margin,
        image_root=image_root,
        image_spacing=image_spacing,
        katex_css=katex_css,
        base_url=base_url,
    )
    return render_markdown_pages(markdown_pages, settings, mode="continuous", return_mode="pdf_bytes")


def render_translation_pdf_batch_page_bytes(
    indices: Sequence[int],
    markdown_pages: Sequence[str],
    width: int,
    height: int,
    font_path: Optional[str],
    font_size: Union[str, int, float],
    margin: str,
    image_root: Optional[str],
    image_spacing: int,
    katex_css: str,
    base_url: Optional[str] = None,
) -> List[Tuple[int, bytes]]:
    settings = RenderSettings(
        font_path=font_path,
        font_size=font_size,
        page_width=width,
        page_height=height,
        margin=margin,
        image_root=image_root,
        image_spacing=image_spacing,
        katex_css=katex_css,
        base_url=base_url,
    )
    return render_markdown_pages(
        markdown_pages,
        settings,
        mode="per_markdown",
        return_mode="pdf_bytes",
        page_indices=indices,
    )
