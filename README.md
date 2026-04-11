# PDF Translate

本项目当前提供两条正式入口：

- `src/translate/pdf_translate.py`：把 PDF 做 OCR、翻译，并输出 Markdown / PDF。
- `src/translate/epub_translate.py`：翻译 EPUB，并回写为新的 EPUB 文件。

两个入口共用 `src/translate/translate_service.py` 中的翻译服务参数，因此都支持通过 OpenAI 兼容接口配置后端模型。当前默认配置是本地 Ollama：

- OCR：`http://localhost:11434/v1` + `deepseek-ocr:3b`
- Translation：`http://localhost:11434/v1` + `gemma4:26b`

## 环境准备

项目要求使用仓库内虚拟环境：

```bash
./.venv/bin/python --version
./.venv/bin/pip --version
```

如果你的翻译或 OCR 服务需要认证，先准备对应环境变量或在命令行里显式传入：

- `--translation-base-url`
- `--translation-api-key`
- `--translation-model`
- `--ocr-base-url`
- `--ocr-api-key`
- `--ocr-model`

## 1. 怎样使用 PDF Translate

PDF 翻译入口是 `src/translate/pdf_translate.py`。

完整流程包括：

1. 把 PDF 页面栅格化并调用 OCR，生成每页 Markdown。
2. 对跨页段落边界做一次 LLM 判断，尽量避免段落被错误切断。
3. 逐页翻译 Markdown。
4. 生成纯译文 PDF，必要时再生成“中文译文 + 原始页面”交织 PDF。

如果你使用默认的本地 Ollama 配置，最小命令不需要再显式传模型参数。

### 最小命令

```bash
./.venv/bin/python src/translate/pdf_translate.py \
  --input tests/data/one_page.pdf \
  --output-dir tests/data/output/one_page
```

### 常用命令

只生成纯译文 PDF：

```bash
./.venv/bin/python src/translate/pdf_translate.py \
  --input /path/to/book.pdf \
  --output-dir /path/to/output/book
```

额外生成“译文页 + 原 PDF 页”交织版：

```bash
./.venv/bin/python src/translate/pdf_translate.py \
  --input /path/to/book.pdf \
  --output-dir /path/to/output/book \
  --generate-interleave-pdf
```

只处理部分页码：

```bash
./.venv/bin/python src/translate/pdf_translate.py \
  --input /path/to/book.pdf \
  --output-dir /path/to/output/book_p10_20 \
  --pages 10-20 \
  --translation-workers 16
```

跳过翻译，只验证 OCR 和渲染链路：

```bash
./.venv/bin/python src/translate/pdf_translate.py \
  --input tests/data/one_page.pdf \
  --output-dir tests/data/output/one_page_no_translation \
  --no-translation
```

### 常用参数

- `--input`：输入 PDF。
- `--output-dir`：输出目录。
- `--pages`：只处理部分页，例如 `1,3,5-7`。
- `--render-engine`：PDF 渲染引擎，支持 `pandoc` 或 `weasyprint`，默认 `pandoc`。
- `--generate-interleave-pdf`：额外生成交织版 PDF。
- `--skip-first-page-translation`：跳过第一页翻译。
- `--translation-workers`：翻译并发数。
- `--ocr-workers`：OCR 并发数。
- `--font-path` / `--font-size`：控制渲染字体；`--font-size` 现在表示最终正文字号，建议显式写单位，如 `10.5pt`。

### PDF 输出内容

运行后，`--output-dir` 下通常会生成：

- `<原文件名>_ocr.json`：OCR 阶段页码元数据。
- `<原文件名>_original.md`：OCR 后并完成跨页合并的原文 Markdown。
- `<原文件名>_cn.md`：逐页中文译文 Markdown。
- `<原文件名>_cn_full.pdf`：纯译文 PDF。
- `<原文件名>_cn.pdf`：交织版 PDF，仅在 `--generate-interleave-pdf` 时生成。
- `images/`：从 PDF 页面提取出的图片资源。
- `ocr/`：逐页 OCR Markdown。
- `ocr/ocr_raw/`：逐页原始 OCR 输出。
- `translate/`：逐页翻译结果。

## 2. 怎样使用 ePub Translate

EPUB 翻译入口是 `src/translate/epub_translate.py`。

它会解析 EPUB spine，抽取可翻译的块级内容，调用翻译模型后再把译文插回文档。支持两种输出：

- 交织版 EPUB：译文块插在原文块前面。
- 中文版 EPUB：通过 `--output-cn` 额外输出，保留更适合中文阅读的版本。

### 最小命令

```bash
./.venv/bin/python src/translate/epub_translate.py \
  --input /path/to/book.epub \
  --output /path/to/book_interleaved.epub
```

### 常用命令

同时输出交织版和中文版：

```bash
./.venv/bin/python src/translate/epub_translate.py \
  --input /path/to/book.epub \
  --output /path/to/book_interleaved.epub \
  --output-cn /path/to/book_cn.epub \
  --overwrite
```

只翻译部分 spine 文档：

```bash
./.venv/bin/python src/translate/epub_translate.py \
  --input /path/to/book.epub \
  --output /path/to/book_part.epub \
  --spine-range 1-3,8 \
  --overwrite
```

### 常用参数

- `--input`：输入 EPUB。
- `--output`：交织版 EPUB 输出路径。
- `--output-cn`：可选，输出中文版 EPUB。
- `--spine-range`：只处理部分 spine 项，例如 `1-3,8`。
- `--overwrite`：允许覆盖已有输出文件。
- `--translation-class`：插入译文节点时附带的 CSS class。
- `--translation-workers`：翻译并发数。

## 翻译服务参数

两个入口都支持这些共享参数：

- `--translation-base-url`
- `--translation-api-key`
- `--translation-model`
- `--translation-temperature`
- `--translation-max-chunk-chars`
- `--translation-workers`

默认翻译配置是本地 Ollama `http://localhost:11434/v1` + `gemma4:26b`。如果后端不是默认配置，再显式提供 base URL、API key 和 model。
