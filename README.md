# PDF Translate

这是一个个人项目，用来将英文pdf翻译成中文。

当前提供几个入口：

- `src/translate/pdf_translate.py`：PDF OCR + 翻译，输出 Markdown / PDF
- `src/translate/epub_translate.py`：EPUB 翻译
- `src/translate/pdf_crop_tool.py`：交互式 PDF 裁边工具

默认模型配置：

- OCR：本地 vLLM `http://localhost:8000/v1` + `deepseek-ai/DeepSeek-OCR`
- Translation：本地 Ollama `http://localhost:11434/v1` + `gemma4:26b`

项目要求使用仓库内虚拟环境：

```bash
./.venv/bin/python --version
./.venv/bin/pip --version
```

本地环境准备：

1. OCR 服务需要本地运行 `vllm + deepseek-ai/DeepSeek-OCR`。(测试时目前只能使用这种方式，`ollama + deepseek-ocr:3b` 现在有兼容性问题。)

假设在```.vllm-env```中安装vllm。

```bash
VLLM_SERVER_DEV_MODE=1 ./.vllm-env/bin/vllm serve \
        deepseek-ai/DeepSeek-OCR \
        --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
        --no-enable-prefix-caching --mm-processor-cache-gb 0 \
        --enable-sleep-mode \
        --port 8000
```

建议保留 `--logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor`，否则可能出现 OCR 输出重复或陷入循环的问题。

2. Translation 服务需要本地运行 `ollama + gemma4:26b`。

虽然 README 里的示例使用默认配置，但后台模型和服务地址都可以通过命令行参数覆盖。

## PDF

最小命令：

```bash
./.venv/bin/python src/translate/pdf_translate.py \
  --input tests/data/one_page.pdf \
  --output-dir tests/data/output/one_page
```

常用参数：

- `--pages`：只处理部分页，例如 `1,3,5-7`
- `--no-generate-interleave-pdf`：关闭交织版 PDF 输出
- `--vllm-sleep`：vLLM sleep/wake 控制，默认开启
- `--no-vllm-sleep`：关闭 vLLM sleep/wake 控制
- `--no-translation`：只跑 OCR，不做翻译
- `--translation-workers`：翻译并发数
- `--ocr-workers`：OCR 并发数
- `--font-path` / `--font-size`：控制输出 PDF 字体

如果不使用默认后端，可显式传入：

- `--translation-base-url`
- `--translation-api-key`
- `--translation-model`
- `--ocr-base-url`
- `--ocr-api-key`
- `--ocr-model`

## PDF Crop Tool

用于手动确定 PDF 裁边参数并导出裁边后的 PDF。

最小命令：

```bash
./.venv/bin/python src/translate/pdf_crop_tool.py \
  --input tests/data/one_page.pdf
```

使用方式：

- 在页面预览上拖拽，框出“保留区域”。
- 所有页面共享同一套裁边设置；翻页时会显示当前共享框在该页上的映射结果。
- 可以直接拖动绿色边框的四条边做微调。
- `Export PDF` 导出裁边后的结果，默认文件名为 `*.cropped.pdf`。

## EPUB

最小命令：

```bash
./.venv/bin/python src/translate/epub_translate.py \
  --input /path/to/book.epub \
  --output /path/to/book_interleaved.epub
```

常用参数：

- `--output-cn`：额外输出中文版 EPUB
- `--spine-range`：只处理部分 spine 项
- `--overwrite`：允许覆盖已有输出
- `--translation-workers`：翻译并发数

共享翻译参数：

- `--translation-base-url`
- `--translation-api-key`
- `--translation-model`
- `--translation-temperature`
- `--translation-max-chunk-chars`
