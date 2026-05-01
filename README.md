# PDF Translate

这是一个个人项目，用来将英文pdf翻译成中文。

当前提供几个入口：

- `src/translate/pdf_translate.py`：PDF OCR + 翻译，输出 Markdown / PDF
- `src/translate/epub_translate.py`：EPUB 翻译
- `src/translate/pdf_crop_tool.py`：交互式 PDF 裁边工具

默认模型配置：

- OCR：本地 vLLM `http://localhost:8000/v1` + `deepseek-ai/DeepSeek-OCR`
- Translation：本地 Ollama `http://localhost:11434/v1` + `gemma4:26b`

当前翻译侧支持两种 profile：

- `generic`：默认 profile，适合普通 OpenAI-compatible chat 模型，继续使用现有 `system + user` prompt 约束。
- `translategemma`：给 `TranslateGemma + vLLM` 这类专用翻译模型使用，改成 `<<<source>>><<<target>>><<<text>>>...` 输入格式。

`translategemma` profile 在未显式覆盖时会自动使用更保守的默认值：

- `--document-type general`
- `--annotation-mode none`
- `--no-latex-repair`

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

2. Translation 服务可以使用默认的 `ollama + gemma4:26b`，也可以使用 `vllm + TranslateGemma`。

默认方案：

- `http://localhost:11434/v1`
- model: `gemma4:26b`

如果使用 `vLLM + TranslateGemma`，推荐先启动一个独立的 OpenAI-compatible 服务，例如：

```bash
./.vllm-env/bin/vllm serve \
  Infomaniak-AI/vllm-translategemma-12b-it \
  --served-model-name translategemma-12b-it \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.85
```

说明：

- 这里使用的是面向 vLLM 兼容封装过的 `TranslateGemma` 模型。
- README 里示例把翻译服务放在 `http://localhost:8001/v1`，避免和 OCR 默认占用的 `8000` 端口冲突。
- 启动后，命令行里把翻译参数改成：
  - `--translation-base-url http://localhost:8001/v1`
  - `--translation-model translategemma-12b-it`
  - `--translation-profile translategemma`

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
- `--translation-profile`
- `--translation-source-lang`
- `--translation-target-lang`
- `--ocr-base-url`
- `--ocr-api-key`
- `--ocr-model`

如果使用 `TranslateGemma + vLLM`，推荐显式传：

```bash
./.venv/bin/python src/translate/pdf_translate.py \
  --input tests/data/one_page.pdf \
  --output-dir tests/data/output/one_page \
  --translation-base-url http://localhost:8000/v1 \
  --translation-model translategemma-12b-it \
  --translation-profile translategemma
```

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
- `--translation-profile`
- `--translation-source-lang`
- `--translation-target-lang`
- `--translation-temperature`
- `--translation-max-chunk-chars`

如果使用 `TranslateGemma + vLLM`，推荐：

```bash
./.venv/bin/python src/translate/epub_translate.py \
  --input /path/to/book.epub \
  --output /path/to/book_interleaved.epub \
  --translation-base-url http://localhost:8001/v1 \
  --translation-model translategemma-12b-it \
  --translation-profile translategemma
```

说明：

- `generic` profile 默认更偏学术文档处理：`document-type=academic`、`annotation-mode=page`、开启 LaTeX repair。
- `translategemma` profile 默认更偏纯翻译：`document-type=general`、关闭 annotation、关闭 LaTeX repair。
- 如果你明确知道文档类型或希望保留/关闭某些后处理，仍然可以显式覆盖这些默认值。
