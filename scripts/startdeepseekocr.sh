#!/bin/bash

VLLM_SERVER_DEV_MODE=1 ./.vllm-env/bin/vllm serve \
        deepseek-ai/DeepSeek-OCR \
        --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
        --no-enable-prefix-caching --mm-processor-cache-gb 0 \
        --enable-sleep-mode \
        --port 8000
        