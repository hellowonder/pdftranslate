#!/bin/bash

while true; do
    VLLM_SERVER_DEV_MODE=1 ./.vllm-env/bin/vllm serve \
        datalab-to/chandra-ocr-2  \
        --enable-sleep-mode   \
        --no-enable-prefix-caching --mm-processor-cache-gb 0 \
        --port 8000 \
        --served-model-name chandra
done

