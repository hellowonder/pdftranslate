#!/bin/bash

VLLM_SERVER_DEV_MODE=1 ./.vllm-env/bin/vllm serve 
        datalab-to/chandra-ocr-2  \
        --enable-sleep-mode   \
        --port 8000 \
        --served-model-name chandra