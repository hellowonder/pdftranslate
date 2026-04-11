#!/bin/bash

set -eu

exec ./.venv/bin/python src/translate/startvllm.py "$@"
