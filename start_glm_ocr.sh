#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

exec ./llama.cpp/build/bin/llama-server \
    -m glm_ocr/GLM-OCR-Q8_0.gguf \
    --mmproj glm_ocr/mmproj-GLM-OCR-Q8_0.gguf \
    --host 0.0.0.0 \
    --port 8882 \
    --ctx-size 4096 \
    -ngl -1
