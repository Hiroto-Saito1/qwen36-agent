#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$WORKSPACE_ROOT/.." && pwd)"

MODEL="$PROJECT_ROOT/models/Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf"
MMPROJ="$PROJECT_ROOT/models/Qwen2.5-VL-7B-Instruct-abliterated.mmproj-f16.gguf"
VL_IMAGE_MIN_TOKENS="${VL_IMAGE_MIN_TOKENS:-256}"
VL_IMAGE_MAX_TOKENS="${VL_IMAGE_MAX_TOKENS:-768}"
VL_CONTEXT_SIZE="${VL_CONTEXT_SIZE:-8192}"

if [[ ! -f "$MODEL" ]]; then
  echo "Vision model not found: $MODEL" >&2
  exit 1
fi

if [[ ! -f "$MMPROJ" ]]; then
  echo "Vision projector not found: $MMPROJ" >&2
  exit 1
fi

exec llama-server \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  --alias qwen2.5-vl-7b-instruct-abliterated-q4km \
  --jinja \
  --image-min-tokens "$VL_IMAGE_MIN_TOKENS" \
  --image-max-tokens "$VL_IMAGE_MAX_TOKENS" \
  -c "$VL_CONTEXT_SIZE" \
  -np 1 \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 8081 \
  --no-webui
