#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$WORKSPACE_ROOT/.." && pwd)"

MODEL="$PROJECT_ROOT/models/Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf"
MMPROJ="$PROJECT_ROOT/models/Qwen2.5-VL-7B-Instruct-abliterated.mmproj-f16.gguf"

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
  --image-min-tokens 1024 \
  -c 16384 \
  -np 1 \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 8081 \
  --no-webui
