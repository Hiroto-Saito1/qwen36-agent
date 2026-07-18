#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$WORKSPACE_ROOT/.." && pwd)"

MODEL="$PROJECT_ROOT/models/Qwen2.5-Coder-14B-Instruct-abliterated-Q4_K_M.gguf"

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  exit 1
fi

exec llama-server \
  -m "$MODEL" \
  --alias qwen2.5-coder-14b-abliterated-q4km \
  --jinja \
  -c 32768 \
  -np 1 \
  -ngl 99 \
  --host 127.0.0.1 \
  --port 8080 \
  --no-webui
