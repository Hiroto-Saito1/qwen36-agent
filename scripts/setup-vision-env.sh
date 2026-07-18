#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$WORKSPACE_ROOT/.." && pwd)"
VISION_ENV="$PROJECT_ROOT/vision-env"
RETRIEVAL_CACHE="$PROJECT_ROOT/models/retrieval"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RETRIEVAL_MODEL="${RETRIEVAL_MODEL:-google/siglip2-base-patch16-224}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ ! -x "$VISION_ENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VISION_ENV"
fi

"$VISION_ENV/bin/python" -m pip install --upgrade pip
"$VISION_ENV/bin/python" -m pip install \
  "torch>=2.8" \
  "transformers>=4.55" \
  pillow \
  safetensors \
  sentencepiece

mkdir -p "$RETRIEVAL_CACHE"

"$VISION_ENV/bin/python" - "$RETRIEVAL_MODEL" "$RETRIEVAL_CACHE" <<'PY'
import sys
from transformers import AutoModel, AutoProcessor

model_id = sys.argv[1]
cache_dir = sys.argv[2]
AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
print(f"retrieval model ready: {model_id}")
PY

echo "vision-env ready: $VISION_ENV"
