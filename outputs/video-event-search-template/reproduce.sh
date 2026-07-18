#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$OUTPUT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$WORKSPACE_ROOT/.." && pwd)"
VISION_PYTHON="${VISION_PYTHON:-$PROJECT_ROOT/vision-env/bin/python}"

source "$WORKSPACE_ROOT/scripts/vl-repro-common.sh"

require_command ffmpeg
require_command ffprobe

INPUT_JSON="${1:-$OUTPUT_DIR/input.example.json}"

if [[ ! -x "$VISION_PYTHON" ]]; then
  echo "vision-env Python not found: $VISION_PYTHON" >&2
  echo "Run: $WORKSPACE_ROOT/scripts/setup-vision-env.sh" >&2
  exit 1
fi

ensure_vl_server

"$VISION_PYTHON" "$OUTPUT_DIR/video_event_search.py" "$INPUT_JSON" \
  --endpoint "$VL_ENDPOINT" \
  --model "$VL_MODEL_ALIAS"

if grep -q "@@@" "$OUTPUT_DIR"/*.md "$OUTPUT_DIR"/*.json 2>/dev/null; then
  echo "Bad marker @@@ found in video-event-search-template output." >&2
  exit 1
fi

echo "Video event search output is in: $OUTPUT_DIR"
