#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$OUTPUT_DIR/../../.." && pwd)"

source "$WORKSPACE_ROOT/scripts/vl-repro-common.sh"

require_command ffmpeg

WORK_DIR="$WORKSPACE_ROOT/work/vl-smoke-test"
VIDEO="$WORK_DIR/testsrc.mp4"
FRAME_DIR="$WORK_DIR/qwen25-vl-f16/frames"

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"

if [[ ! -f "$VIDEO" ]]; then
  ffmpeg \
    -hide_banner \
    -loglevel error \
    -f lavfi \
    -i "testsrc=duration=2:size=320x180:rate=1" \
    -pix_fmt yuv420p \
    "$VIDEO"
fi

ensure_vl_server

"$WORKSPACE_ROOT/scripts/caption-video.py" "$VIDEO" \
  --interval 1 \
  --out-dir "$OUTPUT_DIR" \
  --frame-dir "$FRAME_DIR" \
  --endpoint "$VL_ENDPOINT" \
  --model "$VL_MODEL_ALIAS" \
  --language Japanese

if grep -q "@@@" "$OUTPUT_DIR/captions.md" "$OUTPUT_DIR/captions.json"; then
  echo "Bad caption marker found in smoke-test output." >&2
  exit 1
fi

echo "Smoke-test captions are in: $OUTPUT_DIR"
