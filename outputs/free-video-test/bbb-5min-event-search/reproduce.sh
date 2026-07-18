#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$OUTPUT_DIR/../../.." && pwd)"
PROJECT_ROOT="$(cd "$WORKSPACE_ROOT/.." && pwd)"
VISION_PYTHON="${VISION_PYTHON:-$PROJECT_ROOT/vision-env/bin/python}"

source "$WORKSPACE_ROOT/scripts/vl-repro-common.sh"

require_command curl
require_command ffmpeg
require_command ffprobe

SOURCE_URL="https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4"
WORK_DIR="$WORKSPACE_ROOT/work/free-video-test"
FULL_VIDEO="$WORK_DIR/big_buck_bunny_320x180.mp4"
VIDEO="$WORK_DIR/big_buck_bunny_5min.mp4"
TEMPLATE_DIR="$WORKSPACE_ROOT/outputs/video-event-search-template"

mkdir -p "$WORK_DIR" "$OUTPUT_DIR"

if [[ ! -f "$FULL_VIDEO" ]]; then
  curl --fail --location --output "$FULL_VIDEO.part" "$SOURCE_URL"
  mv "$FULL_VIDEO.part" "$FULL_VIDEO"
fi

if [[ ! -f "$VIDEO" ]]; then
  ffmpeg \
    -hide_banner \
    -loglevel error \
    -i "$FULL_VIDEO" \
    -t 300 \
    -c copy \
    "$VIDEO"
  fi

if [[ ! -x "$VISION_PYTHON" ]]; then
  echo "vision-env Python not found: $VISION_PYTHON" >&2
  echo "Run: $WORKSPACE_ROOT/scripts/setup-vision-env.sh" >&2
  exit 1
fi

ensure_vl_server

"$VISION_PYTHON" "$TEMPLATE_DIR/video_event_search.py" "$OUTPUT_DIR/input.json" \
  --endpoint "$VL_ENDPOINT" \
  --model "$VL_MODEL_ALIAS"

if grep -q "@@@" "$OUTPUT_DIR"/*.md "$OUTPUT_DIR"/*.json 2>/dev/null; then
  echo "Bad marker @@@ found in Big Buck Bunny event-search output." >&2
  exit 1
fi

echo "Big Buck Bunny event search output is in: $OUTPUT_DIR"
