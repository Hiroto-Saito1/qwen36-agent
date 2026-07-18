# Output index

このディレクトリは、残しておきたい生成結果だけを置く場所です。動画素材、抽出フレーム、途中ファイルは `work/` 側に置きます。

## VL smoke test

- Output: `outputs/vl-smoke-test/qwen25-vl-f16/`
- Reproduce: `outputs/vl-smoke-test/qwen25-vl-f16/reproduce.sh`
- Input/work files: `work/vl-smoke-test/`

## Video event search template

- Template: `outputs/video-event-search-template/`
- Main script: `outputs/video-event-search-template/video_event_search.py`
- Example input: `outputs/video-event-search-template/input.example.json`
- Reproduce: `outputs/video-event-search-template/reproduce.sh`
- Work/cache files: `work/video-event-search/`
- Result JSON: `output.json` (updated after each evaluation while the search is running)
- VL image resizing: the 1-hour template uses a 640px max edge by default, configurable by `search.verification_image_max_edge_pixels`
- Semantic input: write all visual requirements in `condition`; retrieval queries are generated internally and recorded in `config.snapshot.json` / `search-trace.json`
- Retrieval setup: `scripts/setup-vision-env.sh`

## Free video 5-minute event search

- Output: `outputs/free-video-test/bbb-5min-event-search/`
- Input JSON: `outputs/free-video-test/bbb-5min-event-search/input.json`
- Reproduce: `outputs/free-video-test/bbb-5min-event-search/reproduce.sh`
- Input/work files: `work/free-video-test/`
- Search cache files: `work/video-event-search/`
- Result JSON: `outputs/free-video-test/bbb-5min-event-search/output.json`
- Human summary: `outputs/free-video-test/bbb-5min-event-search/event-search.md`

Each output directory contains its own `reproduce.sh`. These scripts use the local Qwen2.5-VL server on `http://127.0.0.1:8081/v1`. If the server is not running, the scripts start it and stop only the server process they started.

Video event search also uses `../vision-env/` for the lightweight retrieval model. Run `scripts/setup-vision-env.sh` once before reproducing event-search outputs.
For high-resolution videos, lower `search.verification_image_max_edge_pixels` to reduce VL verification time, or set it to `null` to send original-resolution frames.
