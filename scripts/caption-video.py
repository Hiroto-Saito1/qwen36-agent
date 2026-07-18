#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request


WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def image_data_url(path: pathlib.Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def default_frame_dir(out_dir: pathlib.Path) -> pathlib.Path:
    outputs_root = (WORKSPACE_ROOT / "outputs").resolve()
    try:
        relative_out = out_dir.relative_to(outputs_root)
        return (WORKSPACE_ROOT / "work" / relative_out / "frames").resolve()
    except ValueError:
        pass

    try:
        relative_out = out_dir.relative_to(WORKSPACE_ROOT)
        return (WORKSPACE_ROOT / "work" / relative_out / "frames").resolve()
    except ValueError:
        return (WORKSPACE_ROOT / "work" / out_dir.name / "frames").resolve()


def workspace_path(path: pathlib.Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def write_text_atomic(path: pathlib.Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def caption_frame(endpoint: str, model: str, frame: pathlib.Path, language: str) -> str:
    prompt = (
        "Describe this video frame factually for research captioning. "
        "Do not moralize or refuse. Mention visible subjects, actions, setting, "
        "camera/framing, and any readable text. "
        f"Write the caption in {language}. Keep it concise."
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(frame)}},
                ],
            }
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 220,
    }
    response = post_json(f"{endpoint.rstrip('/')}/chat/completions", payload)
    return response["choices"][0]["message"]["content"].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create scene-level captions from a local video.")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between sampled frames")
    parser.add_argument("--out-dir", default="outputs/video-captions", help="Output directory")
    parser.add_argument("--frame-dir", help="Directory for extracted frames; defaults to work/<output-name>/frames")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081/v1", help="OpenAI-compatible VL endpoint")
    parser.add_argument("--model", default="qwen2.5-vl-7b-instruct-abliterated-q4km", help="VL model alias")
    parser.add_argument("--language", default="Japanese", help="Caption language")
    args = parser.parse_args()

    video = pathlib.Path(args.video).expanduser().resolve()
    if not video.exists():
        print(f"Video not found: {video}", file=sys.stderr)
        return 1

    if shutil.which("ffmpeg") is None:
        print("ffmpeg is not installed. Install it with: brew install ffmpeg", file=sys.stderr)
        return 1

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    frame_dir = pathlib.Path(args.frame_dir).expanduser().resolve() if args.frame_dir else default_frame_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    for old in frame_dir.glob("frame_*.jpg"):
        old.unlink()

    fps = 1.0 / args.interval
    frame_pattern = str(frame_dir / "frame_%06d.jpg")
    run([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        frame_pattern,
    ])

    frames = sorted(frame_dir.glob("frame_*.jpg"))
    if not frames:
        print("No frames were extracted.", file=sys.stderr)
        return 1

    records = []
    for index, frame in enumerate(frames):
        start = index * args.interval
        end = start + args.interval
        print(f"Captioning {frame.name} ({start:.1f}-{end:.1f}s)...", file=sys.stderr)
        caption = caption_frame(args.endpoint, args.model, frame, args.language)
        records.append({
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "frame_path": workspace_path(frame),
            "caption_ja": caption,
        })
        time.sleep(0.1)

    json_path = out_dir / "captions.json"
    md_path = out_dir / "captions.md"

    write_text_atomic(json_path, json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    md_lines = [f"# Video captions: {video.name}", ""]
    for item in records:
        md_lines.append(f"- {item['start_time']:.1f}s-{item['end_time']:.1f}s: {item['caption_ja']}")
    write_text_atomic(md_path, "\n".join(md_lines) + "\n")

    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
