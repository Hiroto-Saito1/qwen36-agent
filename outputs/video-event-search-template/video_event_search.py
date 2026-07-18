#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import mimetypes
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from typing import Any, Optional


SCRIPT_VERSION = "2026-07-18.20"
QUERY_COMPILER_VERSION = "condition-only-v1"
DEFAULT_ENDPOINT = "http://127.0.0.1:8081/v1"
DEFAULT_MODEL = "qwen2.5-vl-7b-instruct-abliterated-q4km"
DEFAULT_RETRIEVAL_MODEL = "google/siglip2-base-patch16-224"
END_GUARD_SECONDS = 0.25
SEARCH_STRATEGIES = {"retrieve_verify"}
EVENT_PHASES = {"match", "near_miss", "unrelated", "uncertain"}
GENERATED_OUTPUT_FILES = (
    "output.json",
    "captions.json",
    "captions.md",
    "event-search.md",
    "search-trace.json",
    "config.snapshot.json",
    "event-search.json",
)


class ConfigError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class SearchConfig:
    config_path: pathlib.Path
    base_dir: pathlib.Path
    video_path: pathlib.Path
    output_dir: pathlib.Path
    condition: str
    language: str
    endpoint: str
    model: str
    strategy: str
    scan_interval: float
    score_threshold: float
    minimum_event_duration: float
    local_scan_interval: float
    boundary_tolerance: float
    retrieval_model: str
    retrieval_cache_dir: pathlib.Path
    retrieval_content_filter: str
    candidate_padding: float
    minimum_candidate_windows: int
    max_candidate_windows: int
    minimum_positive_samples: int
    retrieval_batch_size: int
    retrieval_queries: tuple[str, ...]
    query_plan: dict[str, Any]
    verification_image_max_edge_pixels: Optional[int]
    max_evaluations: Optional[int]
    request_retries: int
    request_timeout: int
    context_seconds: float
    temperature: float
    max_tokens: int
    canonical: dict[str, Any]


def find_workspace_root(start: pathlib.Path) -> pathlib.Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "scripts" / "vl-repro-common.sh").exists():
            return candidate
    return pathlib.Path(__file__).resolve().parents[2]


WORKSPACE_ROOT = find_workspace_root(pathlib.Path(__file__).resolve())
PROJECT_ROOT = WORKSPACE_ROOT.parent


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise ConfigError(f"Required command not found: {command}")


def run_capture(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def run_quiet(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Input JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Input JSON is invalid: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("Input JSON must be an object.")
    return data


def resolve_path(value: str, base_dir: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def require_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' must be a non-empty string.")
    return value.strip()


def optional_text(data: dict[str, Any], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' must be a non-empty string.")
    return value.strip()


def optional_float(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{key}' must be a number.") from exc
    return result


def optional_int_or_null(data: dict[str, Any], key: str) -> Optional[int]:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ConfigError(f"'{key}' must be an integer or null.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{key}' must be an integer or null.") from exc
    if result <= 0:
        raise ConfigError(f"'{key}' must be positive when specified.")
    return result


def optional_int_or_null_default(data: dict[str, Any], key: str, default: Optional[int]) -> Optional[int]:
    if key not in data:
        return default
    return optional_int_or_null(data, key)


def optional_int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool):
        raise ConfigError(f"'{key}' must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{key}' must be an integer.") from exc
    if result <= 0:
        raise ConfigError(f"'{key}' must be positive.")
    return result


def optional_text_list(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key, [])
    if value is None:
        value = []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ConfigError(f"'{key}' must be a list of non-empty strings when specified.")
    return tuple(item.strip() for item in value)


def clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    return max(0.0, min(1.0, result))


def even_dimension(value: int) -> int:
    if value <= 2:
        return max(1, value)
    return max(2, value - (value % 2))


def scaled_dimensions(width: int, height: int, max_edge_pixels: Optional[int]) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if max_edge_pixels is None or max(width, height) <= max_edge_pixels:
        return width, height
    scale = max_edge_pixels / float(max(width, height))
    return even_dimension(round(width * scale)), even_dimension(round(height * scale))


def project_path(path: pathlib.Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_config(path: pathlib.Path, endpoint_override: Optional[str], model_override: Optional[str]) -> SearchConfig:
    config_path = path.expanduser().resolve()
    base_dir = config_path.parent
    data = read_json(config_path)
    search = data.get("search", {})
    if not isinstance(search, dict):
        raise ConfigError("'search' must be an object when specified.")

    removed_top_level = {"caption_interval_seconds"} & set(data)
    removed_search = {
        "candidate_threshold",
        "match_threshold",
        "fallback_top_intervals",
        "candidate_binary",
        "minimum_interval_seconds",
        "neighbor_expansion_intervals",
    } & set(search)
    if removed_top_level or removed_search:
        removed = ", ".join(sorted([*removed_top_level, *(f"search.{key}" for key in removed_search)]))
        raise ConfigError(f"{removed} are from the old candidate_binary schema. Use search.scan_interval_seconds and search.score_threshold.")

    removed_semantic_search = {
        "query_texts",
        "required_visual_checks",
        "not_required_visual_checks",
        "lexical_match_terms",
        "lexical_caption_terms",
    } & set(search)
    if removed_semantic_search:
        removed = ", ".join(sorted(f"search.{key}" for key in removed_semantic_search))
        raise ConfigError(f"{removed} are no longer supported. Put all event semantics in 'condition'.")

    video_path = resolve_path(require_text(data, "video_path"), base_dir)
    output_dir = resolve_path(require_text(data, "output_directory"), base_dir)
    condition = require_text(data, "condition")
    language = optional_text(data, "language", "Japanese")
    endpoint = endpoint_override or optional_text(data, "endpoint", DEFAULT_ENDPOINT)
    model = model_override or optional_text(data, "model", DEFAULT_MODEL)
    strategy = optional_text(search, "strategy", "retrieve_verify")
    scan_interval = optional_float(search, "scan_interval_seconds", 2.0)
    score_threshold = optional_float(search, "score_threshold", 0.82)
    minimum_event_duration = optional_float(search, "minimum_event_duration_seconds", 5.0)
    local_scan_interval = optional_float(search, "local_scan_interval_seconds", 1.0)
    boundary_tolerance = optional_float(search, "boundary_tolerance_seconds", 0.5)
    retrieval_model = optional_text(search, "retrieval_model", DEFAULT_RETRIEVAL_MODEL)
    retrieval_cache_dir = resolve_path(
        optional_text(search, "retrieval_cache_dir", str(PROJECT_ROOT / "models" / "retrieval")),
        base_dir,
    )
    retrieval_content_filter = optional_text(search, "retrieval_content_filter", "none").lower()
    candidate_padding = optional_float(search, "candidate_padding_seconds", 6.0)
    minimum_candidate_windows = optional_int(search, "minimum_candidate_windows", 3)
    max_candidate_windows = optional_int(search, "max_candidate_windows", 8)
    minimum_positive_samples = optional_int(search, "minimum_positive_samples", 2)
    retrieval_batch_size = optional_int(search, "retrieval_batch_size", 32)
    max_evaluations = optional_int_or_null(search, "max_evaluations")
    request_retries = optional_int(search, "request_retries", 2)
    request_timeout = optional_int(search, "request_timeout_seconds", 300)
    context_seconds = optional_float(search, "context_seconds", 1.0)
    temperature = optional_float(search, "temperature", 0.1)
    max_tokens = optional_int(search, "max_tokens", 320)

    verification_image_max_edge_pixels = optional_int_or_null_default(search, "verification_image_max_edge_pixels", 640)

    if not video_path.exists():
        raise ConfigError(f"Video not found: {video_path}")
    if strategy not in SEARCH_STRATEGIES:
        allowed = ", ".join(sorted(SEARCH_STRATEGIES))
        raise ConfigError(f"'search.strategy' must be one of: {allowed}.")
    if scan_interval <= 0:
        raise ConfigError("'search.scan_interval_seconds' must be greater than 0.")
    if not 0.0 <= score_threshold <= 1.0:
        raise ConfigError("'search.score_threshold' must be between 0 and 1.")
    if minimum_event_duration <= 0:
        raise ConfigError("'search.minimum_event_duration_seconds' must be greater than 0.")
    if local_scan_interval <= 0:
        raise ConfigError("'search.local_scan_interval_seconds' must be greater than 0.")
    if boundary_tolerance <= 0:
        raise ConfigError("'search.boundary_tolerance_seconds' must be greater than 0.")
    if candidate_padding < 0:
        raise ConfigError("'search.candidate_padding_seconds' must be greater than or equal to 0.")
    if max_candidate_windows < minimum_candidate_windows:
        raise ConfigError("'search.max_candidate_windows' must be greater than or equal to search.minimum_candidate_windows.")
    if retrieval_content_filter != "none":
        raise ConfigError("'search.retrieval_content_filter' must be 'none'. This template does not add content filters.")
    if context_seconds < 0:
        raise ConfigError("'search.context_seconds' must be greater than or equal to 0.")
    if verification_image_max_edge_pixels is not None and verification_image_max_edge_pixels < 64:
        raise ConfigError("'search.verification_image_max_edge_pixels' must be at least 64, or null.")

    canonical = {
        "video_path": project_path(video_path),
        "output_directory": project_path(output_dir),
        "condition": condition,
        "language": language,
        "model": model,
        "search": {
            "strategy": strategy,
            "scan_interval_seconds": scan_interval,
            "score_threshold": score_threshold,
            "minimum_event_duration_seconds": minimum_event_duration,
            "local_scan_interval_seconds": local_scan_interval,
            "boundary_tolerance_seconds": boundary_tolerance,
            "retrieval_model": retrieval_model,
            "retrieval_cache_dir": project_path(retrieval_cache_dir),
            "retrieval_content_filter": retrieval_content_filter,
            "candidate_padding_seconds": candidate_padding,
            "minimum_candidate_windows": minimum_candidate_windows,
            "max_candidate_windows": max_candidate_windows,
            "minimum_positive_samples": minimum_positive_samples,
            "retrieval_batch_size": retrieval_batch_size,
            "retrieval_query_plan": {},
            "verification_image_max_edge_pixels": verification_image_max_edge_pixels,
            "max_evaluations": max_evaluations,
            "request_retries": request_retries,
            "request_timeout_seconds": request_timeout,
            "context_seconds": context_seconds,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    }

    return SearchConfig(
        config_path=config_path,
        base_dir=base_dir,
        video_path=video_path,
        output_dir=output_dir,
        condition=condition,
        language=language,
        endpoint=endpoint,
        model=model,
        strategy=strategy,
        scan_interval=scan_interval,
        score_threshold=score_threshold,
        minimum_event_duration=minimum_event_duration,
        local_scan_interval=local_scan_interval,
        boundary_tolerance=boundary_tolerance,
        retrieval_model=retrieval_model,
        retrieval_cache_dir=retrieval_cache_dir,
        retrieval_content_filter=retrieval_content_filter,
        candidate_padding=candidate_padding,
        minimum_candidate_windows=minimum_candidate_windows,
        max_candidate_windows=max_candidate_windows,
        minimum_positive_samples=minimum_positive_samples,
        retrieval_batch_size=retrieval_batch_size,
        retrieval_queries=(),
        query_plan={},
        verification_image_max_edge_pixels=verification_image_max_edge_pixels,
        max_evaluations=max_evaluations,
        request_retries=request_retries,
        request_timeout=request_timeout,
        context_seconds=context_seconds,
        temperature=temperature,
        max_tokens=max_tokens,
        canonical=canonical,
    )


def write_text_atomic(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def write_json_atomic(path: pathlib.Path, data: Any) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def workspace_path(path: pathlib.Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def output_path(path: pathlib.Path, output_dir: pathlib.Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(output_dir).as_posix()
    except ValueError:
        return workspace_path(resolved)


def prepare_output_dir(cfg: SearchConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for name in GENERATED_OUTPUT_FILES:
        path = cfg.output_dir / name
        if path.exists() or path.is_symlink():
            path.unlink()
    evidence_root = cfg.output_dir / "evidence"
    if evidence_root.exists() or evidence_root.is_symlink():
        if evidence_root.is_dir() and not evidence_root.is_symlink():
            shutil.rmtree(evidence_root)
        else:
            evidence_root.unlink()
    write_json_atomic(cfg.output_dir / "config.snapshot.json", cfg.canonical)


def config_hash(cfg: SearchConfig) -> str:
    stat = cfg.video_path.stat()
    identity = {
        "script_version": SCRIPT_VERSION,
        "video_path": str(cfg.video_path),
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "condition": cfg.condition,
        "language": cfg.language,
        "model": cfg.model,
        "search": cfg.canonical["search"],
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def ffprobe_duration(video_path: pathlib.Path) -> float:
    raw = run_capture([
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ])
    data = json.loads(raw)
    try:
        duration = float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not read video duration: {video_path}") from exc
    if duration <= 0:
        raise RuntimeError(f"Video duration must be positive: {video_path}")
    return duration


def ffprobe_video_dimensions(video_path: pathlib.Path) -> dict[str, int]:
    raw = run_capture([
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ])
    streams = json.loads(raw).get("streams", [])
    if not streams:
        return {}
    stream = streams[0]
    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError):
        return {}
    if width <= 0 or height <= 0:
        return {}
    return {"width": width, "height": height}


def sample_key(time_seconds: float) -> str:
    return f"{int(round(time_seconds * 1000)):010d}"


def normalize_time(time_seconds: float, duration: float) -> float:
    latest = max(0.0, duration - END_GUARD_SECONDS)
    return round(max(0.0, min(time_seconds, latest)), 3)


def scan_times(duration: float, interval: float) -> list[float]:
    final_time = normalize_time(duration, duration)
    values: list[float] = []
    current = 0.0
    while current < final_time:
        values.append(round(current, 3))
        current += interval
    if not values or final_time - values[-1] > min(interval / 2.0, 1.0):
        values.append(final_time)
    return values


def local_scan_times(start: float, end: float, interval: float, duration: float) -> list[float]:
    start = normalize_time(start, duration)
    end = normalize_time(end, duration)
    values: list[float] = []
    current = start
    while current < end - 0.0005:
        values.append(round(current, 3))
        current += interval
    values.append(end)
    return sorted(set(values))


def extract_frame(video_path: pathlib.Path, time_seconds: float, frame_path: pathlib.Path) -> None:
    if frame_path.exists():
        return
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = frame_path.with_name(f".{frame_path.name}.tmp.jpg")
    if tmp_path.exists():
        tmp_path.unlink()
    run_quiet([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{time_seconds:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-pix_fmt",
        "yuvj420p",
        "-y",
        str(tmp_path),
    ])
    if not tmp_path.exists():
        raise RuntimeError(f"ffmpeg did not create a frame at {time_seconds:.3f}s")
    tmp_path.replace(frame_path)


def extract_scan_frames(video_path: pathlib.Path, times: list[float], frame_dir: pathlib.Path) -> list[dict[str, Any]]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for time_seconds in times:
        frame_path = frame_dir / f"scan_{sample_key(time_seconds)}.jpg"
        extract_frame(video_path, time_seconds, frame_path)
        frames.append({"time": time_seconds, "frame_path": frame_path})
    return frames


def image_dimensions(path: pathlib.Path) -> tuple[int, int]:
    raw = run_capture([
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(path),
    ])
    streams = json.loads(raw).get("streams", [])
    if not streams:
        raise RuntimeError(f"Could not read image dimensions: {path}")
    stream = streams[0]
    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not read image dimensions: {path}") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Image dimensions must be positive: {path}")
    return width, height


def resize_frame_for_verification(
    cfg: SearchConfig,
    cache_dir: pathlib.Path,
    source_path: pathlib.Path,
    role: str,
    time_seconds: float,
) -> dict[str, Any]:
    original_width, original_height = image_dimensions(source_path)
    input_width, input_height = scaled_dimensions(original_width, original_height, cfg.verification_image_max_edge_pixels)
    resized = (input_width, input_height) != (original_width, original_height)
    if resized:
        max_edge = cfg.verification_image_max_edge_pixels
        output_dir = cache_dir / "vl-frames"
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / f"{role}_{sample_key(time_seconds)}_max{max_edge}.jpg"
        if not input_path.exists():
            tmp_path = input_path.with_name(f".{input_path.name}.tmp.jpg")
            if tmp_path.exists():
                tmp_path.unlink()
            run_quiet([
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_path),
                "-vf",
                f"scale={input_width}:{input_height}",
                "-q:v",
                "4",
                "-pix_fmt",
                "yuvj420p",
                "-y",
                str(tmp_path),
            ])
            if not tmp_path.exists():
                raise RuntimeError(f"ffmpeg did not create resized frame: {input_path}")
            tmp_path.replace(input_path)
    else:
        input_path = source_path
    return {
        "role": role,
        "time": round(time_seconds, 3),
        "source_frame_path": workspace_path(source_path),
        "model_frame_path": workspace_path(input_path),
        "original_width": original_width,
        "original_height": original_height,
        "model_input_width": input_width,
        "model_input_height": input_height,
        "resized": resized,
        "_model_frame_abs": input_path,
    }


def image_data_url(path: pathlib.Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        decoded = json.loads(stripped)
        if isinstance(decoded, dict):
            return decoded
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            decoded, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return decoded
    raise ValueError("No JSON object found in model response.")


def normalize_compiled_queries(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []

    queries: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        query = " ".join(item.strip().split())
        if not query:
            continue
        query = query[:180].strip()
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= 3:
            break
    return queries


def condition_query_cache_key(cfg: SearchConfig) -> str:
    identity = {
        "compiler_version": QUERY_COMPILER_VERSION,
        "model": cfg.model,
        "condition": cfg.condition,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def condition_query_cache_path(cfg: SearchConfig) -> pathlib.Path:
    return WORKSPACE_ROOT / "work" / "video-event-search" / "query-plans" / f"{condition_query_cache_key(cfg)}.json"


def build_condition_query_prompt(cfg: SearchConfig) -> str:
    return f"""
You compile concise visual retrieval queries for an image-text retrieval model.

The final vision-language verifier will receive the original condition verbatim.
Your queries are only for broad retrieval recall, not for final truth.

Condition:
{cfg.condition}

Return only one valid JSON object:
{{
  "queries": [
    "short English visual search phrase"
  ]
}}

Rules:
- Produce 1 to 3 short English visual phrases.
- Preserve the required subject, action/state, and spatial relation from the condition.
- Do not add objects, colors, actions, or requirements that are not in the condition.
- Do not broaden the condition into a different scene.
- If the condition says something is not required, do not put that item in the query.
- Use positive visual phrases that an image-text retrieval model can match.
""".strip()


def compile_condition_query_plan(cfg: SearchConfig) -> dict[str, Any]:
    cache_path = condition_query_cache_path(cfg)
    warnings: list[str] = []
    condition_sha256 = hashlib.sha256(cfg.condition.encode("utf-8")).hexdigest()

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            queries = normalize_compiled_queries(cached.get("queries"))
            if queries:
                return {
                    "compiler_version": QUERY_COMPILER_VERSION,
                    "condition_sha256": condition_sha256,
                    "queries": queries,
                    "source": "model",
                    "cache_hit": True,
                    "setup_request_count": 0,
                    "fallback_reason": "",
                    "warnings": [],
                }
            warnings.append("cached query plan had no usable queries")
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"ignored unreadable cached query plan: {exc}")

    prompt = build_condition_query_prompt(cfg)
    payload = {
        "model": cfg.model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 180,
    }

    last_error = ""
    setup_request_count = 0
    for attempt in range(1, cfg.request_retries + 2):
        setup_request_count = attempt
        try:
            response = post_json(f"{cfg.endpoint.rstrip('/')}/chat/completions", payload, cfg.request_timeout)
            content = response["choices"][0]["message"]["content"].strip()
            parsed = extract_json_object(content)
            queries = normalize_compiled_queries(parsed.get("queries"))
            if not queries:
                last_error = "model returned an empty or invalid 'queries' list"
                time.sleep(0.1)
                continue
            cache_payload = {
                "compiler_version": QUERY_COMPILER_VERSION,
                "condition_sha256": condition_sha256,
                "model": cfg.model,
                "queries": queries,
                "created_at_unix": round(time.time(), 3),
            }
            try:
                write_json_atomic(cache_path, cache_payload)
            except OSError as exc:
                warnings.append(f"could not write query plan cache: {exc}")
            return {
                "compiler_version": QUERY_COMPILER_VERSION,
                "condition_sha256": condition_sha256,
                "queries": queries,
                "source": "model",
                "cache_hit": False,
                "setup_request_count": setup_request_count,
                "fallback_reason": "",
                "warnings": warnings,
            }
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.1)

    fallback_query = " ".join(cfg.condition.strip().split())
    return {
        "compiler_version": QUERY_COMPILER_VERSION,
        "condition_sha256": condition_sha256,
        "queries": [fallback_query],
        "source": "fallback",
        "cache_hit": False,
        "setup_request_count": setup_request_count,
        "fallback_reason": last_error or "query compiler did not return usable queries",
        "warnings": [*warnings, "using original condition as retrieval query"],
    }


def attach_condition_query_plan(cfg: SearchConfig, query_plan: dict[str, Any]) -> SearchConfig:
    public_plan = {
        "compiler_version": query_plan.get("compiler_version", QUERY_COMPILER_VERSION),
        "condition_sha256": query_plan.get("condition_sha256", ""),
        "queries": list(query_plan.get("queries", [])),
        "source": query_plan.get("source", ""),
        "cache_hit": bool(query_plan.get("cache_hit", False)),
        "setup_request_count": int(query_plan.get("setup_request_count", 0)),
        "fallback_reason": query_plan.get("fallback_reason", ""),
        "warnings": list(query_plan.get("warnings", [])),
    }
    canonical = json.loads(json.dumps(cfg.canonical, ensure_ascii=False))
    canonical["search"]["retrieval_query_plan"] = public_plan
    return dataclasses.replace(
        cfg,
        retrieval_queries=tuple(public_plan["queries"]),
        query_plan=public_plan,
        canonical=canonical,
    )


def redact_config_text(text: str, cfg: SearchConfig) -> str:
    redacted = str(text)
    replacements: list[tuple[str, str]] = [(cfg.condition, "[condition]")]
    replacements.extend((query, "[retrieval_query]") for query in getattr(cfg, "retrieval_queries", ()))
    for term, replacement in replacements:
        term = str(term).strip()
        if not term:
            continue
        redacted = re.sub(re.escape(term), replacement, redacted, flags=re.IGNORECASE)
    return redacted


def import_retrieval_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from PIL import Image
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        setup = WORKSPACE_ROOT / "scripts" / "setup-vision-env.sh"
        raise ConfigError(
            "Retrieval dependencies are not installed. Run "
            f"{setup} and then retry with the vision-env Python."
        ) from exc
    return torch, functional, Image, (AutoModel, AutoProcessor)


def tensor_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def load_retrieval_stack(cfg: SearchConfig) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    torch, functional, image_module, auto_classes = import_retrieval_dependencies()
    AutoModel, AutoProcessor = auto_classes
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    cfg.retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(cfg.retrieval_model, cache_dir=str(cfg.retrieval_cache_dir))
    model = AutoModel.from_pretrained(cfg.retrieval_model, cache_dir=str(cfg.retrieval_cache_dir))
    model.to(device)
    model.eval()
    return {
        "torch": torch,
        "functional": functional,
        "Image": image_module,
        "processor": processor,
        "model": model,
        "device": device,
    }


def call_feature_method(model: Any, method_name: str, inputs: dict[str, Any]) -> Any:
    method = getattr(model, method_name, None)
    if method is None:
        return None
    accepted = {}
    for key in ("input_ids", "attention_mask", "pixel_values"):
        if key in inputs:
            accepted[key] = inputs[key]
    return method(**accepted)


def score_retrieval_frames(cfg: SearchConfig, scan_frames: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stack = load_retrieval_stack(cfg)
    torch = stack["torch"]
    functional = stack["functional"]
    Image = stack["Image"]
    processor = stack["processor"]
    model = stack["model"]
    device = stack["device"]

    retrieval_queries = list(cfg.retrieval_queries or (cfg.condition,))
    text_inputs = processor(text=retrieval_queries, padding=True, return_tensors="pt")
    text_inputs = tensor_to_device(text_inputs, device)
    with torch.no_grad():
        text_features = call_feature_method(model, "get_text_features", text_inputs)
        if text_features is None:
            text_output = model(**text_inputs)
            text_features = getattr(text_output, "text_embeds", None)
            if text_features is None:
                text_features = getattr(text_output, "pooler_output", None)
        if text_features is None:
            raise RuntimeError(f"Could not extract text features from {cfg.retrieval_model}.")
        text_features = functional.normalize(text_features, dim=-1)

    score_rows: list[list[float]] = []
    for start in range(0, len(scan_frames), cfg.retrieval_batch_size):
        batch = scan_frames[start:start + cfg.retrieval_batch_size]
        images = [Image.open(item["frame_path"]).convert("RGB") for item in batch]
        image_inputs = processor(images=images, return_tensors="pt")
        image_inputs = tensor_to_device(image_inputs, device)
        with torch.no_grad():
            image_features = call_feature_method(model, "get_image_features", image_inputs)
            if image_features is None:
                image_output = model(**image_inputs)
                image_features = getattr(image_output, "image_embeds", None)
                if image_features is None:
                    image_features = getattr(image_output, "pooler_output", None)
            if image_features is None:
                raise RuntimeError(f"Could not extract image features from {cfg.retrieval_model}.")
            image_features = functional.normalize(image_features, dim=-1)
            similarities = image_features @ text_features.T
        score_rows.extend([[float(value) for value in row] for row in similarities.detach().cpu().tolist()])

    query_count = len(retrieval_queries)
    query_minimums = []
    query_maximums = []
    for query_index in range(query_count):
        values = [row[query_index] for row in score_rows]
        query_minimums.append(min(values) if values else 0.0)
        query_maximums.append(max(values) if values else 0.0)

    raw_scores: list[float] = []
    likelihood_scores: list[float] = []
    query_indexes: list[int] = []
    for row in score_rows:
        relatives = []
        for query_index, raw_value in enumerate(row):
            spread = query_maximums[query_index] - query_minimums[query_index]
            relative = 0.0 if spread <= 1e-9 else (raw_value - query_minimums[query_index]) / spread
            relatives.append(clamp_score(relative))
        best_query = max(range(query_count), key=lambda index: relatives[index])
        query_indexes.append(best_query)
        raw_scores.append(row[best_query])
        likelihood_scores.append(relatives[best_query])

    ranked_indexes = sorted(range(len(likelihood_scores)), key=lambda index: likelihood_scores[index], reverse=True)
    rank_by_index = {index: rank + 1 for rank, index in enumerate(ranked_indexes)}

    samples = []
    for index, item in enumerate(scan_frames):
        rank = rank_by_index[index]
        samples.append({
            "time": round(float(item["time"]), 3),
            "frame_path": workspace_path(item["frame_path"]),
            "retrieval_raw_score": round(raw_scores[index], 6),
            "retrieval_likelihood_score": round(likelihood_scores[index], 6),
            "retrieval_rank": rank,
            "best_query_index": query_indexes[index],
            "best_query_label": f"retrieval_query[{query_indexes[index] + 1}]",
            "selected_for_verification": False,
            "selection_reason": "",
        })
    metadata = {
        "model": cfg.retrieval_model,
        "device": device,
        "content_filter": cfg.retrieval_content_filter,
        "raw_score_min": round(min(raw_scores) if raw_scores else 0.0, 6),
        "raw_score_max": round(max(raw_scores) if raw_scores else 0.0, 6),
        "score_normalization": "per_query_minmax_then_max",
        "query_raw_score_ranges": [
            {
                "query_label": f"retrieval_query[{index + 1}]",
                "raw_score_min": round(query_minimums[index], 6),
                "raw_score_max": round(query_maximums[index], 6),
            }
            for index in range(query_count)
        ],
        "query_count": len(retrieval_queries),
        "query_labels": [f"retrieval_query[{index + 1}]" for index in range(len(retrieval_queries))],
    }
    return samples, metadata


def window_from_sample(sample: dict[str, Any], cfg: SearchConfig, duration: float) -> dict[str, Any]:
    center = float(sample["time"])
    start = normalize_time(center - cfg.candidate_padding, duration)
    end = normalize_time(center + cfg.candidate_padding, duration)
    if end < start:
        start, end = end, start
    return {
        "start_time": start,
        "end_time": end,
        "peak_time": round(center, 3),
        "peak_retrieval_likelihood_score": float(sample["retrieval_likelihood_score"]),
        "peak_retrieval_raw_score": float(sample["retrieval_raw_score"]),
        "selected_sample_times": [round(center, 3)],
        "selection_reasons": [sample["selection_reason"]],
    }


def merge_candidate_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not windows:
        return []
    windows = sorted(windows, key=lambda item: (item["start_time"], item["end_time"]))
    merged = [dict(windows[0])]
    for window in windows[1:]:
        current = merged[-1]
        if window["start_time"] <= current["end_time"] + 0.0005:
            current["end_time"] = max(current["end_time"], window["end_time"])
            current["selected_sample_times"].extend(window["selected_sample_times"])
            current["selection_reasons"].extend(window["selection_reasons"])
            if window["peak_retrieval_likelihood_score"] > current["peak_retrieval_likelihood_score"]:
                current["peak_time"] = window["peak_time"]
                current["peak_retrieval_likelihood_score"] = window["peak_retrieval_likelihood_score"]
                current["peak_retrieval_raw_score"] = window["peak_retrieval_raw_score"]
        else:
            merged.append(dict(window))
    for index, window in enumerate(merged, start=1):
        window["index"] = index
        window["width_seconds"] = round(window["end_time"] - window["start_time"], 3)
        window["selected_time_count"] = len(set(window["selected_sample_times"]))
        window["selected_sample_times"] = sorted(set(window["selected_sample_times"]))
        window["selection_reasons"] = sorted(set(window["selection_reasons"]))
    return merged


def select_retrieval_windows(cfg: SearchConfig, duration: float, retrieval_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[float, str] = {}
    for sample in retrieval_samples:
        if float(sample["retrieval_likelihood_score"]) >= cfg.score_threshold:
            chosen[float(sample["time"])] = f"retrieval_likelihood_score >= score_threshold ({sample['retrieval_likelihood_score']:.3f} >= {cfg.score_threshold:.3f})"

    ranked = sorted(retrieval_samples, key=lambda item: (-float(item["retrieval_likelihood_score"]), int(item["retrieval_rank"])))
    while len(merge_candidate_windows([
        window_from_sample({**sample, "selection_reason": chosen[float(sample["time"])]}, cfg, duration)
        for sample in retrieval_samples
        if float(sample["time"]) in chosen
    ])) < cfg.minimum_candidate_windows and ranked:
        sample = ranked.pop(0)
        sample_time = float(sample["time"])
        if sample_time in chosen:
            continue
        chosen[sample_time] = (
            f"top retrieval fallback to keep at least {cfg.minimum_candidate_windows} candidate windows "
            f"(score={sample['retrieval_likelihood_score']:.3f})"
        )

    windows = []
    for sample in retrieval_samples:
        sample_time = float(sample["time"])
        if sample_time not in chosen:
            continue
        sample["selected_for_verification"] = True
        sample["selection_reason"] = chosen[sample_time]
        windows.append(window_from_sample(sample, cfg, duration))

    merged = merge_candidate_windows(windows)
    if len(merged) > cfg.max_candidate_windows:
        kept = sorted(merged, key=lambda item: (-float(item["peak_retrieval_likelihood_score"]), float(item["start_time"])))[:cfg.max_candidate_windows]
        kept_keys = {(item["start_time"], item["end_time"]) for item in kept}
        for sample in retrieval_samples:
            sample_time = float(sample["time"])
            if sample_time not in chosen:
                continue
            retained = any(item["start_time"] <= sample_time <= item["end_time"] for item in kept)
            sample["selected_for_verification"] = retained
            if not retained:
                sample["selection_reason"] = "candidate window pruned by max_candidate_windows"
        merged = sorted([item for item in merged if (item["start_time"], item["end_time"]) in kept_keys], key=lambda item: item["start_time"])
        for index, window in enumerate(merged, start=1):
            window["index"] = index
    return merged


def build_verification_prompt(
    cfg: SearchConfig,
    center_time: float,
    before_time: float,
    after_time: float,
    verification_mode: str,
) -> str:
    if verification_mode == "single":
        image_note = f"""
Image:
1. target frame at {center_time:.3f} seconds

Use only this target frame. Do not infer what happened before or after it.
""".strip()
    else:
        image_note = f"""
Images are chronological:
1. before frame at {before_time:.3f} seconds
2. target frame at {center_time:.3f} seconds
3. after frame at {after_time:.3f} seconds

Use before/after frames only as nearby visual context.
""".strip()
    return f"""
You are verifying a candidate frame from a local research video.

{image_note}

Condition to verify at the target frame:
{cfg.condition}

Interpret the condition literally.
The condition text is the only source of truth.
If the condition includes relaxed wording, exclusions, or "not required" details, obey those words exactly.
Do not add hidden requirements such as color, contact, identity, or intent unless the condition explicitly requires them.
First, describe only what is actually visible in the target frame.
Do not copy words from the condition into the caption unless those things are plainly visible.
Do not infer, guess, or complete missing visual facts from the condition text.

Return only one valid JSON object. Do not wrap it in Markdown.
Use {cfg.language} for caption and evidence.

Required schema:
{{
  "neutral_caption": "short factual caption of only what is visible in the target frame",
  "event_phase": "match | near_miss | unrelated | uncertain",
  "confidence_score": 0.0,
  "evidence": "brief positive visual evidence, if any",
  "negative_evidence": "brief reason the condition is not satisfied, if applicable"
}}

event_phase guide:
- match: the condition is clearly true at the target frame.
- near_miss: visually related, but one required subject, action, state, or spatial relation is missing.
- unrelated: not visually relevant to the condition.
- uncertain: enough of the required visual evidence is present, but the truth is genuinely unclear.

Consistency rules:
- If neutral_caption and evidence say the full condition is visible, set event_phase to match.
- If event_phase is near_miss or unrelated, evidence must not restate the full condition as if it were true.
- If event_phase is near_miss, negative_evidence must name the exact missing element from the condition.
- If the only missing detail is minor ambiguity, use uncertain instead of near_miss.

confidence_score is your confidence in the event_phase classification, from 0.0 to 1.0.
""".strip()


def normalize_verification_result(raw: dict[str, Any], cfg: SearchConfig) -> dict[str, Any]:
    phase = str(raw.get("event_phase", "uncertain")).strip().lower()
    if phase == "related":
        phase = "near_miss"
    if phase not in EVENT_PHASES:
        phase = "uncertain"
    score = clamp_score(raw.get("confidence_score", raw.get("confidence", 0.0)))
    caption = raw.get("neutral_caption", raw.get("caption", ""))
    evidence = str(raw.get("evidence", "")).strip()
    negative_evidence = str(raw.get("negative_evidence", "")).strip()
    caption_text = str(caption).strip()
    caption_text = redact_config_text(caption_text, cfg)
    evidence = redact_config_text(evidence, cfg)
    negative_evidence = redact_config_text(negative_evidence, cfg)
    evidence_lower = evidence.lower()
    negative_markers = (
        "確認できない",
        "確認できません",
        "確認されていない",
        "確認されていません",
        "確認されない",
        "見えない",
        "見えていない",
        "見当たらない",
        "存在しない",
        "ではない",
        "not visible",
        "not clearly visible",
        "not present",
        "not near",
        "no butterfly",
        "cannot confirm",
        "can't confirm",
    )
    evidence_has_negation = any(marker in evidence_lower for marker in negative_markers)
    has_affirmative_evidence = bool(evidence) and not evidence_has_negation
    is_match = bool(phase == "match" and score >= cfg.score_threshold and has_affirmative_evidence)
    return {
        "caption": caption_text,
        "neutral_caption": caption_text,
        "event_phase": phase,
        "confidence_score": score,
        "score_threshold": cfg.score_threshold,
        "passed_threshold": score >= cfg.score_threshold,
        "is_match": is_match,
        "has_affirmative_evidence": has_affirmative_evidence,
        "evidence": evidence,
        "negative_evidence": negative_evidence,
        "parse_ok": True,
    }


def public_verification_images(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]


def request_sample_judgment(
    cfg: SearchConfig,
    cache_dir: pathlib.Path,
    center_time: float,
    image_records: list[dict[str, Any]],
    before_time: float,
    after_time: float,
    verification_mode: str,
) -> dict[str, Any]:
    result_path = cache_dir / "results" / f"sample_{sample_key(center_time)}_{verification_mode}.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    raw_dir = cache_dir / "raw-responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_verification_prompt(cfg, center_time, before_time, after_time, verification_mode)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    role_names = {
        "before": "Before frame",
        "target": "Target frame",
        "after": "After frame",
    }
    for record in image_records:
        label = role_names.get(str(record["role"]), str(record["role"]).title())
        content.append({"type": "text", "text": f"{label} at {float(record['time']):.3f}s"})
        content.append({"type": "image_url", "image_url": {"url": image_data_url(record["_model_frame_abs"])}})
    payload = {
        "model": cfg.model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": cfg.temperature,
        "top_p": 0.9,
        "max_tokens": cfg.max_tokens,
    }

    raw_response_paths: list[str] = []
    last_error = ""
    for attempt in range(1, cfg.request_retries + 2):
        response = post_json(f"{cfg.endpoint.rstrip('/')}/chat/completions", payload, cfg.request_timeout)
        content = response["choices"][0]["message"]["content"].strip()
        raw_path = raw_dir / f"sample_{sample_key(center_time)}_{verification_mode}_attempt_{attempt}.txt"
        write_text_atomic(raw_path, content + "\n")
        raw_response_paths.append(workspace_path(raw_path))
        try:
            parsed = extract_json_object(content)
            result = normalize_verification_result(parsed, cfg)
            target = next(item for item in image_records if item["role"] == "target")
            before = next((item for item in image_records if item["role"] == "before"), None)
            after = next((item for item in image_records if item["role"] == "after"), None)
            result.update({
                "time": center_time,
                "before_time": before_time,
                "after_time": after_time,
                "verification_mode": verification_mode,
                "sent_image_count": len(image_records),
                "verification_image_max_edge_pixels": cfg.verification_image_max_edge_pixels,
                "verification_images": public_verification_images(image_records),
                "frame_path": target["source_frame_path"],
                "model_frame_path": target["model_frame_path"],
                "model_input_width": target["model_input_width"],
                "model_input_height": target["model_input_height"],
                "before_frame_path": before["source_frame_path"] if before else "",
                "after_frame_path": after["source_frame_path"] if after else "",
                "raw_response_paths": raw_response_paths,
            })
            write_json_atomic(result_path, result)
            return result
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(0.1)

    result = {
        "time": center_time,
        "before_time": before_time,
        "after_time": after_time,
        "caption": "",
        "neutral_caption": "",
        "event_phase": "uncertain",
        "confidence_score": 0.0,
        "score_threshold": cfg.score_threshold,
        "passed_threshold": False,
        "is_match": False,
        "has_affirmative_evidence": False,
        "evidence": "",
        "negative_evidence": f"Model response could not be parsed as JSON: {last_error}",
        "parse_ok": False,
        "verification_mode": verification_mode,
        "sent_image_count": len(image_records),
        "verification_image_max_edge_pixels": cfg.verification_image_max_edge_pixels,
        "verification_images": public_verification_images(image_records),
        "frame_path": next(item for item in image_records if item["role"] == "target")["source_frame_path"],
        "model_frame_path": next(item for item in image_records if item["role"] == "target")["model_frame_path"],
        "model_input_width": next(item for item in image_records if item["role"] == "target")["model_input_width"],
        "model_input_height": next(item for item in image_records if item["role"] == "target")["model_input_height"],
        "before_frame_path": next((item["source_frame_path"] for item in image_records if item["role"] == "before"), ""),
        "after_frame_path": next((item["source_frame_path"] for item in image_records if item["role"] == "after"), ""),
        "raw_response_paths": raw_response_paths,
    }
    write_json_atomic(result_path, result)
    return result


def evaluate_sample(
    cfg: SearchConfig,
    cache_dir: pathlib.Path,
    duration: float,
    requested_time: float,
    verification_mode: str,
) -> dict[str, Any]:
    if verification_mode not in {"single", "triple"}:
        raise ValueError(f"Unknown verification_mode: {verification_mode}")
    center_time = normalize_time(requested_time, duration)
    before_time = normalize_time(center_time - cfg.context_seconds, duration)
    after_time = normalize_time(center_time + cfg.context_seconds, duration)
    frame_dir = cache_dir / "frames"
    center_frame = frame_dir / f"t_{sample_key(center_time)}.jpg"
    before_frame = frame_dir / f"t_{sample_key(before_time)}.jpg"
    after_frame = frame_dir / f"t_{sample_key(after_time)}.jpg"
    extract_frame(cfg.video_path, center_time, center_frame)
    image_records = []
    if verification_mode == "triple":
        extract_frame(cfg.video_path, before_time, before_frame)
        image_records.append(resize_frame_for_verification(cfg, cache_dir, before_frame, "before", before_time))
    image_records.append(resize_frame_for_verification(cfg, cache_dir, center_frame, "target", center_time))
    if verification_mode == "triple":
        extract_frame(cfg.video_path, after_time, after_frame)
        image_records.append(resize_frame_for_verification(cfg, cache_dir, after_frame, "after", after_time))
    return request_sample_judgment(
        cfg,
        cache_dir,
        center_time,
        image_records,
        before_time,
        after_time,
        verification_mode,
    )


def sample_confidence(sample: Optional[dict[str, Any]]) -> float:
    if not sample:
        return 0.0
    return clamp_score(sample.get("confidence_score", 0.0))


def should_confirm_primary(primary: dict[str, Any]) -> bool:
    return primary.get("event_phase") in {"match", "near_miss", "uncertain"}


def compose_evaluation_result(
    primary: Optional[dict[str, Any]],
    confirmation: Optional[dict[str, Any]],
    verification_status: str,
    confirmation_reason: str,
) -> dict[str, Any]:
    visible = confirmation or primary
    if visible is None:
        raise ValueError("At least one verification result is required.")
    result = dict(visible)
    result["primary_verification"] = primary
    result["confirmation_verification"] = confirmation
    result["verification_status"] = verification_status
    result["confirmation_reason"] = confirmation_reason
    result["candidate_is_match"] = bool(visible.get("is_match"))
    result["is_match"] = bool(confirmation and confirmation.get("is_match"))
    result["sent_image_count"] = (primary or {}).get("sent_image_count", 0) + (confirmation or {}).get("sent_image_count", 0)
    result["vl_request_count"] = int(primary is not None) + int(confirmation is not None)
    return result


def evaluation_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": round(float(item["time"]), 3),
        "phase": item.get("phase", ""),
        "candidate_window_index": item.get("candidate_window_index"),
        "neutral_caption": item.get("neutral_caption", item.get("caption", "")),
        "caption": item.get("neutral_caption", item.get("caption", "")),
        "event_phase": item.get("event_phase", "uncertain"),
        "confidence_score": item.get("confidence_score", 0.0),
        "score_threshold": item.get("score_threshold", 0.0),
        "passed_threshold": item.get("passed_threshold", False),
        "is_match": item.get("is_match", False),
        "candidate_is_match": item.get("candidate_is_match", item.get("is_match", False)),
        "verification_status": item.get("verification_status", ""),
        "confirmation_reason": item.get("confirmation_reason", ""),
        "verification_mode": item.get("verification_mode", ""),
        "sent_image_count": item.get("sent_image_count", 0),
        "vl_request_count": item.get("vl_request_count", 0),
        "model_input_width": item.get("model_input_width"),
        "model_input_height": item.get("model_input_height"),
        "verification_image_max_edge_pixels": item.get("verification_image_max_edge_pixels"),
        "verification_images": item.get("verification_images", []),
        "primary_verification": item.get("primary_verification"),
        "confirmation_verification": item.get("confirmation_verification"),
        "has_affirmative_evidence": item.get("has_affirmative_evidence", False),
        "evidence": item.get("evidence", ""),
        "negative_evidence": item.get("negative_evidence", ""),
        "parse_ok": item.get("parse_ok", False),
        "frame_path": item.get("frame_path", ""),
    }


def grouped_positive_samples(samples: list[dict[str, Any]], cfg: SearchConfig) -> list[list[dict[str, Any]]]:
    positives = [item for item in samples if item.get("is_match")]
    positives.sort(key=lambda item: item["time"])
    groups: list[list[dict[str, Any]]] = []
    gap_limit = max(cfg.local_scan_interval * 1.75, cfg.minimum_event_duration / 2.0)
    for item in positives:
        if not groups or item["time"] - groups[-1][-1]["time"] > gap_limit:
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def preliminary_occurrences(samples: list[dict[str, Any]], cfg: SearchConfig, duration: float) -> list[dict[str, Any]]:
    occurrences = []
    for group in grouped_positive_samples(samples, cfg):
        representative = max(group, key=sample_confidence)
        accepted = len(group) >= cfg.minimum_positive_samples
        occurrences.append({
            "index": len(occurrences) + 1,
            "status": "accepted" if accepted else "insufficient_positive_samples",
            "start_time": round(max(0.0, group[0]["time"] - cfg.local_scan_interval / 2.0), 3),
            "end_time": round(min(duration, group[-1]["time"] + cfg.local_scan_interval / 2.0), 3),
            "representative_time": round(representative["time"], 3),
            "confidence": round(sample_confidence(representative), 3),
            "positive_sample_count": len(group),
            "minimum_positive_samples": cfg.minimum_positive_samples,
            "matched_sample_times": [round(item["time"], 3) for item in group],
            "representative_caption": representative.get("neutral_caption", representative.get("caption", "")),
            "evidence": representative.get("evidence", ""),
        })
    return occurrences


def nearest_nonmatch_before(samples: list[dict[str, Any]], time_seconds: float) -> Optional[dict[str, Any]]:
    before = [item for item in samples if item["time"] < time_seconds and not item.get("is_match")]
    if not before:
        return None
    return max(before, key=lambda item: item["time"])


def nearest_nonmatch_after(samples: list[dict[str, Any]], time_seconds: float) -> Optional[dict[str, Any]]:
    after = [item for item in samples if item["time"] > time_seconds and not item.get("is_match")]
    if not after:
        return None
    return min(after, key=lambda item: item["time"])


def write_evidence_frames(cfg: SearchConfig, duration: float, occurrences: list[dict[str, Any]]) -> None:
    evidence_root = cfg.output_dir / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    for occurrence in occurrences:
        occurrence_dir = evidence_root / f"occurrence_{occurrence['index']:03d}"
        occurrence_dir.mkdir(parents=True, exist_ok=True)
        representative_time = float(occurrence["representative_time"])
        before_time = normalize_time(representative_time - cfg.context_seconds, duration)
        center_time = normalize_time(representative_time, duration)
        after_time = normalize_time(representative_time + cfg.context_seconds, duration)
        paths = {
            "before_frame": occurrence_dir / "before.jpg",
            "representative_frame": occurrence_dir / "representative.jpg",
            "after_frame": occurrence_dir / "after.jpg",
        }
        extract_frame(cfg.video_path, before_time, paths["before_frame"])
        extract_frame(cfg.video_path, center_time, paths["representative_frame"])
        extract_frame(cfg.video_path, after_time, paths["after_frame"])
        occurrence["evidence_frames"] = {
            "before_time": before_time,
            "representative_time": center_time,
            "after_time": after_time,
            "before_frame": output_path(paths["before_frame"], cfg.output_dir),
            "representative_frame": output_path(paths["representative_frame"], cfg.output_dir),
            "after_frame": output_path(paths["after_frame"], cfg.output_dir),
        }


def state_samples(state: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(state["evaluations"].values(), key=lambda item: item["time"])


def build_result(
    cfg: SearchConfig,
    duration: float,
    cache_dir: pathlib.Path,
    state: dict[str, Any],
    incomplete_reason: Optional[str],
    status_override: Optional[str] = None,
    progress: Optional[dict[str, Any]] = None,
    include_evidence: bool = False,
) -> dict[str, Any]:
    samples = state_samples(state)
    occurrences = state.get("occurrences") or preliminary_occurrences(samples, cfg, duration)
    accepted_occurrences = [item for item in occurrences if item.get("status", "accepted") == "accepted"]
    primary_count = sum(1 for item in samples if item.get("primary_verification"))
    confirmation_count = sum(1 for item in samples if item.get("confirmation_verification"))
    direct_triple_count = sum(
        1
        for item in samples
        if item.get("verification_status") == "confirmed" and not item.get("primary_verification")
    )
    vl_request_count = state.get("vl_request_count", sum(int(item.get("vl_request_count", 1)) for item in samples))
    sent_image_count = state.get("sent_image_count", sum(int(item.get("sent_image_count", 0)) for item in samples))
    if include_evidence:
        write_evidence_frames(cfg, duration, accepted_occurrences)
    status = status_override or ("incomplete" if incomplete_reason else ("found" if accepted_occurrences else "not_found"))
    progress_data = progress or {
        "stage": "complete",
        "last_evaluated_time": round(samples[-1]["time"], 3) if samples else None,
        "last_evaluated_phase": samples[-1].get("phase", "") if samples else "",
        "sample_count": len(samples),
        "vl_request_count": state.get("vl_request_count", 0),
        "candidate_window_count": len(state["candidate_windows"]),
        "updated_at_unix": round(time.time(), 3),
    }
    return {
        "status": status,
        "incomplete_reason": incomplete_reason,
        "progress": progress_data,
        "script_version": SCRIPT_VERSION,
        "video": {
            "path": project_path(cfg.video_path),
            "workspace_path": workspace_path(cfg.video_path),
            "duration_seconds": round(duration, 3),
            **state.get("video_dimensions", {}),
        },
        "model": cfg.model,
        "strategy": cfg.strategy,
        "score_threshold": cfg.score_threshold,
        "retrieval": {
            "model": cfg.retrieval_model,
            "cache_dir": project_path(cfg.retrieval_cache_dir),
            "content_filter": cfg.retrieval_content_filter,
            **state.get("retrieval_metadata", {}),
        },
        "search": {
            "scan_interval_seconds": cfg.scan_interval,
            "local_scan_interval_seconds": cfg.local_scan_interval,
            "boundary_tolerance_seconds": cfg.boundary_tolerance,
            "minimum_event_duration_seconds": cfg.minimum_event_duration,
            "candidate_padding_seconds": cfg.candidate_padding,
            "minimum_candidate_windows": cfg.minimum_candidate_windows,
            "max_candidate_windows": cfg.max_candidate_windows,
            "minimum_positive_samples": cfg.minimum_positive_samples,
            "verification_image_max_edge_pixels": cfg.verification_image_max_edge_pixels,
            "max_evaluations": cfg.max_evaluations,
            "miss_risk_note": (
                "retrieve_verify ranks every scan frame with a non-generative image-text encoder, "
                "then verifies only the best candidate windows with the VL model. Very short events can still be missed "
                "if retrieval does not rank them near the top. Lower score_threshold, increase max_candidate_windows, "
                "or reduce scan_interval_seconds to widen the search."
            ),
        },
        "retrieval_scan": {
            "scan_interval_seconds": cfg.scan_interval,
            "score_threshold": cfg.score_threshold,
            "sample_count": len(state["retrieval_samples"]),
            "samples": state["retrieval_samples"],
            "candidate_windows": state["candidate_windows"],
        },
        "verification": {
            "score_threshold": cfg.score_threshold,
            "sample_count": len(samples),
            "vl_request_count": vl_request_count,
            "primary_evaluation_count": primary_count,
            "confirmation_evaluation_count": confirmation_count + direct_triple_count,
            "single_image_evaluation_count": primary_count,
            "three_image_evaluation_count": confirmation_count + direct_triple_count,
            "total_sent_image_count": sent_image_count,
            "verification_image_max_edge_pixels": cfg.verification_image_max_edge_pixels,
            "match_count": sum(1 for item in samples if item.get("is_match")),
            "evaluations": [evaluation_summary(item) for item in samples],
        },
        "searched_windows": state["searched_windows"],
        "boundary_refinements": state["boundary_refinements"],
        "skipped_windows": state["skipped_windows"],
        "occurrences": accepted_occurrences,
        "rejected_occurrences": [item for item in occurrences if item.get("status") != "accepted"],
        "outputs": {
            "output_json": "output.json",
            "captions_md": "captions.md",
            "captions_json": "captions.json",
            "event_search_md": "event-search.md",
            "search_trace_json": "search-trace.json",
            "config_snapshot_json": "config.snapshot.json",
        },
    }


def write_progress_output(
    cfg: SearchConfig,
    duration: float,
    cache_dir: pathlib.Path,
    state: dict[str, Any],
    incomplete_reason: Optional[str],
    progress: dict[str, Any],
) -> None:
    status_override = None if incomplete_reason else "running"
    result = build_result(cfg, duration, cache_dir, state, incomplete_reason, status_override=status_override, progress=progress)
    write_json_atomic(cfg.output_dir / "output.json", result)


def build_captions(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    local_samples = [item for item in samples if item.get("phase") in {"local_scan", "boundary_start", "boundary_end"}]
    return [{
        "time": round(float(item["time"]), 3),
        "neutral_caption": item.get("neutral_caption", item.get("caption", "")),
        "caption": item.get("neutral_caption", item.get("caption", "")),
        "event_phase": item.get("event_phase", "uncertain"),
        "confidence_score": item.get("confidence_score", 0.0),
        "is_match": item.get("is_match", False),
        "candidate_is_match": item.get("candidate_is_match", item.get("is_match", False)),
        "verification_status": item.get("verification_status", ""),
        "has_affirmative_evidence": item.get("has_affirmative_evidence", False),
        "evidence": item.get("evidence", ""),
        "negative_evidence": item.get("negative_evidence", ""),
        "frame_path": item.get("frame_path", ""),
    } for item in local_samples]


def write_captions_markdown(cfg: SearchConfig, captions: list[dict[str, Any]]) -> None:
    lines = [
        f"# Verified captions: {cfg.video_path.name}",
        "",
        f"- Condition: {cfg.condition}",
        f"- Retrieval scan interval: {cfg.scan_interval:g} seconds",
        f"- Local verification interval: {cfg.local_scan_interval:g} seconds",
        "",
    ]
    for item in captions:
        lines.append(
            f"- {item['time']:.3f}s: {item['neutral_caption']} "
            f"(phase: {item['event_phase']}, confidence: {item['confidence_score']:.3f}, "
            f"match: {item['is_match']}, verification: {item['verification_status']})"
        )
    write_text_atomic(cfg.output_dir / "captions.md", "\n".join(lines) + "\n")


def write_event_markdown(cfg: SearchConfig, result: dict[str, Any]) -> None:
    lines = [
        f"# Video event search: {cfg.video_path.name}",
        "",
        f"- Status: {result['status']}",
        f"- Condition: {cfg.condition}",
        f"- Strategy: {cfg.strategy}",
        f"- Retrieval model: {cfg.retrieval_model}",
        f"- Retrieval scan interval: {cfg.scan_interval:g} seconds",
        f"- Local verification interval: {cfg.local_scan_interval:g} seconds",
        f"- Samples checked: {result['verification']['sample_count']}",
        f"- VL requests: {result['verification']['vl_request_count']}",
        f"- Sent images: {result['verification']['total_sent_image_count']}",
        f"- Verification image max edge: {cfg.verification_image_max_edge_pixels if cfg.verification_image_max_edge_pixels is not None else 'disabled'} px",
        f"- Candidate windows: {len(result['retrieval_scan']['candidate_windows'])}",
        "",
        "## Search note",
        "",
        str(result["search"]["miss_risk_note"]),
        "",
    ]
    if result.get("incomplete_reason"):
        lines.extend(["## Incomplete", "", str(result["incomplete_reason"]), ""])

    occurrences = result.get("occurrences", [])
    if not occurrences:
        lines.extend(["## Occurrences", "", "No matching time was found at the configured resolution.", ""])
    else:
        lines.extend(["## Occurrences", ""])
        for occurrence in occurrences:
            frames = occurrence.get("evidence_frames", {})
            lines.extend([
                f"### Occurrence {occurrence['index']}",
                "",
                f"- Interval: {occurrence['start_time']:.3f}s to {occurrence['end_time']:.3f}s",
                f"- Representative time: {occurrence['representative_time']:.3f}s",
                f"- Confidence: {occurrence['confidence']:.3f}",
                f"- Positive samples: {occurrence['positive_sample_count']}",
                f"- Matched sample times: {', '.join(f'{value:.3f}s' for value in occurrence['matched_sample_times'])}",
                f"- Caption: {occurrence['representative_caption']}",
                f"- Evidence: {occurrence['evidence']}",
            ])
            if frames:
                lines.extend([
                    f"- Before frame: {frames['before_frame']} ({frames['before_time']:.3f}s)",
                    f"- Representative frame: {frames['representative_frame']} ({frames['representative_time']:.3f}s)",
                    f"- After frame: {frames['after_frame']} ({frames['after_time']:.3f}s)",
                ])
            lines.append("")
    write_text_atomic(cfg.output_dir / "event-search.md", "\n".join(lines))


def build_search_trace(cfg: SearchConfig, duration: float, cache_dir: pathlib.Path, state: dict[str, Any]) -> dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "strategy": cfg.strategy,
        "cache_dir": workspace_path(cache_dir),
        "score_threshold": cfg.score_threshold,
        "retrieval_query_plan": cfg.query_plan,
        "retrieval": state.get("retrieval_metadata", {}),
        "retrieval_samples": state["retrieval_samples"],
        "candidate_windows": state["candidate_windows"],
        "searched_windows": state["searched_windows"],
        "skipped_windows": state["skipped_windows"],
        "boundary_refinements": state["boundary_refinements"],
        "verification": {
            "vl_request_count": state.get("vl_request_count", 0),
            "total_sent_image_count": state.get("sent_image_count", 0),
            "verification_image_max_edge_pixels": cfg.verification_image_max_edge_pixels,
        },
        "evaluations": [evaluation_summary(item) for item in state_samples(state)],
        "duration_seconds": round(duration, 3),
    }


def write_outputs(
    cfg: SearchConfig,
    duration: float,
    cache_dir: pathlib.Path,
    state: dict[str, Any],
    incomplete_reason: Optional[str],
) -> dict[str, Any]:
    samples = state_samples(state)
    captions = build_captions(samples)
    result = build_result(cfg, duration, cache_dir, state, incomplete_reason, include_evidence=True)
    search_trace = build_search_trace(cfg, duration, cache_dir, state)

    legacy_json_path = cfg.output_dir / "event-search.json"
    if legacy_json_path.exists():
        legacy_json_path.unlink()

    write_json_atomic(cfg.output_dir / "captions.json", captions)
    write_json_atomic(cfg.output_dir / "output.json", result)
    write_json_atomic(cfg.output_dir / "search-trace.json", search_trace)
    write_captions_markdown(cfg, captions)
    write_event_markdown(cfg, result)
    return result


def new_state() -> dict[str, Any]:
    return {
        "video_dimensions": {},
        "retrieval_samples": [],
        "retrieval_metadata": {},
        "candidate_windows": [],
        "evaluations": {},
        "vl_request_count": 0,
        "sent_image_count": 0,
        "searched_windows": [],
        "skipped_windows": [],
        "boundary_refinements": [],
        "occurrences": [],
        "incomplete_reason": None,
    }


def evaluation_limit_reached(cfg: SearchConfig, state: dict[str, Any]) -> bool:
    return cfg.max_evaluations is not None and int(state.get("vl_request_count", 0)) >= cfg.max_evaluations


def record_vl_request(state: dict[str, Any], result: dict[str, Any]) -> None:
    state["vl_request_count"] = int(state.get("vl_request_count", 0)) + 1
    state["sent_image_count"] = int(state.get("sent_image_count", 0)) + int(result.get("sent_image_count", 0))


def ensure_evaluation(
    cfg: SearchConfig,
    duration: float,
    cache_dir: pathlib.Path,
    state: dict[str, Any],
    time_seconds: float,
    phase: str,
    candidate_window_index: Optional[int],
    progress_writer: Any,
) -> bool:
    normalized = normalize_time(time_seconds, duration)
    key = sample_key(normalized)
    if key in state["evaluations"]:
        return True
    if evaluation_limit_reached(cfg, state):
        return False

    def store_result(result: dict[str, Any], progress_stage: str) -> None:
        result["phase"] = phase
        result["candidate_window_index"] = candidate_window_index
        state["evaluations"][key] = result
        progress_writer({
            "stage": progress_stage,
            "last_evaluated_time": normalized,
            "last_evaluated_phase": phase,
            "sample_count": len(state["evaluations"]),
            "vl_request_count": state.get("vl_request_count", 0),
            "candidate_window_count": len(state["candidate_windows"]),
            "updated_at_unix": round(time.time(), 3),
        })

    if phase == "local_scan":
        print(f"Verifying {normalized:.3f}s ({phase}, single image)...", file=sys.stderr)
        primary = evaluate_sample(cfg, cache_dir, duration, normalized, "single")
        record_vl_request(state, primary)
        result = compose_evaluation_result(
            primary,
            None,
            "primary_only",
            "single-image primary check did not require three-image confirmation",
        )
        store_result(result, f"{phase}_primary")
        if not should_confirm_primary(primary):
            return True
        if evaluation_limit_reached(cfg, state):
            blocked = compose_evaluation_result(
                primary,
                None,
                "confirmation_blocked_by_max_evaluations",
                "three-image confirmation was required, but max_evaluations was reached",
            )
            store_result(blocked, f"{phase}_confirmation_blocked")
            return False
        print(f"Verifying {normalized:.3f}s ({phase}, three-image confirmation)...", file=sys.stderr)
        confirmation = evaluate_sample(cfg, cache_dir, duration, normalized, "triple")
        record_vl_request(state, confirmation)
        confirmed = compose_evaluation_result(
            primary,
            confirmation,
            "confirmed",
            f"primary event_phase={primary.get('event_phase', 'uncertain')} requires three-image confirmation",
        )
        store_result(confirmed, f"{phase}_confirm")
        return True

    print(f"Verifying {normalized:.3f}s ({phase}, three images)...", file=sys.stderr)
    confirmation = evaluate_sample(cfg, cache_dir, duration, normalized, "triple")
    record_vl_request(state, confirmation)
    result = compose_evaluation_result(
        None,
        confirmation,
        "confirmed",
        "boundary refinement uses three-image verification",
    )
    store_result(result, phase)
    return True


def refine_boundary(
    cfg: SearchConfig,
    duration: float,
    cache_dir: pathlib.Path,
    state: dict[str, Any],
    negative_time: float,
    positive_time: float,
    phase: str,
    candidate_window_index: Optional[int],
    progress_writer: Any,
) -> float:
    left = normalize_time(min(negative_time, positive_time), duration)
    right = normalize_time(max(negative_time, positive_time), duration)
    positive_is_right = positive_time >= negative_time
    while right - left > cfg.boundary_tolerance + 0.0005:
        midpoint = round((left + right) / 2.0, 3)
        if not ensure_evaluation(cfg, duration, cache_dir, state, midpoint, phase, candidate_window_index, progress_writer):
            state["incomplete_reason"] = f"Stopped after max_evaluations={cfg.max_evaluations} VL requests."
            break
        sample = state["evaluations"][sample_key(midpoint)]
        if sample.get("is_match") == positive_is_right:
            right = midpoint
        else:
            left = midpoint
    return right if positive_is_right else left


def finalize_occurrences(
    cfg: SearchConfig,
    duration: float,
    cache_dir: pathlib.Path,
    state: dict[str, Any],
    progress_writer: Any,
) -> Optional[str]:
    samples = state_samples(state)
    preliminary = preliminary_occurrences(samples, cfg, duration)
    accepted = [item for item in preliminary if item["status"] == "accepted"]
    refined: list[dict[str, Any]] = []
    for occurrence in accepted:
        group = [state["evaluations"][sample_key(value)] for value in occurrence["matched_sample_times"]]
        first = group[0]
        last = group[-1]
        previous_negative = nearest_nonmatch_before(samples, first["time"])
        next_negative = nearest_nonmatch_after(samples, last["time"])
        start_time = occurrence["start_time"]
        end_time = occurrence["end_time"]
        if previous_negative:
            start_time = refine_boundary(
                cfg,
                duration,
                cache_dir,
                state,
                previous_negative["time"],
                first["time"],
                "boundary_start",
                occurrence.get("candidate_window_index"),
                progress_writer,
            )
        if next_negative:
            end_time = refine_boundary(
                cfg,
                duration,
                cache_dir,
                state,
                next_negative["time"],
                last["time"],
                "boundary_end",
                occurrence.get("candidate_window_index"),
                progress_writer,
            )
        occurrence = dict(occurrence)
        occurrence["start_time"] = round(min(start_time, end_time), 3)
        occurrence["end_time"] = round(max(start_time, end_time), 3)
        occurrence["boundary_tolerance_seconds"] = cfg.boundary_tolerance
        refined.append(occurrence)
        state["boundary_refinements"].append({
            "occurrence_index": occurrence["index"],
            "start_time": occurrence["start_time"],
            "end_time": occurrence["end_time"],
            "boundary_tolerance_seconds": cfg.boundary_tolerance,
        })
    for index, occurrence in enumerate(refined, start=1):
        occurrence["index"] = index
    state["occurrences"] = refined
    return state.get("incomplete_reason")


def run_search(
    cfg: SearchConfig,
    duration: float,
    cache_dir: pathlib.Path,
    state: dict[str, Any],
    progress_writer: Any,
) -> tuple[dict[str, Any], Optional[str]]:
    incomplete_reason: Optional[str] = None
    progress_writer({
        "stage": "starting",
        "last_evaluated_time": None,
        "last_evaluated_phase": "",
        "sample_count": 0,
        "vl_request_count": 0,
        "candidate_window_count": 0,
        "updated_at_unix": round(time.time(), 3),
    })

    times = scan_times(duration, cfg.scan_interval)
    print(f"Extracting {len(times)} retrieval frames...", file=sys.stderr)
    scan_frames = extract_scan_frames(cfg.video_path, times, cache_dir / "retrieval-frames")
    print(f"Scoring retrieval frames with {cfg.retrieval_model}...", file=sys.stderr)
    retrieval_samples, retrieval_metadata = score_retrieval_frames(cfg, scan_frames)
    state["retrieval_samples"] = retrieval_samples
    state["retrieval_metadata"] = retrieval_metadata
    state["candidate_windows"] = select_retrieval_windows(cfg, duration, retrieval_samples)
    for window in state["candidate_windows"]:
        window["local_scan_times"] = local_scan_times(window["start_time"], window["end_time"], cfg.local_scan_interval, duration)
    progress_writer({
        "stage": "retrieval_complete",
        "last_evaluated_time": None,
        "last_evaluated_phase": "retrieval",
        "sample_count": 0,
        "vl_request_count": state.get("vl_request_count", 0),
        "candidate_window_count": len(state["candidate_windows"]),
        "updated_at_unix": round(time.time(), 3),
    })

    if not state["candidate_windows"]:
        state["skipped_windows"].append({"reason": "retrieval produced no candidate windows"})
        return state, None

    for window in state["candidate_windows"]:
        window_record = {
            "index": window["index"],
            "start_time": window["start_time"],
            "end_time": window["end_time"],
            "peak_time": window["peak_time"],
            "peak_retrieval_likelihood_score": window["peak_retrieval_likelihood_score"],
            "sample_count": len(window["local_scan_times"]),
        }
        state["searched_windows"].append(window_record)
        for time_seconds in window["local_scan_times"]:
            if not ensure_evaluation(cfg, duration, cache_dir, state, time_seconds, "local_scan", window["index"], progress_writer):
                incomplete_reason = f"Stopped after max_evaluations={cfg.max_evaluations} VL requests."
                break
        if incomplete_reason:
            break

    if incomplete_reason is None:
        incomplete_reason = finalize_occurrences(cfg, duration, cache_dir, state, progress_writer)
        progress_writer({
            "stage": "boundary_complete",
            "last_evaluated_time": None,
            "last_evaluated_phase": "boundary",
            "sample_count": len(state["evaluations"]),
            "vl_request_count": state.get("vl_request_count", 0),
            "candidate_window_count": len(state["candidate_windows"]),
            "updated_at_unix": round(time.time(), 3),
        })
    return state, incomplete_reason


def self_test() -> int:
    tests_dir = WORKSPACE_ROOT / "tests"
    if not tests_dir.exists():
        return fail(f"Test directory not found: {tests_dir}")
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(tests_dir)],
        cwd=str(WORKSPACE_ROOT),
    )
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Search a local video for all times matching a JSON-defined visual condition.")
    parser.add_argument("input_json", nargs="?", help="JSON config path")
    parser.add_argument("--endpoint", help="Override OpenAI-compatible VL endpoint")
    parser.add_argument("--model", help="Override model alias")
    parser.add_argument("--self-test", action="store_true", help="Run local unit tests that do not call the model")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.input_json:
        parser.error("input_json is required unless --self-test is used")

    try:
        require_command("ffmpeg")
        require_command("ffprobe")
        cfg = load_config(pathlib.Path(args.input_json), args.endpoint, args.model)
        query_plan = compile_condition_query_plan(cfg)
        cfg = attach_condition_query_plan(cfg, query_plan)
        prepare_output_dir(cfg)
        duration = ffprobe_duration(cfg.video_path)
        cache_dir = WORKSPACE_ROOT / "work" / "video-event-search" / config_hash(cfg)
        cache_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(cache_dir / "config.cache-identity.json", {
            "script_version": SCRIPT_VERSION,
            "config": cfg.canonical,
            "duration_seconds": round(duration, 3),
        })

        state = new_state()
        state["video_dimensions"] = ffprobe_video_dimensions(cfg.video_path)
        incomplete_reason: Optional[str] = None

        def run_progress(progress: dict[str, Any]) -> None:
            write_progress_output(cfg, duration, cache_dir, state, incomplete_reason, progress)

        state, incomplete_reason = run_search(cfg, duration, cache_dir, state, run_progress)
        result = write_outputs(cfg, duration, cache_dir, state, incomplete_reason)
    except ConfigError as exc:
        return fail(str(exc))
    except subprocess.CalledProcessError as exc:
        return fail(f"Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}")
    except Exception as exc:
        return fail(f"Video event search failed: {exc}")

    print(cfg.output_dir / "output.json")
    print(cfg.output_dir / "event-search.md")
    print(f"status={result['status']} occurrences={len(result['occurrences'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
