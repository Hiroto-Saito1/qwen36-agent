#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$WORKSPACE_ROOT/.." && pwd)"

VL_ENDPOINT="${VL_ENDPOINT:-http://127.0.0.1:8081/v1}"
VL_MODEL_ALIAS="${VL_MODEL_ALIAS:-qwen2.5-vl-7b-instruct-abliterated-q4km}"
VL_SERVER_PID=""

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
}

vl_model_available() {
  local body
  body="$(curl -fsS --max-time 5 "$VL_ENDPOINT/models" 2>/dev/null || true)"
  [[ "$body" == *"$VL_MODEL_ALIAS"* ]]
}

vl_endpoint_alive() {
  curl -fsS --max-time 3 "$VL_ENDPOINT/models" >/dev/null 2>&1
}

stop_owned_vl_server() {
  if [[ -n "${VL_SERVER_PID:-}" ]]; then
    kill "$VL_SERVER_PID" >/dev/null 2>&1 || true
    wait "$VL_SERVER_PID" >/dev/null 2>&1 || true
  fi
}

ensure_vl_server() {
  require_command curl
  require_command llama-server

  if vl_model_available; then
    return 0
  fi

  if vl_endpoint_alive; then
    echo "Endpoint is alive, but it is not serving the expected model: $VL_MODEL_ALIAS" >&2
    echo "Stop the service on $VL_ENDPOINT or set VL_ENDPOINT to the correct server." >&2
    exit 1
  fi

  mkdir -p "$PROJECT_ROOT/logs"
  "$SCRIPT_DIR/start-vl-server.sh" > "$PROJECT_ROOT/logs/vl-server-repro.log" 2>&1 &
  VL_SERVER_PID="$!"
  trap stop_owned_vl_server EXIT

  for _ in $(seq 1 180); do
    if vl_model_available; then
      return 0
    fi
    if ! kill -0 "$VL_SERVER_PID" >/dev/null 2>&1; then
      echo "VL server exited before becoming ready. Recent log:" >&2
      tail -40 "$PROJECT_ROOT/logs/vl-server-repro.log" >&2 || true
      exit 1
    fi
    sleep 1
  done

  echo "VL server did not become ready within 180 seconds. Recent log:" >&2
  tail -40 "$PROJECT_ROOT/logs/vl-server-repro.log" >&2 || true
  exit 1
}
