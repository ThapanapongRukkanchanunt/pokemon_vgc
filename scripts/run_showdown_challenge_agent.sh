#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv_torch}"
NODE_BIN="${NODE_BIN:-node}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
PACKAGE_DIR="${PACKAGE_DIR:-models/torch/final_mb_agent}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-showdown.env}"
TORCH_DEVICE="${TORCH_DEVICE:-cpu}"
RESTART_DELAY_SECONDS="${RESTART_DELAY_SECONDS:-10}"

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

while true; do
  "$NODE_BIN" scripts/run_showdown_ladder.js \
    --mode challenge \
    --package "$PACKAGE_DIR" \
    --credentials "$CREDENTIALS_FILE" \
    --games 0 \
    --python "$PYTHON_BIN" \
    --torch-device "$TORCH_DEVICE"
  status=$?
  if [[ "$status" -eq 0 ]]; then
    exit 0
  fi
  printf 'Challenge agent exited with status %s; restarting in %ss.\n' \
    "$status" "$RESTART_DELAY_SECONDS" >&2
  sleep "$RESTART_DELAY_SECONDS"
done
