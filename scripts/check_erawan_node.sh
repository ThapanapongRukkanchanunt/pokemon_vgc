#!/usr/bin/env bash
set -euo pipefail

NODE_BIN="${NODE_BIN:-node}"

if command -v module >/dev/null 2>&1; then
  module avail 2>&1 | grep -i node || true
fi

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "FAIL Node is not available: $NODE_BIN" >&2
  echo "Set NODE_BIN to a Linux Node executable, for example:" >&2
  echo "  export NODE_BIN=\$HOME/node-v24.14.0-linux-x64/bin/node" >&2
  exit 1
fi

echo "node=$("$NODE_BIN" --version)"
NPM_BIN="$(dirname "$NODE_BIN")/npm"
if [[ -x "$NPM_BIN" ]]; then
  echo "npm=$("$NPM_BIN" --version)"
fi

"$NODE_BIN" --check scripts/run_large_scale_training.js
"$NODE_BIN" --check scripts/build_value_dataset.js
"$NODE_BIN" --check scripts/build_search_improved_dataset.js
"$NODE_BIN" --check scripts/prepare_gpu_training_bundle.js

"$NODE_BIN" scripts/run_large_scale_training.js \
  --run-id erawan_node_check \
  --iterations 1 \
  --games 1 \
  --stages generate \
  --train-participants random,maxdamage \
  --out-root experiments/erawan_node_check \
  --log-dir logs/battles/erawan_node_check \
  --compact-logs \
  --overwrite

echo "PASS ERAWAN Node smoke check"
