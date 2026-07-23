#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NODE_BIN="${NODE_BIN:-node}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INFERENCE_DEVICE="${TORCH_INFERENCE_DEVICE:-cpu}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-mb_alphastar_fairppo90}"
RUN_ID="${RUN_ID:-mb_alphastar_fairppo90_pfsp}"
ROOT="experiments/mb_alpha_league/${RUN_ID}"
STATUS_FILE="${ROOT}/finalization/acceptance_status.json"
mkdir -p "$(dirname "$STATUS_FILE")"

stage="initializing"
write_status() {
  "$NODE_BIN" -e "const fs=require('fs'); fs.writeFileSync(process.argv[1], JSON.stringify({updated_at:new Date().toISOString(),state:process.argv[2],stage:process.argv[3]},null,2)+'\n')" "$STATUS_FILE" "$1" "$stage"
}
on_error() {
  write_status failed
}
trap on_error ERR
write_status running

stage="preview_fallback_gate"
write_status running
selection_id="${RUN_ID}_preview_selection"
preview_dir="${ROOT}/preview_selection"
preview_selection="${preview_dir}/${selection_id}.json"
SOURCE_RUN_ID="$RUN_ID" \
SELECTION_ID="$selection_id" \
MODEL_MANIFEST="${ROOT}/selection/${RUN_ID}_checkpoint_selection_manifest.json" \
PREVIEW_ITERATIONS="011 012 013 014 015" \
FALLBACK_PREVIEW_CHECKPOINT="models/torch/${SOURCE_RUN_ID}/iter_010/universal_preview/checkpoint.pt" \
FALLBACK_PREVIEW_ID="preview_original_iter_010" \
GAMES_PER_PAIRING=3 \
OUT_DIR="$preview_dir" \
REUSE_EXISTING=1 \
PYTHON_BIN="$PYTHON_BIN" \
TORCH_INFERENCE_DEVICE="$TORCH_INFERENCE_DEVICE" \
bash scripts/run_mb_preview_selection.sh

stage="fresh_seed_finalists"
write_status running
final_holdout_dir="${ROOT}/final_holdout"
final_holdout_id="${RUN_ID}_final_holdout"
finalists_id="${RUN_ID}_finalists"
finalists_dir="${ROOT}/finalists"
RUN_ID="$RUN_ID" \
FINALISTS_ID="$finalists_id" \
SOURCE_HOLDOUT_REPORT="${final_holdout_dir}/${final_holdout_id}_report.json" \
SELECTED_MANIFEST="${final_holdout_dir}/${final_holdout_id}_validated_manifest.json" \
PREVIEW_SELECTION="$preview_selection" \
BASELINE_MODELS_DIR="models/torch/${RUN_ID}/iter_010/agents" \
GAMES_PER_PAIRING=10 \
OUT_DIR="$finalists_dir" \
REUSE_EXISTING=1 \
PYTHON_BIN="$PYTHON_BIN" \
TORCH_INFERENCE_DEVICE="$TORCH_INFERENCE_DEVICE" \
bash scripts/run_mb_finalists_evaluation.sh

stage="package"
write_status running
"$NODE_BIN" scripts/package_final_agent.js \
  --team-selection "${finalists_dir}/${finalists_id}_final_team.json" \
  --preview-selection "$preview_selection" \
  --out-dir models/torch/final_mb_agent \
  --overwrite

stage="offline_package_smoke"
write_status running
"$NODE_BIN" scripts/smoke_final_agent_package.js \
  --package models/torch/final_mb_agent \
  --python "$PYTHON_BIN" \
  --torch-device "$TORCH_INFERENCE_DEVICE"

stage="ladder_protocol_smoke"
write_status running
"$NODE_BIN" scripts/check_showdown_ladder_client.js
"$NODE_BIN" scripts/check_public_state.js
"$NODE_BIN" scripts/check_rollout_public_state.js
"$NODE_BIN" scripts/check_showdown_connection.js

stage="complete"
write_status complete
trap - ERR
echo "Final M-B agent acceptance complete."
echo "Package: models/torch/final_mb_agent/manifest.json"
echo "Finalists: ${finalists_dir}/${finalists_id}_report.json"
