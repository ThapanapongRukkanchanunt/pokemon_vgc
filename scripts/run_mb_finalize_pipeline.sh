#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NODE_BIN="${NODE_BIN:-node}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
TORCH_INFERENCE_DEVICE="${TORCH_INFERENCE_DEVICE:-cpu}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-mb_alphastar_fairppo90}"
CONTINUATION_RUN_ID="${CONTINUATION_RUN_ID:-${SOURCE_RUN_ID}_pfsp}"
RESUME="${RESUME:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

SOURCE_ROOT="experiments/mb_alpha_league/${SOURCE_RUN_ID}"
CONT_ROOT="experiments/mb_alpha_league/${CONTINUATION_RUN_ID}"
ORIGINAL_PREVIEW_MODEL="models/torch/${SOURCE_RUN_ID}/iter_010/universal_preview/checkpoint.pt"
STATUS_DIR="${CONT_ROOT}/finalization"
STATUS_FILE="${STATUS_DIR}/status.json"
mkdir -p "$STATUS_DIR"

stage="initializing"
write_status() {
  local state="$1"
  "$NODE_BIN" -e "const fs=require('fs'); fs.writeFileSync(process.argv[1], JSON.stringify({updated_at:new Date().toISOString(),state:process.argv[2],stage:process.argv[3]},null,2)+'\\n')" "$STATUS_FILE" "$state" "$stage"
}
on_error() {
  write_status failed
}
trap on_error ERR
write_status running

initial_manifest="${SOURCE_ROOT}/selection/${SOURCE_RUN_ID}_checkpoint_selection_manifest.json"
stage="wait_initial_checkpoint_selection"
write_status running
while [[ ! -f "$initial_manifest" ]]; do
  echo "Waiting for initial checkpoint selection: $initial_manifest"
  sleep "$WAIT_SECONDS"
done

stage="initial_preview_selection"
write_status running
initial_preview_selection="${SOURCE_ROOT}/preview_selection/${SOURCE_RUN_ID}_preview_selection.json"
if [[ "$RESUME" != "1" || ! -f "$initial_preview_selection" ]]; then
  SOURCE_RUN_ID="$SOURCE_RUN_ID" \
  MODEL_MANIFEST="$initial_manifest" \
  GAMES_PER_PAIRING=3 \
  PYTHON_BIN="$PYTHON_BIN" \
  TORCH_INFERENCE_DEVICE="$TORCH_INFERENCE_DEVICE" \
  bash scripts/run_mb_preview_selection.sh
fi

stage="initial_holdout"
write_status running
initial_holdout_id="${SOURCE_RUN_ID}_holdout"
initial_holdout_dir="${SOURCE_ROOT}/holdout"
initial_validated_manifest="${initial_holdout_dir}/${initial_holdout_id}_validated_manifest.json"
if [[ "$RESUME" != "1" || ! -f "$initial_validated_manifest" ]]; then
  SOURCE_RUN_ID="$SOURCE_RUN_ID" \
  HOLDOUT_ID="$initial_holdout_id" \
  SELECTED_MANIFEST="$initial_manifest" \
  PREVIEW_SELECTION="$initial_preview_selection" \
  GAMES_PER_PAIRING=3 \
  SEARCH_GAMES_PER_PAIRING=1 \
  PYTHON_BIN="$PYTHON_BIN" \
  TORCH_INFERENCE_DEVICE="$TORCH_INFERENCE_DEVICE" \
  bash scripts/run_mb_holdout_evaluation.sh
fi

stage="pfsp_continuation"
write_status running
continuation_report="${CONT_ROOT}/report.json"
initial_preview_model="$($NODE_BIN -e "console.log(require('./${initial_preview_selection}').checkpoint)")"
if [[ "$RESUME" != "1" || ! -f "$continuation_report" ]]; then
  SOURCE_RUN_ID="$SOURCE_RUN_ID" \
  RUN_ID="$CONTINUATION_RUN_ID" \
  START_MODEL_MANIFEST="$initial_validated_manifest" \
  START_PREVIEW_MODEL="$initial_preview_model" \
  START_ITERATION=11 \
  ITERATIONS=5 \
  LEAGUE_GAMES=1000 \
  HISTORICAL_PROBABILITY=0.35 \
  PFSP_EXPONENT=2 \
  PFSP_PRIOR_GAMES=2 \
  EPSILON=0.02 \
  PPO_EPOCHS=4 \
  PPO_LR=2e-5 \
  PPO_TARGET_KL=0.01 \
  PYTHON_BIN="$PYTHON_BIN" \
  TRAIN_DEVICE="$TRAIN_DEVICE" \
  TORCH_INFERENCE_DEVICE="$TORCH_INFERENCE_DEVICE" \
  bash scripts/run_mb_pfsp_continuation.sh
fi

stage="post_pfsp_checkpoint_selection"
write_status running
post_selection_id="${CONTINUATION_RUN_ID}_checkpoint_selection"
post_manifest="${CONT_ROOT}/selection/${post_selection_id}_manifest.json"
if [[ "$RESUME" != "1" || ! -f "$post_manifest" ]]; then
  SOURCE_RUN_ID="$CONTINUATION_RUN_ID" \
  SELECTION_ID="$post_selection_id" \
  CANDIDATE_ITERATIONS="011 012 013 014 015" \
  FIXED_PREVIEW_ITERATION=015 \
  REFERENCE_ITERATION=015 \
  GAMES_PER_PAIRING=5 \
  PYTHON_BIN="$PYTHON_BIN" \
  TORCH_INFERENCE_DEVICE="$TORCH_INFERENCE_DEVICE" \
  bash scripts/run_mb_candidate_selection.sh
fi

stage="post_pfsp_preview_selection"
write_status running
post_preview_id="${CONTINUATION_RUN_ID}_preview_selection"
post_preview_selection="${CONT_ROOT}/preview_selection/${post_preview_id}.json"
if [[ "$RESUME" != "1" || ! -f "$post_preview_selection" ]]; then
  SOURCE_RUN_ID="$CONTINUATION_RUN_ID" \
  SELECTION_ID="$post_preview_id" \
  MODEL_MANIFEST="$post_manifest" \
  PREVIEW_ITERATIONS="011 012 013 014 015" \
  FALLBACK_PREVIEW_CHECKPOINT="$ORIGINAL_PREVIEW_MODEL" \
  FALLBACK_PREVIEW_ID="preview_original_iter_010" \
  GAMES_PER_PAIRING=3 \
  PYTHON_BIN="$PYTHON_BIN" \
  TORCH_INFERENCE_DEVICE="$TORCH_INFERENCE_DEVICE" \
  bash scripts/run_mb_preview_selection.sh
fi

stage="final_holdout"
write_status running
final_holdout_id="${CONTINUATION_RUN_ID}_final_holdout"
final_holdout_dir="${CONT_ROOT}/final_holdout"
provisional_team_selection="${final_holdout_dir}/${final_holdout_id}_final_team.json"
if [[ "$RESUME" != "1" || ! -f "$provisional_team_selection" ]]; then
  SOURCE_RUN_ID="$CONTINUATION_RUN_ID" \
  HOLDOUT_ID="$final_holdout_id" \
  SELECTED_MANIFEST="$post_manifest" \
  PREVIEW_SELECTION="$post_preview_selection" \
  BASELINE_MODELS_DIR="models/torch/${SOURCE_RUN_ID}/iter_010/agents" \
  BASELINE_PREVIEW="models/torch/${SOURCE_RUN_ID}/iter_010/universal_preview/checkpoint.pt" \
  GAMES_PER_PAIRING=3 \
  SEARCH_GAMES_PER_PAIRING=1 \
  OUT_DIR="$final_holdout_dir" \
  PYTHON_BIN="$PYTHON_BIN" \
  TORCH_INFERENCE_DEVICE="$TORCH_INFERENCE_DEVICE" \
  bash scripts/run_mb_holdout_evaluation.sh
fi

stage="finalists_evaluation"
write_status running
finalists_id="${CONTINUATION_RUN_ID}_finalists"
finalists_dir="${CONT_ROOT}/finalists"
final_team_selection="${finalists_dir}/${finalists_id}_final_team.json"
if [[ "$RESUME" != "1" || ! -f "$final_team_selection" ]]; then
  RUN_ID="$CONTINUATION_RUN_ID" \
  FINALISTS_ID="$finalists_id" \
  SOURCE_HOLDOUT_REPORT="${final_holdout_dir}/${final_holdout_id}_report.json" \
  SELECTED_MANIFEST="${final_holdout_dir}/${final_holdout_id}_validated_manifest.json" \
  PREVIEW_SELECTION="$post_preview_selection" \
  BASELINE_MODELS_DIR="models/torch/${CONTINUATION_RUN_ID}/iter_010/agents" \
  GAMES_PER_PAIRING=10 \
  OUT_DIR="$finalists_dir" \
  PYTHON_BIN="$PYTHON_BIN" \
  TORCH_INFERENCE_DEVICE="$TORCH_INFERENCE_DEVICE" \
  bash scripts/run_mb_finalists_evaluation.sh
fi

stage="package_final_agent"
write_status running
package_dir="models/torch/final_mb_agent"
if [[ "$RESUME" != "1" || ! -f "${package_dir}/manifest.json" ]]; then
  "$NODE_BIN" scripts/package_final_agent.js \
    --team-selection "$final_team_selection" \
    --preview-selection "$post_preview_selection" \
    --out-dir "$package_dir" \
    --overwrite
fi

stage="complete"
write_status complete
trap - ERR
echo "Final M-B agent pipeline complete."
echo "Package: ${package_dir}/manifest.json"
echo "Final holdout: ${final_holdout_dir}/${final_holdout_id}_report.json"
