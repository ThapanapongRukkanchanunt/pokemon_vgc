#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NODE_BIN="${NODE_BIN:-node}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INFERENCE_DEVICE="${TORCH_INFERENCE_DEVICE:-cpu}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-mb_alphastar_fairppo90}"
SELECTION_ID="${SELECTION_ID:-${SOURCE_RUN_ID}_checkpoint_selection}"
CANDIDATE_ITERATIONS="${CANDIDATE_ITERATIONS:-006 007 008 009 010}"
FIXED_PREVIEW_ITERATION="${FIXED_PREVIEW_ITERATION:-010}"
GAMES_PER_PAIRING="${GAMES_PER_PAIRING:-5}"
TOP_K="${TOP_K:-1}"
ROLLOUT_MAX_DECISIONS="${ROLLOUT_MAX_DECISIONS:-120}"
OUT_DIR="${OUT_DIR:-experiments/mb_alpha_league/${SOURCE_RUN_ID}/selection}"

fixed_preview="models/torch/${SOURCE_RUN_ID}/iter_${FIXED_PREVIEW_ITERATION}/universal_preview/checkpoint.pt"
[[ -f "$fixed_preview" ]] || { echo "Missing fixed preview checkpoint: $fixed_preview" >&2; exit 1; }
mkdir -p "$OUT_DIR"

summary_args=()
for iteration in $CANDIDATE_ITERATIONS; do
  tag="iter_${iteration}"
  run_id="${SELECTION_ID}_${tag}"
  models_dir="models/torch/${SOURCE_RUN_ID}/${tag}/agents"
  [[ -d "$models_dir" ]] || { echo "Missing candidate agents: $models_dir" >&2; exit 1; }
  "$NODE_BIN" scripts/evaluate_alpha_league_agents.js \
    --run-id "$run_id" \
    --seed "${SELECTION_ID}:shared" \
    --models-dir "$models_dir" \
    --team-preview-model "$fixed_preview" \
    --preview-mode learned \
    --opponent-agent random \
    --out-dir "$OUT_DIR" \
    --log-dir "logs/battles/${run_id}" \
    --games-per-pairing "$GAMES_PER_PAIRING" \
    --side-swaps \
    --top-k "$TOP_K" \
    --rollout-max-decisions "$ROLLOUT_MAX_DECISIONS" \
    --python "$PYTHON_BIN" \
    --torch-device "$TORCH_INFERENCE_DEVICE" \
    --delete-battle-logs \
    --overwrite
  summary_args+=(--summary "${OUT_DIR}/${run_id}_summary.json")
done

"$NODE_BIN" scripts/select_alpha_league_candidates.js \
  "${summary_args[@]}" \
  --reference iter_010 \
  --out "${OUT_DIR}/${SELECTION_ID}_report.json" \
  --manifest "${OUT_DIR}/${SELECTION_ID}_manifest.json"

echo "Selection report: ${OUT_DIR}/${SELECTION_ID}_report.json"
echo "Mixed checkpoint manifest: ${OUT_DIR}/${SELECTION_ID}_manifest.json"
