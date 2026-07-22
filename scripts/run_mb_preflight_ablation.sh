#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NODE_BIN="${NODE_BIN:-node}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INFERENCE_DEVICE="${TORCH_INFERENCE_DEVICE:-cpu}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-mb_alphastar_preview90}"
SOURCE_ITERATION="${SOURCE_ITERATION:-iter_010}"
PREFLIGHT_ID="${PREFLIGHT_ID:-${SOURCE_RUN_ID}_${SOURCE_ITERATION}_preview_ablation}"
MODELS_DIR="${MODELS_DIR:-models/torch/${SOURCE_RUN_ID}/${SOURCE_ITERATION}/agents}"
PREVIEW_MODEL="${PREVIEW_MODEL:-models/torch/${SOURCE_RUN_ID}/${SOURCE_ITERATION}/universal_preview/checkpoint.pt}"
OUT_DIR="${OUT_DIR:-experiments/mb_alpha_league/${SOURCE_RUN_ID}/preflight}"
GAMES_PER_PAIRING="${GAMES_PER_PAIRING:-3}"
TOP_K="${TOP_K:-1}"
ROLLOUT_MAX_DECISIONS="${ROLLOUT_MAX_DECISIONS:-120}"
DELETE_BATTLE_LOGS="${DELETE_BATTLE_LOGS:-1}"
OVERWRITE="${OVERWRITE:-1}"

[[ -d "$MODELS_DIR" ]] || { echo "Missing models directory: $MODELS_DIR" >&2; exit 1; }
[[ -f "$PREVIEW_MODEL" ]] || { echo "Missing preview model: $PREVIEW_MODEL" >&2; exit 1; }

cleanup_args=()
if [[ "$DELETE_BATTLE_LOGS" == "1" ]]; then
  cleanup_args=(--delete-battle-logs)
else
  cleanup_args=(--compact-logs)
fi

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_args=(--overwrite)
fi

for mode in learned random battle-model; do
  mode_tag="${mode//-/_}"
  preview_args=(--preview-mode "$mode")
  if [[ "$mode" == "learned" ]]; then
    preview_args+=(--team-preview-model "$PREVIEW_MODEL")
  fi

  echo "=== preflight ${mode}: ${GAMES_PER_PAIRING} game(s) per side and pairing ==="
  "$NODE_BIN" scripts/evaluate_alpha_league_agents.js \
    --run-id "${PREFLIGHT_ID}_${mode_tag}" \
    --seed "${PREFLIGHT_ID}:shared" \
    --models-dir "$MODELS_DIR" \
    "${preview_args[@]}" \
    --out-dir "$OUT_DIR" \
    --log-dir "logs/battles/${PREFLIGHT_ID}_${mode_tag}" \
    --games-per-pairing "$GAMES_PER_PAIRING" \
    --side-swaps \
    --top-k "$TOP_K" \
    --rollout-max-decisions "$ROLLOUT_MAX_DECISIONS" \
    --python "$PYTHON_BIN" \
    --torch-device "$TORCH_INFERENCE_DEVICE" \
    "${cleanup_args[@]}" \
    "${overwrite_args[@]}"
done

"$NODE_BIN" scripts/summarize_preview_ablation.js \
  --out-dir "$OUT_DIR" \
  --run-prefix "$PREFLIGHT_ID"

echo "Preflight report: ${OUT_DIR}/${PREFLIGHT_ID}_report.json"
