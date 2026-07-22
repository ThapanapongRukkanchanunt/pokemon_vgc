#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NODE_BIN="${NODE_BIN:-node}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INFERENCE_DEVICE="${TORCH_INFERENCE_DEVICE:-cpu}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-mb_alphastar_fairppo90}"
SELECTION_ID="${SELECTION_ID:-${SOURCE_RUN_ID}_preview_selection}"
MODEL_MANIFEST="${MODEL_MANIFEST:-experiments/mb_alpha_league/${SOURCE_RUN_ID}/selection/${SOURCE_RUN_ID}_checkpoint_selection_manifest.json}"
GAMES_PER_PAIRING="${GAMES_PER_PAIRING:-3}"
OUT_DIR="${OUT_DIR:-experiments/mb_alpha_league/${SOURCE_RUN_ID}/preview_selection}"

[[ -f "$MODEL_MANIFEST" ]] || { echo "Missing selected model manifest: $MODEL_MANIFEST" >&2; exit 1; }
mkdir -p "$OUT_DIR"
candidate_args=()
baseline_args=()

for iteration in 005 006 010; do
  id="preview_iter_${iteration}"
  checkpoint="models/torch/${SOURCE_RUN_ID}/iter_${iteration}/universal_preview/checkpoint.pt"
  run_id="${SELECTION_ID}_${id}"
  [[ -f "$checkpoint" ]] || { echo "Missing preview checkpoint: $checkpoint" >&2; exit 1; }
  "$NODE_BIN" scripts/evaluate_alpha_league_agents.js \
    --run-id "$run_id" \
    --seed "${SELECTION_ID}:shared" \
    --model-manifest "$MODEL_MANIFEST" \
    --team-preview-model "$checkpoint" \
    --preview-mode learned \
    --opponent-agent random \
    --out-dir "$OUT_DIR" \
    --log-dir "logs/battles/${run_id}" \
    --games-per-pairing "$GAMES_PER_PAIRING" \
    --side-swaps \
    --top-k 1 \
    --python "$PYTHON_BIN" \
    --torch-device "$TORCH_INFERENCE_DEVICE" \
    --delete-battle-logs \
    --overwrite
  candidate_args+=(--candidate "${id}=${OUT_DIR}/${run_id}_summary.json,${checkpoint}")
done

for mode in random battle-model; do
  id="${mode//-/_}"
  run_id="${SELECTION_ID}_${id}"
  "$NODE_BIN" scripts/evaluate_alpha_league_agents.js \
    --run-id "$run_id" \
    --seed "${SELECTION_ID}:shared" \
    --model-manifest "$MODEL_MANIFEST" \
    --preview-mode "$mode" \
    --opponent-agent random \
    --out-dir "$OUT_DIR" \
    --log-dir "logs/battles/${run_id}" \
    --games-per-pairing "$GAMES_PER_PAIRING" \
    --side-swaps \
    --top-k 1 \
    --python "$PYTHON_BIN" \
    --torch-device "$TORCH_INFERENCE_DEVICE" \
    --delete-battle-logs \
    --overwrite
  baseline_args+=(--baseline "${id}=${OUT_DIR}/${run_id}_summary.json")
done

"$NODE_BIN" scripts/select_preview_candidates.js \
  "${candidate_args[@]}" \
  "${baseline_args[@]}" \
  --out "${OUT_DIR}/${SELECTION_ID}_report.json" \
  --selection "${OUT_DIR}/${SELECTION_ID}.json"

echo "Preview selection: ${OUT_DIR}/${SELECTION_ID}.json"
