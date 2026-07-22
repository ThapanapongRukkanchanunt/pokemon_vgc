#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NODE_BIN="${NODE_BIN:-node}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INFERENCE_DEVICE="${TORCH_INFERENCE_DEVICE:-cpu}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-mb_alphastar_fairppo90}"
HOLDOUT_ID="${HOLDOUT_ID:-${SOURCE_RUN_ID}_holdout}"
SELECTED_MANIFEST="${SELECTED_MANIFEST:-experiments/mb_alpha_league/${SOURCE_RUN_ID}/selection/${SOURCE_RUN_ID}_checkpoint_selection_manifest.json}"
PREVIEW_SELECTION="${PREVIEW_SELECTION:-experiments/mb_alpha_league/${SOURCE_RUN_ID}/preview_selection/${SOURCE_RUN_ID}_preview_selection.json}"
BASELINE_MODELS_DIR="${BASELINE_MODELS_DIR:-models/torch/${SOURCE_RUN_ID}/iter_010/agents}"
BASELINE_PREVIEW="${BASELINE_PREVIEW:-models/torch/${SOURCE_RUN_ID}/iter_010/universal_preview/checkpoint.pt}"
GAMES_PER_PAIRING="${GAMES_PER_PAIRING:-3}"
SEARCH_GAMES_PER_PAIRING="${SEARCH_GAMES_PER_PAIRING:-1}"
OUT_DIR="${OUT_DIR:-experiments/mb_alpha_league/${SOURCE_RUN_ID}/holdout}"

[[ -f "$SELECTED_MANIFEST" ]] || { echo "Missing selected manifest: $SELECTED_MANIFEST" >&2; exit 1; }
[[ -f "$PREVIEW_SELECTION" ]] || { echo "Missing preview selection: $PREVIEW_SELECTION" >&2; exit 1; }
PREVIEW_MODEL="$($NODE_BIN -e "console.log(require('./${PREVIEW_SELECTION}').checkpoint)")"
[[ -f "$PREVIEW_MODEL" ]] || { echo "Missing selected preview checkpoint: $PREVIEW_MODEL" >&2; exit 1; }
mkdir -p "$OUT_DIR"

selected_args=()
baseline_args=()
for opponent in random maxdamage heuristic; do
  for roster in selected baseline; do
    run_id="${HOLDOUT_ID}_${roster}_${opponent}"
    model_args=(--models-dir "$BASELINE_MODELS_DIR")
    if [[ "$roster" == "selected" ]]; then
      model_args=(--model-manifest "$SELECTED_MANIFEST")
    fi
    "$NODE_BIN" scripts/evaluate_alpha_league_agents.js \
      --run-id "$run_id" \
      --seed "${HOLDOUT_ID}:${opponent}:shared" \
      "${model_args[@]}" \
      --team-preview-model "$PREVIEW_MODEL" \
      --preview-mode learned \
      --opponent-agent "$opponent" \
      --out-dir "$OUT_DIR" \
      --log-dir "logs/battles/${run_id}" \
      --games-per-pairing "$GAMES_PER_PAIRING" \
      --side-swaps \
      --top-k 1 \
      --python "$PYTHON_BIN" \
      --torch-device "$TORCH_INFERENCE_DEVICE" \
      --delete-battle-logs \
      --overwrite
    summary="${OUT_DIR}/${run_id}_summary.json"
    if [[ "$roster" == "selected" ]]; then
      selected_args+=(--selected "${opponent}=${summary}")
    else
      baseline_args+=(--baseline "${opponent}=${summary}")
    fi
  done
done

validated_manifest="${OUT_DIR}/${HOLDOUT_ID}_validated_manifest.json"
report="${OUT_DIR}/${HOLDOUT_ID}_report.json"
final_selection="${OUT_DIR}/${HOLDOUT_ID}_final_team.json"
"$NODE_BIN" scripts/analyze_mb_holdout.js \
  "${selected_args[@]}" \
  "${baseline_args[@]}" \
  --selected-manifest "$SELECTED_MANIFEST" \
  --baseline-models-dir "$BASELINE_MODELS_DIR" \
  --validated-manifest "$validated_manifest" \
  --out "$report" \
  --final-selection "$final_selection"

head_run="${HOLDOUT_ID}_validated_vs_iter010"
"$NODE_BIN" scripts/evaluate_alpha_league_agents.js \
  --run-id "$head_run" \
  --seed "${HOLDOUT_ID}:rl_head_to_head" \
  --model-manifest "$validated_manifest" \
  --team-preview-model "$PREVIEW_MODEL" \
  --preview-mode learned \
  --opponent-agent rl \
  --opponent-models-dir "$BASELINE_MODELS_DIR" \
  --opponent-team-preview-model "$BASELINE_PREVIEW" \
  --opponent-preview-mode learned \
  --out-dir "$OUT_DIR" \
  --log-dir "logs/battles/${head_run}" \
  --games-per-pairing "$GAMES_PER_PAIRING" \
  --side-swaps \
  --top-k 1 \
  --python "$PYTHON_BIN" \
  --torch-device "$TORCH_INFERENCE_DEVICE" \
  --delete-battle-logs \
  --overwrite

for top_k in 1 4; do
  search_run="${HOLDOUT_ID}_validated_topk${top_k}"
  "$NODE_BIN" scripts/evaluate_alpha_league_agents.js \
    --run-id "$search_run" \
    --seed "${HOLDOUT_ID}:topk_shared" \
    --model-manifest "$validated_manifest" \
    --team-preview-model "$PREVIEW_MODEL" \
    --preview-mode learned \
    --opponent-agent heuristic \
    --out-dir "$OUT_DIR" \
    --log-dir "logs/battles/${search_run}" \
    --games-per-pairing "$SEARCH_GAMES_PER_PAIRING" \
    --side-swaps \
    --top-k "$top_k" \
    --rollout-max-decisions 120 \
    --python "$PYTHON_BIN" \
    --torch-device "$TORCH_INFERENCE_DEVICE" \
    --delete-battle-logs \
    --overwrite
done

"$NODE_BIN" scripts/analyze_mb_holdout.js \
  "${selected_args[@]}" \
  "${baseline_args[@]}" \
  --selected-manifest "$SELECTED_MANIFEST" \
  --baseline-models-dir "$BASELINE_MODELS_DIR" \
  --validated-manifest "$validated_manifest" \
  --out "$report" \
  --final-selection "$final_selection" \
  --head-to-head "${OUT_DIR}/${head_run}_summary.json" \
  --search-top1 "${OUT_DIR}/${HOLDOUT_ID}_validated_topk1_summary.json" \
  --search "${OUT_DIR}/${HOLDOUT_ID}_validated_topk4_summary.json"

echo "Holdout report: $report"
echo "Validated manifest: $validated_manifest"
echo "Final team selection: $final_selection"
