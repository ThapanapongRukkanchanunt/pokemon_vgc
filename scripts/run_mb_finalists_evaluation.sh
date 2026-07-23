#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NODE_BIN="${NODE_BIN:-node}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INFERENCE_DEVICE="${TORCH_INFERENCE_DEVICE:-cpu}"
RUN_ID="${RUN_ID:-mb_alphastar_fairppo90_pfsp}"
FINALISTS_ID="${FINALISTS_ID:-${RUN_ID}_finalists}"
SOURCE_HOLDOUT_REPORT="${SOURCE_HOLDOUT_REPORT:-experiments/mb_alpha_league/${RUN_ID}/final_holdout/${RUN_ID}_final_holdout_report.json}"
SELECTED_MANIFEST="${SELECTED_MANIFEST:-experiments/mb_alpha_league/${RUN_ID}/final_holdout/${RUN_ID}_final_holdout_validated_manifest.json}"
PREVIEW_SELECTION="${PREVIEW_SELECTION:-experiments/mb_alpha_league/${RUN_ID}/preview_selection/${RUN_ID}_preview_selection.json}"
BASELINE_MODELS_DIR="${BASELINE_MODELS_DIR:-models/torch/${RUN_ID}/iter_010/agents}"
FINALIST_TEAMS="${FINALIST_TEAMS:-}"
GAMES_PER_PAIRING="${GAMES_PER_PAIRING:-10}"
REUSE_EXISTING="${REUSE_EXISTING:-1}"
OUT_DIR="${OUT_DIR:-experiments/mb_alpha_league/${RUN_ID}/finalists}"

for required in "$SOURCE_HOLDOUT_REPORT" "$SELECTED_MANIFEST" "$PREVIEW_SELECTION"; do
  [[ -f "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
if [[ -z "$FINALIST_TEAMS" ]]; then
  FINALIST_TEAMS="$($NODE_BIN -e "const r=require('./${SOURCE_HOLDOUT_REPORT}'); console.log(r.final_team_ranking.slice(0,2).map(x=>x.team_id).join(','))")"
fi
[[ -n "$FINALIST_TEAMS" ]] || { echo "No finalist teams were selected" >&2; exit 1; }
PREVIEW_MODEL="$($NODE_BIN -e "console.log(require('./${PREVIEW_SELECTION}').checkpoint)")"
[[ -f "$PREVIEW_MODEL" ]] || { echo "Missing selected preview checkpoint: $PREVIEW_MODEL" >&2; exit 1; }
mkdir -p "$OUT_DIR"

selected_args=()
baseline_args=()
for opponent in random maxdamage heuristic; do
  for roster in selected baseline; do
    eval_id="${FINALISTS_ID}_${roster}_${opponent}"
    summary="${OUT_DIR}/${eval_id}_summary.json"
    model_args=(--models-dir "$BASELINE_MODELS_DIR")
    if [[ "$roster" == "selected" ]]; then
      model_args=(--model-manifest "$SELECTED_MANIFEST")
    fi
    if [[ "$REUSE_EXISTING" != "1" || ! -f "$summary" ]]; then
      "$NODE_BIN" scripts/evaluate_alpha_league_agents.js \
        --run-id "$eval_id" \
        --seed "${FINALISTS_ID}:${opponent}:shared" \
        "${model_args[@]}" \
        --agent-teams "$FINALIST_TEAMS" \
        --team-preview-model "$PREVIEW_MODEL" \
        --preview-mode learned \
        --opponent-agent "$opponent" \
        --out-dir "$OUT_DIR" \
        --log-dir "logs/battles/${eval_id}" \
        --games-per-pairing "$GAMES_PER_PAIRING" \
        --side-swaps \
        --top-k 1 \
        --python "$PYTHON_BIN" \
        --torch-device "$TORCH_INFERENCE_DEVICE" \
        --delete-battle-logs \
        --overwrite
    fi
    if [[ "$roster" == "selected" ]]; then
      selected_args+=(--selected "${opponent}=${summary}")
    else
      baseline_args+=(--baseline "${opponent}=${summary}")
    fi
  done
done

report="${OUT_DIR}/${FINALISTS_ID}_report.json"
validated_manifest="${OUT_DIR}/${FINALISTS_ID}_validated_manifest.json"
final_selection="${OUT_DIR}/${FINALISTS_ID}_final_team.json"
"$NODE_BIN" scripts/analyze_mb_holdout.js \
  "${selected_args[@]}" \
  "${baseline_args[@]}" \
  --selected-manifest "$SELECTED_MANIFEST" \
  --baseline-models-dir "$BASELINE_MODELS_DIR" \
  --validated-manifest "$validated_manifest" \
  --out "$report" \
  --final-selection "$final_selection"

echo "Finalists: $FINALIST_TEAMS"
echo "Finalist report: $report"
echo "Final team selection: $final_selection"
