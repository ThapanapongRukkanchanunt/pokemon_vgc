#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NODE_BIN="${NODE_BIN:-node}"
PYTHON_BIN="${PYTHON_BIN:-.venv_torch/bin/python}"
TORCH_INFERENCE_DEVICE="${TORCH_INFERENCE_DEVICE:-cuda}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-mb_alphastar_fairppo90_pfsp}"
AGENT_TEAM="${AGENT_TEAM:-mb-004}"
ITERATION="${ITERATION:-iter_015}"
MODELS_DIR="${MODELS_DIR:-models/torch/${SOURCE_RUN_ID}/${ITERATION}/agents}"
PREVIEW_MODEL="${PREVIEW_MODEL:-models/torch/${SOURCE_RUN_ID}/iter_010/universal_preview/checkpoint.pt}"
GAMES_PER_PAIRING="${GAMES_PER_PAIRING:-5}"
RUN_PREFIX="${RUN_PREFIX:-${AGENT_TEAM}_mega_policy_pilot}"
OUT_DIR="${OUT_DIR:-experiments/mb_alpha_league/${SOURCE_RUN_ID}/mega_policy_eval}"

[[ -f "${MODELS_DIR}/${AGENT_TEAM}/checkpoint.pt" ]] || {
  echo "Missing battle checkpoint: ${MODELS_DIR}/${AGENT_TEAM}/checkpoint.pt" >&2
  exit 1
}
[[ -f "$PREVIEW_MODEL" ]] || {
  echo "Missing preview checkpoint: $PREVIEW_MODEL" >&2
  exit 1
}
mkdir -p "$OUT_DIR"

model_args=()
guarded_args=()
for opponent in random maxdamage heuristic; do
  for policy in model sole_usable; do
    run_id="${RUN_PREFIX}_${policy}_${opponent}"
    "$NODE_BIN" scripts/evaluate_alpha_league_agents.js \
      --run-id "$run_id" \
      --seed "${RUN_PREFIX}:${opponent}:shared" \
      --models-dir "$MODELS_DIR" \
      --agent-teams "$AGENT_TEAM" \
      --team-preview-model "$PREVIEW_MODEL" \
      --preview-mode learned \
      --opponent-agent "$opponent" \
      --out-dir "$OUT_DIR" \
      --log-dir "logs/battles/${run_id}" \
      --games-per-pairing "$GAMES_PER_PAIRING" \
      --side-swaps \
      --top-k 1 \
      --mega-policy "$policy" \
      --python "$PYTHON_BIN" \
      --torch-device "$TORCH_INFERENCE_DEVICE" \
      --delete-battle-logs \
      --overwrite
    summary="${OUT_DIR}/${run_id}_summary.json"
    if [[ "$policy" == "model" ]]; then
      model_args+=(--model "${opponent}=${summary}")
    else
      guarded_args+=(--guarded "${opponent}=${summary}")
    fi
  done
done

report="${OUT_DIR}/${RUN_PREFIX}_report.json"
"$NODE_BIN" scripts/analyze_mega_policy_eval.js \
  "${model_args[@]}" \
  "${guarded_args[@]}" \
  --out "$report"

echo "Mega policy evaluation report: $report"
