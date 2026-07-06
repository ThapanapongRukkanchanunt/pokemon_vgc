#!/usr/bin/env bash
#SBATCH --job-name=poke-data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
RUN_ID="${RUN_ID:-phase6_erawan_loop_iter_001}"
POLICY_MODEL="${POLICY_MODEL:-models/bc_policy/phase6_search_improved/model.json}"
VALUE_MODEL="${VALUE_MODEL:-models/value_model/phase4_mixed_q/model.json}"
NODE_BIN="${NODE_BIN:-node}"

if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

cd "$REPO_ROOT"

"$NODE_BIN" scripts/build_value_dataset.js \
  --trace-dir "logs/battles/${RUN_ID}_train" \
  --out-dir data/datasets/value \
  --name "${RUN_ID}_value" \
  --overwrite

"$NODE_BIN" scripts/validate_value_dataset.js \
  --dataset "data/datasets/value/${RUN_ID}_value.jsonl" \
  --summary "data/datasets/value/${RUN_ID}_value.summary.json"

"$NODE_BIN" scripts/build_search_improved_dataset.js \
  --dataset "data/datasets/value/${RUN_ID}_value.jsonl" \
  --policy-model "$POLICY_MODEL" \
  --value-model "$VALUE_MODEL" \
  --out-dir data/datasets/search \
  --name "${RUN_ID}_search" \
  --progress-every 500 \
  --overwrite

"$NODE_BIN" scripts/validate_bc_dataset.js \
  --dataset "data/datasets/search/${RUN_ID}_search.jsonl" \
  --summary "data/datasets/search/${RUN_ID}_search.summary.json"
