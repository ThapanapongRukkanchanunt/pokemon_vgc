#!/usr/bin/env bash
#SBATCH --job-name=poke-gen
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
RUN_ID="${RUN_ID:-phase6_erawan_loop_iter_001}"
GAMES_PER_SHARD="${GAMES_PER_SHARD:-1000}"
NODE_BIN="${NODE_BIN:-node}"

if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

cd "$REPO_ROOT"

SHARD_ID="${SLURM_ARRAY_TASK_ID:-1}"
SHARD_PAD="$(printf "%03d" "$SHARD_ID")"
SHARD_RUN_ID="${RUN_ID}_shard_${SHARD_PAD}"

"$NODE_BIN" scripts/run_large_scale_training.js \
  --run-id "$SHARD_RUN_ID" \
  --iterations 1 \
  --games "$GAMES_PER_SHARD" \
  --stages generate \
  --out-root "experiments/large_scale/${RUN_ID}/shards" \
  --log-dir "logs/battles/${RUN_ID}_train/shard_${SHARD_PAD}" \
  --compact-logs \
  --overwrite
