#!/usr/bin/env bash
#SBATCH --job-name=poke-node
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
NODE_BIN="${NODE_BIN:-node}"

if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

cd "$REPO_ROOT"
NODE_BIN="$NODE_BIN" bash scripts/check_erawan_node.sh
