#!/usr/bin/env bash
#SBATCH --job-name=poke-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
TRAIN_RUN_ID="${TRAIN_RUN_ID:-phase6_erawan_loop}"
DATASET_ID="${DATASET_ID:-phase6_erawan_loop_iter_001}"
ITERATION="${ITERATION:-1}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv_torch}"

if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
fi

cd "$REPO_ROOT"

ITER_NUM=$((10#$ITERATION))
ITER_PAD="$(printf "%03d" "$ITER_NUM")"
export ITERATION
export PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
export NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
export VALUE_DATASET="${VALUE_DATASET:-$REPO_ROOT/data/datasets/value/${DATASET_ID}_value.jsonl}"
export POLICY_DATASET="${POLICY_DATASET:-$REPO_ROOT/data/datasets/search/${DATASET_ID}_search.jsonl}"
if [[ -z "${VALUE_CHECKPOINT:-}" ]]; then
  if (( ITER_NUM > 1 )); then
    PREV_PAD="$(printf "%03d" "$((ITER_NUM - 1))")"
    export VALUE_CHECKPOINT="$REPO_ROOT/models/torch/${TRAIN_RUN_ID}/iter_${PREV_PAD}/value/checkpoint.pt"
  else
    export VALUE_CHECKPOINT="$REPO_ROOT/models/torch/${TRAIN_RUN_ID}/iter_${ITER_PAD}/previous_value_checkpoint_missing.pt"
  fi
else
  export VALUE_CHECKPOINT
fi
if [[ -z "${POLICY_CHECKPOINT:-}" ]]; then
  if (( ITER_NUM > 1 )); then
    PREV_PAD="$(printf "%03d" "$((ITER_NUM - 1))")"
    export POLICY_CHECKPOINT="$REPO_ROOT/models/torch/${TRAIN_RUN_ID}/iter_${PREV_PAD}/policy/checkpoint.pt"
  else
    export POLICY_CHECKPOINT="$REPO_ROOT/models/torch/${TRAIN_RUN_ID}/iter_${ITER_PAD}/previous_policy_checkpoint_missing.pt"
  fi
else
  export POLICY_CHECKPOINT
fi

OUT_DIR="${OUT_DIR:-$REPO_ROOT/models/torch/${TRAIN_RUN_ID}/iter_${ITER_PAD}}"

bash scripts/hpc_train_iteration.sh "$REPO_ROOT" "$OUT_DIR" cuda
