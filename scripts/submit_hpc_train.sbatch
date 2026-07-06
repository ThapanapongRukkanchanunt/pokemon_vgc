#!/usr/bin/env bash
#SBATCH --job-name=pokemon-rl-p6-1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# ERAWAN defaults:
# - regular gpu partition allows 4 CPU cores per GPU job;
# - use a Python venv with PyTorch/CUDA installed;
# - override VENV_DIR, VALUE_EPOCHS, POLICY_EPOCHS, or batch sizes via sbatch --export.
BUNDLE_DIR="${BUNDLE_DIR:-$SLURM_SUBMIT_DIR}"
OUT_DIR="${OUT_DIR:-$BUNDLE_DIR/outputs}"
DEVICE="${DEVICE:-cuda}"
VENV_DIR="${VENV_DIR:-$BUNDLE_DIR/.venv_torch}"

if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
fi

export PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
export NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
export VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-256}"
export POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-64}"

bash "$BUNDLE_DIR/scripts/hpc_train_iteration.sh" "$BUNDLE_DIR" "$OUT_DIR" "$DEVICE"
