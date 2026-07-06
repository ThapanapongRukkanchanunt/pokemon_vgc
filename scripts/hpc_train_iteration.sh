#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BUNDLE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BUNDLE_DIR="${1:-${BUNDLE_DIR:-$DEFAULT_BUNDLE_DIR}}"
OUT_DIR="${2:-${OUT_DIR:-$BUNDLE_DIR/outputs}}"
DEVICE="${3:-${DEVICE:-cuda}}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VALUE_EPOCHS="${VALUE_EPOCHS:-5}"
POLICY_EPOCHS="${POLICY_EPOCHS:-5}"
VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-256}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
ITERATION="${ITERATION:-1}"
VOCAB_SIZE="${VOCAB_SIZE:-65536}"
D_MODEL="${D_MODEL:-128}"
N_HEADS="${N_HEADS:-4}"
N_LAYERS="${N_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.1}"
MAX_STATE_TOKENS="${MAX_STATE_TOKENS:-384}"
MAX_ACTION_TOKENS="${MAX_ACTION_TOKENS:-64}"

VALUE_DATASET="${VALUE_DATASET:-$BUNDLE_DIR/data/value_dataset.jsonl}"
POLICY_DATASET="${POLICY_DATASET:-$BUNDLE_DIR/data/search_dataset.jsonl}"
VALUE_CHECKPOINT="${VALUE_CHECKPOINT:-$BUNDLE_DIR/checkpoints/value_checkpoint.pt}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-$BUNDLE_DIR/checkpoints/policy_checkpoint.pt}"

if [[ ! -f "$VALUE_DATASET" ]]; then
  echo "Missing value dataset: $VALUE_DATASET" >&2
  exit 1
fi

if [[ ! -f "$POLICY_DATASET" ]]; then
  echo "Missing policy/search dataset: $POLICY_DATASET" >&2
  exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/check_torch_gpu.py" --require-cuda

VALUE_ARGS=(
  "$SCRIPT_DIR/train_value_alphastar_torch.py"
  --dataset "$VALUE_DATASET"
  --out-dir "$OUT_DIR/value"
  --device "$DEVICE"
  --epochs "$VALUE_EPOCHS"
  --batch-size "$VALUE_BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --num-workers "$NUM_WORKERS"
  --iteration "$ITERATION"
  --vocab-size "$VOCAB_SIZE"
  --d-model "$D_MODEL"
  --n-heads "$N_HEADS"
  --n-layers "$N_LAYERS"
  --dropout "$DROPOUT"
  --max-state-tokens "$MAX_STATE_TOKENS"
  --max-action-tokens "$MAX_ACTION_TOKENS"
  --overwrite
)

if [[ -f "$VALUE_CHECKPOINT" ]]; then
  VALUE_ARGS+=(--init-checkpoint "$VALUE_CHECKPOINT" --resume-optimizer)
fi

POLICY_ARGS=(
  "$SCRIPT_DIR/train_policy_alphastar_torch.py"
  --dataset "$POLICY_DATASET"
  --out-dir "$OUT_DIR/policy"
  --device "$DEVICE"
  --epochs "$POLICY_EPOCHS"
  --batch-size "$POLICY_BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --num-workers "$NUM_WORKERS"
  --iteration "$ITERATION"
  --vocab-size "$VOCAB_SIZE"
  --d-model "$D_MODEL"
  --n-heads "$N_HEADS"
  --n-layers "$N_LAYERS"
  --dropout "$DROPOUT"
  --max-state-tokens "$MAX_STATE_TOKENS"
  --max-action-tokens "$MAX_ACTION_TOKENS"
  --overwrite
)

if [[ -f "$POLICY_CHECKPOINT" ]]; then
  POLICY_ARGS+=(--init-checkpoint "$POLICY_CHECKPOINT" --resume-optimizer)
fi

"$PYTHON_BIN" "${VALUE_ARGS[@]}"
"$PYTHON_BIN" "${POLICY_ARGS[@]}"

echo "Wrote GPU training outputs:"
echo "  $OUT_DIR/value/checkpoint.pt"
echo "  $OUT_DIR/policy/checkpoint.pt"
