#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-phase6_torch_full}"
ITERATIONS="${ITERATIONS:-5}"
GAMES="${GAMES:-10000}"
EVAL_GAMES="${EVAL_GAMES:-100}"
NODE_BIN="${NODE_BIN:-node}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
TORCH_INFERENCE_DEVICE="${TORCH_INFERENCE_DEVICE:-cpu}"
TRAIN_PARTICIPANTS="${TRAIN_PARTICIPANTS:-torch_current_policy,random,maxdamage,heuristic}"
EVAL_PARTICIPANTS="${EVAL_PARTICIPANTS:-torch_current_policy,random,maxdamage,heuristic}"
SEARCH_POLICY_MODEL="${SEARCH_POLICY_MODEL:-models/bc_policy/phase6_search_improved/model.json}"
SEARCH_VALUE_MODEL="${SEARCH_VALUE_MODEL:-models/value_model/phase4_mixed_q/model.json}"
BOOTSTRAP_POLICY_DATASET="${BOOTSTRAP_POLICY_DATASET:-data/datasets/search/phase6_search_improved.jsonl}"
BOOTSTRAP_POLICY_CHECKPOINT="${BOOTSTRAP_POLICY_CHECKPOINT:-}"
BOOTSTRAP_POLICY_EPOCHS="${BOOTSTRAP_POLICY_EPOCHS:-3}"
VALUE_EPOCHS="${VALUE_EPOCHS:-5}"
POLICY_EPOCHS="${POLICY_EPOCHS:-5}"
VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-256}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-64}"
OVERWRITE="${OVERWRITE:-1}"
SEARCH_PROGRESS_EVERY="${SEARCH_PROGRESS_EVERY:-500}"

require_file() {
  local file_path="$1"
  local label="$2"
  if [[ ! -f "$file_path" ]]; then
    echo "Missing $label: $file_path" >&2
    exit 1
  fi
}

iteration_tag() {
  printf "iter_%03d" "$1"
}

node_args_overwrite=()
if [[ "$OVERWRITE" == "1" ]]; then
  node_args_overwrite=(--overwrite)
fi

require_file "$SEARCH_POLICY_MODEL" "Node JSON policy model for search-label building"
require_file "$SEARCH_VALUE_MODEL" "Node JSON value model for search-label building"

"$PYTHON_BIN" scripts/check_torch_gpu.py --require-cuda

if [[ -z "$BOOTSTRAP_POLICY_CHECKPOINT" ]]; then
  require_file "$BOOTSTRAP_POLICY_DATASET" "bootstrap PyTorch policy dataset"
  BOOTSTRAP_DIR="models/torch/${RUN_ID}/bootstrap/policy"
  echo "Bootstrapping PyTorch policy checkpoint: $BOOTSTRAP_DIR"
  "$PYTHON_BIN" scripts/train_policy_alphastar_torch.py \
    --dataset "$BOOTSTRAP_POLICY_DATASET" \
    --out-dir "$BOOTSTRAP_DIR" \
    --device "$TRAIN_DEVICE" \
    --epochs "$BOOTSTRAP_POLICY_EPOCHS" \
    --batch-size "$POLICY_BATCH_SIZE" \
    --overwrite
  BOOTSTRAP_POLICY_CHECKPOINT="$BOOTSTRAP_DIR/checkpoint.pt"
fi

require_file "$BOOTSTRAP_POLICY_CHECKPOINT" "bootstrap PyTorch policy checkpoint"

previous_policy_checkpoint="$BOOTSTRAP_POLICY_CHECKPOINT"
previous_value_checkpoint=""

for ((iteration = 1; iteration <= ITERATIONS; iteration++)); do
  tag="$(iteration_tag "$iteration")"
  dataset_id="${RUN_ID}_${tag}"
  train_run_id="${dataset_id}"
  train_out_dir="models/torch/${RUN_ID}/${tag}"

  echo
  echo "=== ${RUN_ID} ${tag}: play training games and build value dataset ==="
  "$NODE_BIN" scripts/run_large_scale_training.js \
    --run-id "$train_run_id" \
    --iterations 1 \
    --games "$GAMES" \
    --stages generate,value-dataset \
    --train-participants "$TRAIN_PARTICIPANTS" \
    --current-model "$previous_policy_checkpoint" \
    --python "$PYTHON_BIN" \
    --torch-device "$TORCH_INFERENCE_DEVICE" \
    --compact-train-logs \
    "${node_args_overwrite[@]}"

  value_dataset="data/datasets/value/${dataset_id}_value.jsonl"
  search_dataset="data/datasets/search/${dataset_id}_search.jsonl"
  search_summary="data/datasets/search/${dataset_id}_search.summary.json"
  require_file "$value_dataset" "iteration value dataset"

  echo
  echo "=== ${RUN_ID} ${tag}: build search-improved policy dataset ==="
  if [[ "$OVERWRITE" != "1" && -f "$search_dataset" && -f "$search_summary" ]]; then
    echo "search dataset exists; skipping rebuild ($search_dataset)"
  else
    "$NODE_BIN" scripts/build_search_improved_dataset.js \
      --dataset "$value_dataset" \
      --policy-model "$SEARCH_POLICY_MODEL" \
      --value-model "$SEARCH_VALUE_MODEL" \
      --out-dir data/datasets/search \
      --name "${dataset_id}_search" \
      --progress-every "$SEARCH_PROGRESS_EVERY" \
      "${node_args_overwrite[@]}"
  fi
  "$NODE_BIN" scripts/validate_bc_dataset.js \
    --dataset "$search_dataset" \
    --summary "$search_summary"

  echo
  echo "=== ${RUN_ID} ${tag}: train PyTorch value and policy checkpoints ==="
  export ITERATION="$iteration"
  export PYTHON_BIN
  export VALUE_DATASET="$value_dataset"
  export POLICY_DATASET="$search_dataset"
  export VALUE_EPOCHS
  export POLICY_EPOCHS
  export VALUE_BATCH_SIZE
  export POLICY_BATCH_SIZE
  export POLICY_CHECKPOINT="$previous_policy_checkpoint"
  if [[ -n "$previous_value_checkpoint" ]]; then
    export VALUE_CHECKPOINT="$previous_value_checkpoint"
  else
    unset VALUE_CHECKPOINT || true
  fi
  bash scripts/hpc_train_iteration.sh "$REPO_ROOT" "$train_out_dir" "$TRAIN_DEVICE"

  previous_value_checkpoint="${train_out_dir}/value/checkpoint.pt"
  previous_policy_checkpoint="${train_out_dir}/policy/checkpoint.pt"
  require_file "$previous_value_checkpoint" "trained value checkpoint"
  require_file "$previous_policy_checkpoint" "trained policy checkpoint"

  if [[ "$EVAL_GAMES" != "0" ]]; then
    eval_run_id="${dataset_id}_post_train"
    echo
    echo "=== ${RUN_ID} ${tag}: eval trained PyTorch policy with replay output ==="
    "$NODE_BIN" scripts/run_large_scale_training.js \
      --run-id "$eval_run_id" \
      --iterations 1 \
      --games 1 \
      --eval-games "$EVAL_GAMES" \
      --stages eval \
      --eval-participants "$EVAL_PARTICIPANTS" \
      --current-model "$previous_policy_checkpoint" \
      --python "$PYTHON_BIN" \
      --torch-device "$TORCH_INFERENCE_DEVICE" \
      --compact-train-logs \
      "${node_args_overwrite[@]}"
    "$NODE_BIN" scripts/validate_battle_traces.js \
      --log-dir "logs/battles/${eval_run_id}_eval"
  fi
done

echo
echo "Full PyTorch loop complete."
echo "Latest policy checkpoint: $previous_policy_checkpoint"
echo "Latest value checkpoint: $previous_value_checkpoint"
