#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-phase8_ppo}"
ITERATIONS="${ITERATIONS:-3}"
ROLLOUT_GAMES="${ROLLOUT_GAMES:-100}"
EVAL_GAMES="${EVAL_GAMES:-100}"
NODE_BIN="${NODE_BIN:-node}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
TORCH_INFERENCE_DEVICE="${TORCH_INFERENCE_DEVICE:-cpu}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
TEACHER_MODEL="${TEACHER_MODEL:-models/bc_policy/phase6_search_improved/model.json}"
VALUE_MODEL="${VALUE_MODEL:-models/value_model/phase4_mixed_q/model.json}"
ROLLOUT_OPPONENTS="${ROLLOUT_OPPONENTS:-random,maxdamage,heuristic,search_balanced,hmm_belief}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-64}"
PPO_LR="${PPO_LR:-2e-5}"
OVERWRITE="${OVERWRITE:-1}"

require_file() {
  local file_path="$1"
  local label="$2"
  if [[ ! -f "$file_path" ]]; then
    echo "Missing $label: $file_path" >&2
    exit 1
  fi
}

resolve_init_checkpoint() {
  if [[ -n "$INIT_CHECKPOINT" ]]; then
    echo "$INIT_CHECKPOINT"
    return
  fi

  local candidates=(
    "models/torch/phase6_torch_full/iter_005/policy/checkpoint.pt"
    "models/torch/phase6_torch_full/iter_004/policy/checkpoint.pt"
    "models/torch/phase6_torch_full/iter_003/policy/checkpoint.pt"
    "models/torch/phase6_torch_full/iter_002/policy/checkpoint.pt"
    "models/torch/phase6_torch_full/iter_001/policy/checkpoint.pt"
    "models/torch/phase6_remote_torch/iter_005/policy/checkpoint.pt"
    "models/torch/phase6_remote_torch/iter_004/policy/checkpoint.pt"
    "models/torch/phase6_remote_torch/iter_003/policy/checkpoint.pt"
    "models/torch/phase6_remote_torch/iter_002/policy/checkpoint.pt"
    "models/torch/phase6_remote_torch/iter_001/policy/checkpoint.pt"
    "experiments/torch_smoke/policy/checkpoint.pt"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done

  echo "models/torch/phase6_torch_full/iter_001/policy/checkpoint.pt"
}

iteration_tag() {
  printf "iter_%03d" "$1"
}

node_args_overwrite=()
if [[ "$OVERWRITE" == "1" ]]; then
  node_args_overwrite=(--overwrite)
fi

INIT_CHECKPOINT="$(resolve_init_checkpoint)"
require_file "$INIT_CHECKPOINT" "warm-start checkpoint"
require_file "$TEACHER_MODEL" "teacher/search policy model"
require_file "$VALUE_MODEL" "teacher/search value model"

"$PYTHON_BIN" scripts/check_torch_gpu.py --require-cuda

previous_checkpoint="$INIT_CHECKPOINT"
for ((iteration = 1; iteration <= ITERATIONS; iteration++)); do
  tag="$(iteration_tag "$iteration")"
  rollout_id="${RUN_ID}_${tag}"
  rollout_dataset="data/datasets/rl/${rollout_id}_rollouts.jsonl"
  rollout_summary="data/datasets/rl/${rollout_id}_rollouts.summary.json"
  train_out_dir="models/torch/${RUN_ID}/${tag}/ppo"

  echo
  echo "=== ${RUN_ID} ${tag}: generate PPO rollouts ==="
  "$NODE_BIN" scripts/generate_ppo_rollouts.js \
    --run-id "$rollout_id" \
    --model "$previous_checkpoint" \
    --teacher-model "$TEACHER_MODEL" \
    --value-model "$VALUE_MODEL" \
    --games "$ROLLOUT_GAMES" \
    --opponents "$ROLLOUT_OPPONENTS" \
    --python "$PYTHON_BIN" \
    --torch-device "$TORCH_INFERENCE_DEVICE" \
    "${node_args_overwrite[@]}"

  "$NODE_BIN" scripts/validate_ppo_rollouts.js \
    --dataset "$rollout_dataset" \
    --summary "$rollout_summary"

  echo
  echo "=== ${RUN_ID} ${tag}: train PPO actor-critic ==="
  ppo_args_resume=()
  if [[ "$iteration" -gt 1 ]]; then
    ppo_args_resume=(--resume-optimizer)
  fi
  "$PYTHON_BIN" scripts/train_ppo_torch.py \
    --rollouts "$rollout_dataset" \
    --out-dir "$train_out_dir" \
    --init-checkpoint "$previous_checkpoint" \
    "${ppo_args_resume[@]}" \
    --device "$TRAIN_DEVICE" \
    --epochs "$PPO_EPOCHS" \
    --batch-size "$PPO_BATCH_SIZE" \
    --learning-rate "$PPO_LR" \
    --iteration "$iteration" \
    --overwrite

  previous_checkpoint="${train_out_dir}/checkpoint.pt"
  require_file "$previous_checkpoint" "trained PPO checkpoint"

  if [[ "$EVAL_GAMES" != "0" ]]; then
    echo
    echo "=== ${RUN_ID} ${tag}: evaluate final RL checkpoint ==="
    "$NODE_BIN" scripts/evaluate_selectors.js \
      --model "$TEACHER_MODEL" \
      --rl-model "$previous_checkpoint" \
      --value-model "$VALUE_MODEL" \
      --out-dir "experiments/selectors/${rollout_id}_final_rl_eval" \
      --log-dir "logs/battles/${rollout_id}_final_rl_eval" \
      --games "$EVAL_GAMES" \
      --seed "${rollout_id}_final_rl_eval" \
      --matchups final_rl_vs_random,final_rl_vs_maxdamage,final_rl_vs_heuristic,final_rl_vs_search_balanced,final_rl_vs_hmm_belief \
      --python "$PYTHON_BIN" \
      --torch-device "$TORCH_INFERENCE_DEVICE" \
      --overwrite
    "$NODE_BIN" scripts/validate_battle_traces.js \
      --log-dir "logs/battles/${rollout_id}_final_rl_eval"
  fi
done

echo
echo "PPO loop complete."
echo "Latest final RL checkpoint: $previous_checkpoint"
