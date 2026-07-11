#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-mb_alphastar_league}"
NODE_BIN="${NODE_BIN:-node}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
TORCH_INFERENCE_DEVICE="${TORCH_INFERENCE_DEVICE:-cpu}"

BOOTSTRAP_GAMES="${BOOTSTRAP_GAMES:-10000}"
BOOTSTRAP_BC_EPOCHS="${BOOTSTRAP_BC_EPOCHS:-5}"
BOOTSTRAP_BATCH_SIZE="${BOOTSTRAP_BATCH_SIZE:-64}"
BOOTSTRAP_LR="${BOOTSTRAP_LR:-3e-4}"

ITERATIONS="${ITERATIONS:-10}"
LEAGUE_GAMES="${LEAGUE_GAMES:-1000}"
EVAL_GAMES_PER_PAIRING="${EVAL_GAMES_PER_PAIRING:-1}"
EPSILON_START="${EPSILON_START:-0.20}"
EPSILON_END="${EPSILON_END:-0.02}"
TOP_K="${TOP_K:-4}"
ROLLOUT_MAX_DECISIONS="${ROLLOUT_MAX_DECISIONS:-120}"

PPO_EPOCHS="${PPO_EPOCHS:-2}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-64}"
PPO_LR="${PPO_LR:-2e-5}"
PREVIEW_VALUE_EPOCHS="${PREVIEW_VALUE_EPOCHS:-${UNIVERSAL_PREVIEW_EPOCHS:-2}}"
PREVIEW_VALUE_BATCH_SIZE="${PREVIEW_VALUE_BATCH_SIZE:-64}"
PREVIEW_VALUE_LR="${PREVIEW_VALUE_LR:-3e-4}"

OVERWRITE="${OVERWRITE:-1}"
COMPACT_LOGS="${COMPACT_LOGS:-1}"
DELETE_PLAY_LOGS="${DELETE_PLAY_LOGS:-1}"
DELETE_BOOTSTRAP_LOGS="${DELETE_BOOTSTRAP_LOGS:-0}"
DELETE_ROLLOUTS="${DELETE_ROLLOUTS:-0}"
SKIP_BOOTSTRAP_RANDOM="${SKIP_BOOTSTRAP_RANDOM:-0}"
SKIP_BOOTSTRAP_BC="${SKIP_BOOTSTRAP_BC:-0}"
SKIP_BOOTSTRAP_POLICY="${SKIP_BOOTSTRAP_POLICY:-0}"
SKIP_BOOTSTRAP_PREVIEW_DATASET="${SKIP_BOOTSTRAP_PREVIEW_DATASET:-0}"
SKIP_BOOTSTRAP_PREVIEW_MODEL="${SKIP_BOOTSTRAP_PREVIEW_MODEL:-0}"

EXPERIMENT_DIR="experiments/mb_alpha_league/${RUN_ID}"
BOOTSTRAP_DIR="${EXPERIMENT_DIR}/bootstrap"
BOOTSTRAP_LOG_DIR="logs/battles/${RUN_ID}_bootstrap_random"
BOOTSTRAP_BC_DATASET="data/datasets/bc/${RUN_ID}_bootstrap_bc.jsonl"
BOOTSTRAP_CHECKPOINT="models/torch/${RUN_ID}/bootstrap/policy/checkpoint.pt"
PREVIEW_REPLAY_DATASET="data/datasets/preview/${RUN_ID}_preview_replay.jsonl"
BOOTSTRAP_PREVIEW_CHECKPOINT="models/torch/${RUN_ID}/bootstrap/universal_preview/checkpoint.pt"

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_args=(--overwrite)
fi

compact_args=()
if [[ "$COMPACT_LOGS" == "1" ]]; then
  compact_args=(--compact-logs)
fi

delete_play_args=()
if [[ "$DELETE_PLAY_LOGS" == "1" ]]; then
  delete_play_args=(--delete-battle-logs)
fi

require_file() {
  local file_path="$1"
  local label="$2"
  if [[ ! -f "$file_path" ]]; then
    echo "Missing ${label}: ${file_path}" >&2
    exit 1
  fi
}

iteration_tag() {
  printf "iter_%03d" "$1"
}

epsilon_for_iteration() {
  "$PYTHON_BIN" - "$1" "$ITERATIONS" "$EPSILON_START" "$EPSILON_END" <<'PY'
import sys
iteration = int(sys.argv[1])
iterations = max(1, int(sys.argv[2]))
start = float(sys.argv[3])
end = float(sys.argv[4])
if iterations == 1:
    value = end
else:
    progress = (iteration - 1) / (iterations - 1)
    value = start + (end - start) * progress
print(f"{value:.6f}")
PY
}

mapfile -t TEAM_IDS < <("$NODE_BIN" -e "const p=require('./data/teams/team_pool.json'); for (const t of p.teams) console.log(t.id);")
if [[ "${#TEAM_IDS[@]}" -ne 10 ]]; then
  echo "Expected exactly 10 teams in data/teams/team_pool.json; got ${#TEAM_IDS[@]}" >&2
  exit 1
fi

echo "=== ${RUN_ID}: validate M-B team pool ==="
"$NODE_BIN" scripts/validate_team_pool.js

echo
if [[ "$SKIP_BOOTSTRAP_RANDOM" == "1" ]]; then
  echo "=== ${RUN_ID}: skip random bootstrap; using existing traces in ${BOOTSTRAP_LOG_DIR} ==="
  if [[ ! -d "$BOOTSTRAP_LOG_DIR" ]]; then
    echo "Missing bootstrap trace directory: ${BOOTSTRAP_LOG_DIR}" >&2
    exit 1
  fi
else
  echo "=== ${RUN_ID}: random bootstrap ${BOOTSTRAP_GAMES} games ==="
  "$NODE_BIN" scripts/generate_random_team_league.js \
    --run-id "${RUN_ID}_bootstrap_random" \
    --games "$BOOTSTRAP_GAMES" \
    --out-dir "$BOOTSTRAP_DIR" \
    --log-dir "$BOOTSTRAP_LOG_DIR" \
    --seed "${RUN_ID}:bootstrap" \
    --progress-every 100 \
    "${compact_args[@]}" \
    "${overwrite_args[@]}"
fi

echo
if [[ "$SKIP_BOOTSTRAP_BC" == "1" ]]; then
  echo "=== ${RUN_ID}: skip bootstrap BC build; using ${BOOTSTRAP_BC_DATASET} ==="
else
  echo "=== ${RUN_ID}: build bootstrap BC dataset ==="
  "$NODE_BIN" scripts/build_bc_dataset.js \
    --trace-dir "$BOOTSTRAP_LOG_DIR" \
    --out-dir data/datasets/bc \
    --name "${RUN_ID}_bootstrap_bc" \
    --agent random_agent \
    --exclude-team-preview \
    --overwrite
fi
require_file "$BOOTSTRAP_BC_DATASET" "bootstrap BC dataset"

echo
if [[ "$SKIP_BOOTSTRAP_PREVIEW_DATASET" == "1" ]]; then
  echo "=== ${RUN_ID}: skip bootstrap preview dataset; using ${PREVIEW_REPLAY_DATASET} ==="
else
  echo "=== ${RUN_ID}: build 90-action team-preview value replay ==="
  "$NODE_BIN" scripts/build_team_preview_dataset.js \
    --trace-dir "$BOOTSTRAP_LOG_DIR" \
    --out "$PREVIEW_REPLAY_DATASET" \
    --overwrite
fi
require_file "$PREVIEW_REPLAY_DATASET" "team-preview replay dataset"

if [[ "$DELETE_BOOTSTRAP_LOGS" == "1" && "$SKIP_BOOTSTRAP_BC" != "1" ]]; then
  echo "Deleting bootstrap battle logs: $BOOTSTRAP_LOG_DIR"
  rm -rf "$BOOTSTRAP_LOG_DIR"
fi

echo
if [[ "$SKIP_BOOTSTRAP_POLICY" == "1" ]]; then
  echo "=== ${RUN_ID}: skip bootstrap policy training; using ${BOOTSTRAP_CHECKPOINT} ==="
else
  echo "=== ${RUN_ID}: train bootstrap policy ==="
  "$PYTHON_BIN" scripts/train_policy_alphastar_torch.py \
    --dataset "$BOOTSTRAP_BC_DATASET" \
    --out-dir "models/torch/${RUN_ID}/bootstrap/policy" \
    --device "$TRAIN_DEVICE" \
    --epochs "$BOOTSTRAP_BC_EPOCHS" \
    --batch-size "$BOOTSTRAP_BATCH_SIZE" \
    --learning-rate "$BOOTSTRAP_LR" \
    --seed "${RUN_ID}_bootstrap_policy" \
    --overwrite
fi
require_file "$BOOTSTRAP_CHECKPOINT" "bootstrap policy checkpoint"

echo
if [[ "$SKIP_BOOTSTRAP_PREVIEW_MODEL" == "1" ]]; then
  echo "=== ${RUN_ID}: skip bootstrap preview value training; using ${BOOTSTRAP_PREVIEW_CHECKPOINT} ==="
else
  echo "=== ${RUN_ID}: train bootstrap 90-action team-preview value model ==="
  "$PYTHON_BIN" scripts/train_value_alphastar_torch.py \
    --dataset "$PREVIEW_REPLAY_DATASET" \
    --out-dir "models/torch/${RUN_ID}/bootstrap/universal_preview" \
    --device "$TRAIN_DEVICE" \
    --epochs "$PREVIEW_VALUE_EPOCHS" \
    --batch-size "$PREVIEW_VALUE_BATCH_SIZE" \
    --learning-rate "$PREVIEW_VALUE_LR" \
    --group-validation-by battle_id \
    --seed "${RUN_ID}_bootstrap_preview" \
    --overwrite
fi
require_file "$BOOTSTRAP_PREVIEW_CHECKPOINT" "bootstrap team-preview value checkpoint"

CURRENT_MODELS_DIR="models/torch/${RUN_ID}/iter_000/agents"
mkdir -p "$CURRENT_MODELS_DIR"
for team_id in "${TEAM_IDS[@]}"; do
  mkdir -p "${CURRENT_MODELS_DIR}/${team_id}"
  cp "$BOOTSTRAP_CHECKPOINT" "${CURRENT_MODELS_DIR}/${team_id}/checkpoint.pt"
done

PREVIEW_MODEL="$BOOTSTRAP_PREVIEW_CHECKPOINT"

for ((iteration = 1; iteration <= ITERATIONS; iteration++)); do
  tag="$(iteration_tag "$iteration")"
  iteration_id="${RUN_ID}_${tag}"
  epsilon="$(epsilon_for_iteration "$iteration")"
  next_models_dir="models/torch/${RUN_ID}/${tag}/agents"
  preview_out_dir="models/torch/${RUN_ID}/${tag}/universal_preview"
  preview_next="${preview_out_dir}/checkpoint.pt"

  echo
  echo "=== ${RUN_ID} ${tag}: league PPO rollouts (${LEAGUE_GAMES} games, epsilon=${epsilon}) ==="
  "$NODE_BIN" scripts/generate_alpha_league_rollouts.js \
    --run-id "$iteration_id" \
    --models-dir "$CURRENT_MODELS_DIR" \
    --team-preview-model "$PREVIEW_MODEL" \
    --games "$LEAGUE_GAMES" \
    --epsilon "$epsilon" \
    --top-k "$TOP_K" \
    --rollout-max-decisions "$ROLLOUT_MAX_DECISIONS" \
    --python "$PYTHON_BIN" \
    --torch-device "$TORCH_INFERENCE_DEVICE" \
    --log-dir "logs/battles/${iteration_id}_league" \
    "${compact_args[@]}" \
    "${delete_play_args[@]}" \
    "${overwrite_args[@]}"

  "$NODE_BIN" scripts/validate_ppo_rollouts.js \
    --dataset "data/datasets/rl/${iteration_id}_all_rollouts.jsonl" \
    --summary "data/datasets/rl/${iteration_id}_all_rollouts.validation.json"

  "$NODE_BIN" scripts/build_team_preview_dataset.js \
    --rollouts "data/datasets/rl/${iteration_id}_all_rollouts.jsonl" \
    --out "$PREVIEW_REPLAY_DATASET" \
    --append

  echo
  echo "=== ${RUN_ID} ${tag}: update universal 90-action team-preview value model ==="
  "$PYTHON_BIN" scripts/train_value_alphastar_torch.py \
    --dataset "$PREVIEW_REPLAY_DATASET" \
    --out-dir "$preview_out_dir" \
    --init-checkpoint "$PREVIEW_MODEL" \
    --resume-optimizer \
    --device "$TRAIN_DEVICE" \
    --epochs "$PREVIEW_VALUE_EPOCHS" \
    --batch-size "$PREVIEW_VALUE_BATCH_SIZE" \
    --learning-rate "$PREVIEW_VALUE_LR" \
    --group-validation-by battle_id \
    --iteration "$iteration" \
    --overwrite
  require_file "$preview_next" "universal preview checkpoint"

  echo
  echo "=== ${RUN_ID} ${tag}: update 10 specialized team agents ==="
  mkdir -p "$next_models_dir"
  for team_id in "${TEAM_IDS[@]}"; do
    team_rollouts="data/datasets/rl/${iteration_id}_${team_id}_rollouts.jsonl"
    require_file "$team_rollouts" "rollouts for ${team_id}"
    resume_args=()
    if [[ "$iteration" -gt 1 ]]; then
      resume_args=(--resume-optimizer)
    fi
    "$PYTHON_BIN" scripts/train_ppo_torch.py \
      --rollouts "$team_rollouts" \
      --out-dir "${next_models_dir}/${team_id}" \
      --init-checkpoint "${CURRENT_MODELS_DIR}/${team_id}/checkpoint.pt" \
      "${resume_args[@]}" \
      --device "$TRAIN_DEVICE" \
      --epochs "$PPO_EPOCHS" \
      --batch-size "$PPO_BATCH_SIZE" \
      --learning-rate "$PPO_LR" \
      --exclude-request-types team_preview \
      --iteration "$iteration" \
      --overwrite
    require_file "${next_models_dir}/${team_id}/checkpoint.pt" "trained checkpoint for ${team_id}"
  done

  echo
  echo "=== ${RUN_ID} ${tag}: evaluate team agents vs random teams ==="
  "$NODE_BIN" scripts/evaluate_alpha_league_agents.js \
    --run-id "${iteration_id}_eval" \
    --models-dir "$next_models_dir" \
    --team-preview-model "$preview_next" \
    --out-dir "${EXPERIMENT_DIR}/eval" \
    --log-dir "logs/battles/${iteration_id}_eval" \
    --games-per-pairing "$EVAL_GAMES_PER_PAIRING" \
    --python "$PYTHON_BIN" \
    --torch-device "$TORCH_INFERENCE_DEVICE" \
    "${compact_args[@]}" \
    "${delete_play_args[@]}" \
    "${overwrite_args[@]}"

  if [[ "$DELETE_ROLLOUTS" == "1" ]]; then
    echo "Deleting rollout JSONL files for ${iteration_id}"
    rm -f "data/datasets/rl/${iteration_id}"*_rollouts.jsonl
  fi

  CURRENT_MODELS_DIR="$next_models_dir"
  PREVIEW_MODEL="$preview_next"
done

echo
echo "=== ${RUN_ID}: collect report ==="
"$NODE_BIN" scripts/collect_alpha_league_report.js \
  --run-id "$RUN_ID" \
  --iterations "$ITERATIONS" \
  --out-dir "$EXPERIMENT_DIR"

echo
echo "M-B AlphaStar-style league complete."
echo "Latest specialized agent dir: ${CURRENT_MODELS_DIR}"
echo "Latest universal preview checkpoint: ${PREVIEW_MODEL}"
echo "Report: ${EXPERIMENT_DIR}/report.json"
