#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

NODE_BIN="${NODE_BIN:-node}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
TORCH_INFERENCE_DEVICE="${TORCH_INFERENCE_DEVICE:-cpu}"

SOURCE_RUN_ID="${SOURCE_RUN_ID:-mb_alphastar_fairppo90}"
RUN_ID="${RUN_ID:-mb_alphastar_fairppo90_pfsp}"
START_MODEL_MANIFEST="${START_MODEL_MANIFEST:-experiments/mb_alpha_league/${SOURCE_RUN_ID}/selection/${SOURCE_RUN_ID}_checkpoint_selection_manifest.json}"
START_PREVIEW_MODEL="${START_PREVIEW_MODEL:-models/torch/${SOURCE_RUN_ID}/iter_010/universal_preview/checkpoint.pt}"
SOURCE_PREVIEW_DATASET="${SOURCE_PREVIEW_DATASET:-data/datasets/preview/${SOURCE_RUN_ID}_preview_replay.jsonl}"

START_ITERATION="${START_ITERATION:-11}"
ITERATIONS="${ITERATIONS:-5}"
LEAGUE_GAMES="${LEAGUE_GAMES:-1000}"
HISTORICAL_PROBABILITY="${HISTORICAL_PROBABILITY:-0.35}"
PFSP_EXPONENT="${PFSP_EXPONENT:-2}"
PFSP_PRIOR_GAMES="${PFSP_PRIOR_GAMES:-2}"
EPSILON="${EPSILON:-0.02}"
TOP_K="${TOP_K:-4}"
ROLLOUT_MAX_DECISIONS="${ROLLOUT_MAX_DECISIONS:-120}"

PPO_EPOCHS="${PPO_EPOCHS:-4}"
PPO_BATCH_SIZE="${PPO_BATCH_SIZE:-64}"
PPO_LR="${PPO_LR:-2e-5}"
PPO_TARGET_KL="${PPO_TARGET_KL:-0.01}"
PREVIEW_EPOCHS="${PREVIEW_EPOCHS:-2}"
PREVIEW_BATCH_SIZE="${PREVIEW_BATCH_SIZE:-64}"
PREVIEW_LR="${PREVIEW_LR:-3e-4}"
EVAL_GAMES_PER_PAIRING="${EVAL_GAMES_PER_PAIRING:-3}"

EXPERIMENT_DIR="experiments/mb_alpha_league/${RUN_ID}"
PREVIEW_DATASET="data/datasets/preview/${RUN_ID}_preview_replay.jsonl"
HISTORY_DIR="${EXPERIMENT_DIR}/history"
INITIAL_TAG="iter_$(printf '%03d' $((START_ITERATION - 1)))"
INITIAL_MODELS_DIR="models/torch/${RUN_ID}/${INITIAL_TAG}/agents"
INITIAL_PREVIEW_DIR="models/torch/${RUN_ID}/${INITIAL_TAG}/universal_preview"

[[ -f "$START_MODEL_MANIFEST" ]] || { echo "Missing selected model manifest: $START_MODEL_MANIFEST" >&2; exit 1; }
[[ -f "$START_PREVIEW_MODEL" ]] || { echo "Missing selected preview checkpoint: $START_PREVIEW_MODEL" >&2; exit 1; }
[[ -f "$SOURCE_PREVIEW_DATASET" ]] || { echo "Missing source preview dataset: $SOURCE_PREVIEW_DATASET" >&2; exit 1; }
mkdir -p "$EXPERIMENT_DIR/eval" "$HISTORY_DIR" "$INITIAL_MODELS_DIR" "$INITIAL_PREVIEW_DIR"

mapfile -t TEAM_IDS < <("$NODE_BIN" -e "const p=require('./data/teams/team_pool.json'); for (const t of p.teams) console.log(t.id);")
mapfile -t SELECTED_ROWS < <("$NODE_BIN" -e "const m=require('./${START_MODEL_MANIFEST}'); for (const [id,e] of Object.entries(m.models||m)) console.log(id+'\\t'+(typeof e==='string'?e:e.checkpoint));")
declare -A SELECTED_MODELS
for row in "${SELECTED_ROWS[@]}"; do
  team_id="${row%%$'\t'*}"
  SELECTED_MODELS["$team_id"]="${row#*$'\t'}"
done
for team_id in "${TEAM_IDS[@]}"; do
  checkpoint="${SELECTED_MODELS[$team_id]:-}"
  [[ -f "$checkpoint" ]] || { echo "Missing selected checkpoint for ${team_id}: ${checkpoint}" >&2; exit 1; }
  mkdir -p "${INITIAL_MODELS_DIR}/${team_id}"
  ln -f "$checkpoint" "${INITIAL_MODELS_DIR}/${team_id}/checkpoint.pt"
done
ln -f "$START_PREVIEW_MODEL" "${INITIAL_PREVIEW_DIR}/checkpoint.pt"
cp "$SOURCE_PREVIEW_DATASET" "$PREVIEW_DATASET"

CURRENT_MODELS_DIR="$INITIAL_MODELS_DIR"
CURRENT_PREVIEW_MODEL="${INITIAL_PREVIEW_DIR}/checkpoint.pt"
final_iteration=$((START_ITERATION + ITERATIONS - 1))

for ((iteration = START_ITERATION; iteration <= final_iteration; iteration++)); do
  tag="iter_$(printf '%03d' "$iteration")"
  iteration_id="${RUN_ID}_${tag}"
  history_manifest="${HISTORY_DIR}/${tag}_history.json"
  snapshot_args=()
  for source_iteration in 006 007 008 009 010; do
    snapshot_args+=(--snapshot "${SOURCE_RUN_ID}_${source_iteration}=models/torch/${SOURCE_RUN_ID}/iter_${source_iteration}/agents,models/torch/${SOURCE_RUN_ID}/iter_${source_iteration}/universal_preview/checkpoint.pt")
  done
  for ((prior = START_ITERATION - 1; prior < iteration; prior++)); do
    prior_tag="iter_$(printf '%03d' "$prior")"
    snapshot_args+=(--snapshot "${RUN_ID}_${prior_tag}=models/torch/${RUN_ID}/${prior_tag}/agents,models/torch/${RUN_ID}/${prior_tag}/universal_preview/checkpoint.pt")
  done
  "$NODE_BIN" scripts/create_league_history_manifest.js --out "$history_manifest" "${snapshot_args[@]}"

  echo "=== ${RUN_ID} ${tag}: PFSP historical league (${LEAGUE_GAMES} games) ==="
  "$NODE_BIN" scripts/generate_alpha_league_rollouts.js \
    --run-id "$iteration_id" \
    --models-dir "$CURRENT_MODELS_DIR" \
    --team-preview-model "$CURRENT_PREVIEW_MODEL" \
    --history-manifest "$history_manifest" \
    --historical-probability "$HISTORICAL_PROBABILITY" \
    --pfsp-exponent "$PFSP_EXPONENT" \
    --pfsp-prior-games "$PFSP_PRIOR_GAMES" \
    --games "$LEAGUE_GAMES" \
    --epsilon "$EPSILON" \
    --top-k "$TOP_K" \
    --rollout-max-decisions "$ROLLOUT_MAX_DECISIONS" \
    --python "$PYTHON_BIN" \
    --torch-device "$TORCH_INFERENCE_DEVICE" \
    --log-dir "logs/battles/${iteration_id}_league" \
    --compact-logs \
    --delete-battle-logs \
    --overwrite

  all_rollouts="data/datasets/rl/${iteration_id}_all_rollouts.jsonl"
  "$NODE_BIN" scripts/validate_ppo_rollouts.js \
    --dataset "$all_rollouts" \
    --summary "data/datasets/rl/${iteration_id}_all_rollouts.validation.json"
  "$NODE_BIN" scripts/build_team_preview_dataset.js \
    --rollouts "$all_rollouts" \
    --out "$PREVIEW_DATASET" \
    --append

  next_models_dir="models/torch/${RUN_ID}/${tag}/agents"
  next_preview_dir="models/torch/${RUN_ID}/${tag}/universal_preview"
  "$PYTHON_BIN" scripts/train_value_alphastar_torch.py \
    --dataset "$PREVIEW_DATASET" \
    --out-dir "$next_preview_dir" \
    --init-checkpoint "$CURRENT_PREVIEW_MODEL" \
    --resume-optimizer \
    --device "$TRAIN_DEVICE" \
    --epochs "$PREVIEW_EPOCHS" \
    --batch-size "$PREVIEW_BATCH_SIZE" \
    --learning-rate "$PREVIEW_LR" \
    --group-validation-by battle_id \
    --iteration "$iteration" \
    --overwrite

  mkdir -p "$next_models_dir"
  for team_id in "${TEAM_IDS[@]}"; do
    team_rollouts="data/datasets/rl/${iteration_id}_${team_id}_rollouts.jsonl"
    [[ -s "$team_rollouts" ]] || { echo "Missing rollout rows for ${team_id}" >&2; exit 1; }
    "$PYTHON_BIN" scripts/train_ppo_torch.py \
      --rollouts "$team_rollouts" \
      --out-dir "${next_models_dir}/${team_id}" \
      --init-checkpoint "${CURRENT_MODELS_DIR}/${team_id}/checkpoint.pt" \
      --resume-optimizer \
      --device "$TRAIN_DEVICE" \
      --epochs "$PPO_EPOCHS" \
      --batch-size "$PPO_BATCH_SIZE" \
      --learning-rate "$PPO_LR" \
      --target-kl "$PPO_TARGET_KL" \
      --exclude-request-types team_preview \
      --iteration "$iteration" \
      --overwrite
  done

  "$NODE_BIN" scripts/evaluate_alpha_league_agents.js \
    --run-id "${iteration_id}_eval" \
    --models-dir "$next_models_dir" \
    --team-preview-model "${next_preview_dir}/checkpoint.pt" \
    --preview-mode learned \
    --opponent-agent random \
    --out-dir "${EXPERIMENT_DIR}/eval" \
    --log-dir "logs/battles/${iteration_id}_eval" \
    --games-per-pairing "$EVAL_GAMES_PER_PAIRING" \
    --side-swaps \
    --top-k 1 \
    --python "$PYTHON_BIN" \
    --torch-device "$TORCH_INFERENCE_DEVICE" \
    --delete-battle-logs \
    --overwrite

  rm -f "data/datasets/rl/${iteration_id}"*_rollouts.jsonl
  CURRENT_MODELS_DIR="$next_models_dir"
  CURRENT_PREVIEW_MODEL="${next_preview_dir}/checkpoint.pt"
done

"$NODE_BIN" scripts/collect_alpha_league_report.js \
  --run-id "$RUN_ID" \
  --start-iteration "$START_ITERATION" \
  --iterations "$ITERATIONS" \
  --out-dir "$EXPERIMENT_DIR"

echo "PFSP continuation complete: ${EXPERIMENT_DIR}/report.json"
