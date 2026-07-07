# Pokemon VGC M-B AlphaStar League Runbook

This workspace now targets `[Gen 9 Champions] VGC 2026 Reg M-B`.

Active team pool:

- `data/teams/team_pool.json`
- `data/teams/imports/mb_*.txt`
- source snapshot: `data/teams/vgcpastes_mb_source.csv`
- raw Pokepaste snapshot: `data/teams/raw_pokepastes/mb_*.txt`

The old Reg M-A team pool has been removed from the active data path.

## Training Shape

The main runner is:

```bash
bash scripts/run_mb_alphastar_league.sh
```

Default flow:

1. Validate the 10-team M-B pool.
2. Play `10000` random-vs-random games across the 10 teams.
3. Build a BC dataset from random-agent decisions.
4. Train a bootstrap AlphaStar-like legal-action policy.
5. Initialize 10 team agents from that bootstrap checkpoint.
6. Run `10` league iterations.
7. Each iteration plays `1000` PPO league games between random team-agent pairings.
8. Each team agent updates its own specialized PPO model from its own rollout rows.
9. A shared universal preview checkpoint updates from all rollout rows.
10. Each team agent is evaluated against all 10 teams controlled by the random agent.
11. A report is written to `experiments/mb_alpha_league/<RUN_ID>/report.json`.

Important defaults:

```bash
RUN_ID=mb_alphastar_league
BOOTSTRAP_GAMES=10000
ITERATIONS=10
LEAGUE_GAMES=1000
EVAL_GAMES_PER_PAIRING=1
EPSILON_START=0.20
EPSILON_END=0.02
TOP_K=4
ROLLOUT_MAX_DECISIONS=120
DELETE_PLAY_LOGS=1
```

The PPO agent uses epsilon-greedy exploration for both team preview and battle
actions. The shared team-preview checkpoint is passed through
`--team-preview-model`; specialized team checkpoints score battle actions.
For battle actions, the agent takes the model top `TOP_K`, forks the current
Showdown state with `Battle.fromJSON()`, applies each candidate, and simulates
forward with model top-1 choices until the clone ends or reaches
`ROLLOUT_MAX_DECISIONS`. The best rollout result can override the greedy model
choice before epsilon exploration is applied. Team preview uses the universal
preview checkpoint directly.

The neural state encoder includes:

- own roster species, items, abilities, natures, and EV strings;
- opponent team roster from team preview context;
- predicted opponent back slots from the team-pool roster minus visible active mons;
- existing public active HP/species state and HMM belief diagnostics.

## GPU Server Run

Use git for code transfer to the GPU server:

```bash
git remote add origin https://github.com/ThapanapongRukkanchanunt/pokemon_vgc.git
git add .
git commit -m "Switch VGC to Reg M-B AlphaStar league"
git push -u origin main
```

On the GPU server:

```bash
git clone https://github.com/ThapanapongRukkanchanunt/pokemon_vgc.git
cd pokemon_vgc
git pull

python3 -m venv .venv_torch
source .venv_torch/bin/activate
pip install torch
npm ci --prefix vendor/pokemon-showdown
node vendor/pokemon-showdown/build

NODE_BIN=node \
PYTHON_BIN="$PWD/.venv_torch/bin/python" \
TRAIN_DEVICE=cuda \
TORCH_INFERENCE_DEVICE=cuda \
RUN_ID=mb_alphastar_league \
bash scripts/run_mb_alphastar_league.sh
```

Pull results back from the server with `scp`:

```bash
scp -r user@gpu-server:/path/to/pokemon_vgc/experiments/mb_alpha_league/mb_alphastar_league experiments/mb_alpha_league/
scp -r user@gpu-server:/path/to/pokemon_vgc/models/torch/mb_alphastar_league models/torch/
```

If disk is tight on the server, keep:

- `experiments/mb_alpha_league/<RUN_ID>/report.json`
- `experiments/mb_alpha_league/<RUN_ID>/eval/*_summary.json`
- `models/torch/<RUN_ID>/**/checkpoint.pt`
- `models/torch/<RUN_ID>/**/metrics.json`

The runner already deletes per-battle logs by default after extracting rollout
rows. Set `DELETE_ROLLOUTS=1` to remove rollout JSONL files after training each
iteration.

If the run stops after random bootstrap but before bootstrap policy training,
reuse the existing traces/dataset instead of replaying the 10,000 random games:

```bash
NODE_BIN=node \
PYTHON_BIN="$PWD/.venv_torch/bin/python" \
TRAIN_DEVICE=cuda \
TORCH_INFERENCE_DEVICE=cuda \
RUN_ID=mb_alphastar_league \
SKIP_BOOTSTRAP_RANDOM=1 \
SKIP_BOOTSTRAP_BC=1 \
bash scripts/run_mb_alphastar_league.sh
```
