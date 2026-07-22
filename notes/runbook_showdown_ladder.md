# Pokemon Showdown Ladder Runbook

The ladder client targets `gen9championsvgc2026regmb`, uses the packaged team,
accepts open team sheets, enables the battle timer, and stops after the bounded
`--games` count.

## Setup

```bash
cd ~/pokemon_vgc
git pull --ff-only
source .venv_torch/bin/activate
cp showdown.env.example showdown.env
chmod 600 showdown.env
```

Fill `SHOWDOWN_USERNAME` and `SHOWDOWN_PASSWORD` in `showdown.env`. This file is
ignored by Git. Do not pass the password on the command line.

## Smoke Tests

```bash
node scripts/check_showdown_ladder_client.js
node scripts/check_public_state.js
node scripts/check_showdown_connection.js
```

The connection smoke stops after receiving the anonymous `challstr`; it does
not log in or search for a battle.

## Package

After final holdout selection:

```bash
node scripts/package_final_agent.js \
  --team-selection experiments/mb_alpha_league/FINAL_RUN/holdout/FINAL_TEAM.json \
  --preview-selection experiments/mb_alpha_league/FINAL_RUN/preview_selection/PREVIEW.json \
  --out-dir models/torch/final_mb_agent \
  --overwrite
```

`manifest.json` records the selected team, source checkpoints, inference mode,
and SHA-256 hashes for both checkpoints and the team import.

## One Ladder Game

```bash
screen -dmS mb_ladder bash -lc '
  cd ~/pokemon_vgc &&
  source .venv_torch/bin/activate &&
  node scripts/run_showdown_ladder.js \
    --package models/torch/final_mb_agent \
    --credentials showdown.env \
    --games 1 \
    --python .venv_torch/bin/python \
    --torch-device cpu
'
```

Monitor with `screen -r mb_ladder`. Ladder protocol and decision logs are
written under `logs/ladder/`.

Live play uses deterministic top-1 inference. Top-k rollout search is evaluated
offline because a live client does not have Showdown's hidden simulator state.
