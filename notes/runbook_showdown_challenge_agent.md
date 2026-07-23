# Pokemon Showdown Challenge Agent

This mode logs in and accepts incoming challenges only when the challenge format
is exactly `gen9championsvgc2026regmb`. It never sends a ladder `/search`
command. Challenges in other formats are rejected.

The process requests `/savereplay` after every completed battle and also writes
a local protocol transcript and standalone HTML replay under the run directory
in `logs/ladder/`.

```bash
screen -dmS mb_challenges bash -lc '
  cd ~/pokemon_vgc &&
  source .venv_torch/bin/activate &&
  while true; do
    node scripts/run_showdown_ladder.js \
      --mode challenge \
      --package models/torch/final_mb_agent \
      --credentials showdown.env \
      --games 0 \
      --python .venv_torch/bin/python \
      --torch-device cpu
    status=$?
    if [ "$status" -eq 0 ]; then break; fi
    sleep 10
  done
'
```

Attach with `screen -r mb_challenges`. Stop gracefully with `Ctrl-C`; the
launcher writes a final summary and closes the model scorer.
