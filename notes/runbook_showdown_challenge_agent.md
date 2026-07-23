# Pokemon Showdown Challenge Agent

This mode logs in and accepts incoming challenges only when the challenge format
is exactly `gen9championsvgc2026regmb`. It never sends a ladder `/search`
command. Challenges in other formats are rejected.

The process requests `/savereplay` after every completed battle and also writes
a local protocol transcript and standalone HTML replay under the run directory
in `logs/ladder/`.

```bash
cd ~/pokemon_vgc
screen -dmS mb_challenges bash scripts/run_showdown_challenge_agent.sh
```

Attach with `screen -r mb_challenges`. Stop gracefully with `Ctrl-C`; the
launcher writes a final summary and closes the model scorer.
