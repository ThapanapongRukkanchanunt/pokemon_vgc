# Project Tooling Notes

- This folder is the Pokemon VGC project workspace.
- C++ compilation is available with `g++`.
- Python is available as Python 3.12.10 via:
  `C:\Users\thaip\AppData\Local\Microsoft\WindowsApps\python3.exe`
  (And Python 3.13.2 is available at:
  `C:\Users\CS-DELL-7470\AppData\Local\Programs\Python\Python313\python.exe`
  in this sandbox).
- In the Codex sandbox, `python`, `python3`, and `py` may not resolve from PATH.
  If Python is needed, use one of the full paths above.

## Project Layout

- `src/`: VGC battle agents, selectors, value/policy utilities, belief utilities, and team-building code.
- `scripts/`: local, GPU, and ERAWAN training/evaluation scripts.
- `data/`: teams and datasets used by the VGC training loops.
- `vendor/pokemon-showdown/`: Pokemon Showdown engine checkout used by the simulator.
- `notes/`: local, remote GPU, and ERAWAN runbooks.
- `FULL_TRAINING_RUNBOOK.md`: main VGC full-loop training runbook.

Generated artifacts such as `logs/`, `models/`, `experiments/`, compressed bundles,
and `vendor/pokemon-showdown/node_modules/` are intentionally ignored by Git.
