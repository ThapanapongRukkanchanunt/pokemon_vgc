# Local PC Runbook

Last updated: 2026-06-12

Use this when running on the Windows PC at `C:\Users\thaip\Documents\pokemon_rl`.

This runbook covers local game generation, local CUDA smoke tests, local PyTorch policy eval, and creating the portable bundle for another machine.

## Requirements

- Windows PowerShell.
- The repo at:

```powershell
cd C:\Users\thaip\Documents\pokemon_rl
```

- Bundled Node:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe --version
```

- Python. If `python` is not on PATH, use the bundled Codex Python:

```powershell
C:\Users\thaip\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe --version
```

## Create Local PyTorch Venv

Use CUDA PyTorch if this PC has a CUDA-capable GPU:

```powershell
C:\Users\thaip\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m venv .venv_torch_cuda
.\.venv_torch_cuda\Scripts\python.exe -m pip install --upgrade pip
.\.venv_torch_cuda\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.\.venv_torch_cuda\Scripts\python.exe -m pip install numpy
.\.venv_torch_cuda\Scripts\python.exe scripts\check_torch_gpu.py --require-cuda
```

If CUDA is not available and you only want CPU inference tests:

```powershell
C:\Users\thaip\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m venv .venv_torch_cpu
.\.venv_torch_cpu\Scripts\python.exe -m pip install --upgrade pip
.\.venv_torch_cpu\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv_torch_cpu\Scripts\python.exe -m pip install numpy
.\.venv_torch_cpu\Scripts\python.exe scripts\check_torch_gpu.py
```

## Local CUDA Smoke

```powershell
.\.venv_torch_cuda\Scripts\python.exe scripts\train_value_alphastar_torch.py --dataset data\datasets\value\phase4_mixed_q.jsonl --out-dir experiments\torch_local_cuda_smoke\value --device cuda --limit 128 --epochs 1 --batch-size 32 --vocab-size 4096 --d-model 32 --n-heads 4 --n-layers 1 --max-state-tokens 128 --max-action-tokens 32 --overwrite
```

```powershell
.\.venv_torch_cuda\Scripts\python.exe scripts\train_policy_alphastar_torch.py --dataset data\datasets\search\phase6_search_improved.jsonl --out-dir experiments\torch_local_cuda_smoke\policy --device cuda --limit 128 --epochs 1 --batch-size 8 --vocab-size 4096 --d-model 32 --n-heads 4 --n-layers 1 --max-state-tokens 128 --max-action-tokens 32 --overwrite
```

## Live PyTorch Policy Eval

PyTorch policy checkpoints can drive live battles through `torch_policy`.

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\evaluate_selectors.js --model experiments\torch_local_cuda_smoke\policy\checkpoint.pt --out-dir experiments\selectors\torch_local_cuda_smoke_eval --log-dir logs\battles\torch_local_cuda_smoke_eval --games 10 --seed torch_local_cuda_smoke_eval --matchups torch_policy_vs_random,torch_policy_vs_maxdamage,torch_policy_vs_heuristic --python .\.venv_torch_cuda\Scripts\python.exe --torch-device cpu --overwrite
```

Keep inference on CPU by default with `--torch-device cpu`; that leaves the GPU for training.

Validate eval traces:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\validate_battle_traces.js --log-dir logs\battles\torch_local_cuda_smoke_eval
```

## Manual Iteration Loop On PC

Generate training games and build a value dataset:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\run_large_scale_training.js --run-id phase6_local_torch_iter_001 --iterations 1 --games 10000 --stages generate,value-dataset --train-participants torch_current_policy,random,maxdamage,heuristic --current-model experiments\torch_local_cuda_smoke\policy\checkpoint.pt --python .\.venv_torch_cuda\Scripts\python.exe --torch-device cpu --compact-train-logs --overwrite
```

Build search-improved labels from that value dataset:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\build_search_improved_dataset.js --dataset data\datasets\value\phase6_local_torch_iter_001_value.jsonl --policy-model models\bc_policy\phase6_search_improved\model.json --value-model models\value_model\phase4_mixed_q\model.json --out-dir data\datasets\search --name phase6_local_torch_iter_001_search --progress-every 500 --overwrite
```

Train the PyTorch value checkpoint:

```powershell
.\.venv_torch_cuda\Scripts\python.exe scripts\train_value_alphastar_torch.py --dataset data\datasets\value\phase6_local_torch_iter_001_value.jsonl --out-dir models\torch\phase6_local_torch\iter_001\value --device cuda --epochs 5 --batch-size 256 --overwrite
```

Train the PyTorch policy checkpoint:

```powershell
.\.venv_torch_cuda\Scripts\python.exe scripts\train_policy_alphastar_torch.py --dataset data\datasets\search\phase6_local_torch_iter_001_search.jsonl --out-dir models\torch\phase6_local_torch\iter_001\policy --init-checkpoint experiments\torch_local_cuda_smoke\policy\checkpoint.pt --resume-optimizer --device cuda --epochs 5 --batch-size 64 --overwrite
```

Eval after training and keep replays:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\run_large_scale_training.js --run-id phase6_local_torch_iter_001_post_train --iterations 1 --games 1 --eval-games 100 --stages eval --eval-participants torch_current_policy,random,maxdamage,heuristic --current-model models\torch\phase6_local_torch\iter_001\policy\checkpoint.pt --python .\.venv_torch_cuda\Scripts\python.exe --torch-device cpu --compact-train-logs --overwrite
```

Eval outputs:

```text
experiments/large_scale/phase6_local_torch_iter_001_post_train/eval_summary.json
logs/battles/phase6_local_torch_iter_001_post_train_eval/*.replay.html
logs/battles/phase6_local_torch_iter_001_post_train_eval/*.summary.json
logs/battles/phase6_local_torch_iter_001_post_train_eval/*.trace.jsonl
```

## Create Portable Bundle

Create the bundle for a remote server or ERAWAN:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\prepare_full_training_bundle.js --out-dir experiments\full_training_bundle --overwrite
```

Create the `tar.gz` archive used by the remote runbooks:

```powershell
tar -czf .\experiments\full_training_bundle.tar.gz -C .\experiments\full_training_bundle .
```

The bundle contains:

```text
FULL_TRAINING_RUNBOOK.md
notes/runbook_local_pc.md
notes/runbook_remote_gpu_server.md
notes/runbook_hpc_erawan.md
scripts/
src/
data/teams/
data/datasets/search/phase6_search_improved.jsonl
data/datasets/value/phase4_mixed_q.jsonl
models/bc_policy/phase6_search_improved/
models/value_model/phase4_mixed_q/
vendor/pokemon-showdown/
```
