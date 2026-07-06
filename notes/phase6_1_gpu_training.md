# Phase 6.1 GPU Training Handoff

Last updated: 2026-06-12

This runbook uses the tested PyTorch AlphaStar-like architecture:

```text
state/action tokenization
  -> transformer state encoder
  -> action embedding/projection
  -> legal-action policy scorer and action-value scorer
```

The architecture has been smoke-tested on CPU with:

- direct policy/value forward and backward passes;
- tiny policy training;
- tiny value training;
- incremental resume with `--init-checkpoint --resume-optimizer`.

Current live-model status:

- PyTorch policy checkpoints can now drive live Showdown battles through the `torch_policy` agent.
- The live PyTorch bridge uses `scripts/torch_policy_server.py` as a persistent scorer under Node.
- Full battle eval summaries and replay HTML are available for PyTorch policy checkpoints.
- Value checkpoints are still training artifacts only. The live policy/value/search selector path still uses the Node JSON value model until a PyTorch value bridge is added.
- Search-label building still consumes Node JSON policy/value models for now.

## Canonical Runbooks

Use these split runbooks for actual operation:

```text
notes/runbook_local_pc.md
notes/runbook_remote_gpu_server.md
notes/runbook_hpc_erawan.md
```

This file remains the Phase 6.1 handoff and reference map.

## Eval Summaries And Replays

For live PyTorch policy eval after each model-training iteration, point the eval runner at the policy `checkpoint.pt` and use the `torch_policy` matchup.

Local CPU inference smoke after training:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\evaluate_selectors.js --model models\torch\phase6_gpu_loop\iter_001\policy\checkpoint.pt --out-dir experiments\selectors\torch_iter_001_eval --log-dir logs\battles\torch_iter_001_eval --games 100 --seed torch_iter_001_eval --matchups torch_policy_vs_random,torch_policy_vs_maxdamage,torch_policy_vs_heuristic --python .\.venv_torch_cuda\Scripts\python.exe --torch-device cpu --overwrite
```

One direct PyTorch policy battle:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\run_random_battle.js --p1-agent torch_policy --p1-model models\torch\phase6_gpu_loop\iter_001\policy\checkpoint.pt --p1-python .\.venv_torch_cuda\Scripts\python.exe --p1-torch-device cpu --p2-agent random --seed torch_policy_live_check --log-dir logs\battles\torch_policy_live_check
```

Outputs:

```text
experiments/selectors/torch_iter_001_eval/summary.json
logs/battles/torch_iter_001_eval/*.summary.json
logs/battles/torch_iter_001_eval/*.replay.html
logs/battles/torch_iter_001_eval/*.trace.jsonl
```

The larger iterative runner can also run PyTorch policy eval by using the `torch_current_policy` participant:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\run_large_scale_training.js --run-id phase6_torch_iter_001_eval --iterations 1 --games 20 --eval-games 100 --stages eval --eval-participants torch_current_policy,random,maxdamage,heuristic --current-model models\torch\phase6_gpu_loop\iter_001\policy\checkpoint.pt --python .\.venv_torch_cuda\Scripts\python.exe --torch-device cpu --compact-train-logs --overwrite
```

For the older Node JSON loop, use the Node-compatible iterative pipeline with `--eval-games`.

Use `--compact-train-logs`, not `--compact-logs`, so training protocol/replay files are deleted but eval replay HTML is kept:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\run_large_scale_training.js --run-id phase6_eval_loop --iterations 5 --games 1000 --eval-games 100 --compact-train-logs --value-epochs 5 --policy-epochs 10 --eval-every 5 --overwrite
```

Outputs after each iteration:

```text
experiments/large_scale/phase6_eval_loop/iter_001/eval_summary.json
logs/battles/phase6_eval_loop_iter_001_eval/*.summary.json
logs/battles/phase6_eval_loop_iter_001_eval/*.replay.html
logs/battles/phase6_eval_loop_iter_001_eval/*.trace.jsonl
```

Do not use this when you need eval replay HTML:

```powershell
--compact-logs
```

That option removes replay HTML for both train and eval games.

For a tiny eval/replay smoke test:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\run_large_scale_training.js --run-id phase6_eval_smoke --iterations 1 --games 20 --eval-games 6 --compact-train-logs --value-epochs 1 --policy-epochs 1 --eval-every 1 --overwrite
```

Expected smoke outputs:

```text
experiments/large_scale/phase6_eval_smoke/iter_001/eval_summary.json
logs/battles/phase6_eval_smoke_iter_001_eval/*.replay.html
```

The replay-preserving eval flag was smoke-tested with:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\run_large_scale_training.js --run-id phase6_eval_replay_flag_smoke --iterations 1 --games 1 --eval-games 2 --stages eval --eval-participants random,maxdamage --compact-train-logs --overwrite
```

It kept:

```text
logs/battles/phase6_eval_replay_flag_smoke_eval/*.replay.html
experiments/large_scale/phase6_eval_replay_flag_smoke/eval_summary.json
```

For ERAWAN, the same Node-compatible eval loop can be submitted as a CPU job once Node is available:

```bash
sbatch --partition=cpu --cpus-per-task=4 --time=24:00:00 --wrap="cd $PWD && ${NODE_BIN:-node} scripts/run_large_scale_training.js --run-id phase6_erawan_eval_loop --iterations 5 --games 1000 --eval-games 100 --compact-train-logs --value-epochs 5 --policy-epochs 10 --eval-every 5 --overwrite"
```

For ERAWAN PyTorch checkpoint eval on CPU nodes, activate the venv, then run:

```bash
sbatch --partition=cpu --cpus-per-task=4 --time=24:00:00 --wrap="cd $PWD && source .venv_torch/bin/activate && ${NODE_BIN:-node} scripts/evaluate_selectors.js --model models/torch/phase6_erawan_loop/iter_001/policy/checkpoint.pt --out-dir experiments/selectors/phase6_erawan_iter_001_eval --log-dir logs/battles/phase6_erawan_iter_001_eval --games 100 --seed phase6_erawan_iter_001_eval --matchups torch_policy_vs_random,torch_policy_vs_maxdamage,torch_policy_vs_heuristic --python $PWD/.venv_torch/bin/python --torch-device cpu --overwrite"
```

## Shared Model Defaults

The default architecture used by both local GPU and ERAWAN commands is:

```text
VOCAB_SIZE=65536
D_MODEL=128
N_HEADS=4
N_LAYERS=2
DROPOUT=0.1
MAX_STATE_TOKENS=384
MAX_ACTION_TOKENS=64
```

For a larger ERAWAN A100/H100 test, use:

```bash
D_MODEL=256 N_HEADS=8 N_LAYERS=4 MAX_STATE_TOKENS=512 MAX_ACTION_TOKENS=96 VALUE_BATCH_SIZE=256 POLICY_BATCH_SIZE=64
```

Keep `D_MODEL` divisible by `N_HEADS`.

## Portable Full-Loop Bundle

Create a portable bundle that can run `play -> train -> eval -> repeat` on another machine:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\prepare_full_training_bundle.js --out-dir experiments\full_training_bundle --overwrite
```

The bundle includes:

```text
scripts/run_torch_full_loop.sh
scripts/torch_policy_server.py
scripts/train_policy_alphastar_torch.py
scripts/train_value_alphastar_torch.py
scripts/run_large_scale_training.js
src/
data/teams/
data/datasets/search/phase6_search_improved.jsonl
data/datasets/value/phase4_mixed_q.jsonl
models/bc_policy/phase6_search_improved/
models/value_model/phase4_mixed_q/
vendor/pokemon-showdown/
FULL_TRAINING_RUNBOOK.md
notes/runbook_local_pc.md
notes/runbook_remote_gpu_server.md
notes/runbook_hpc_erawan.md
```

Run the bundle smoke on Linux/ERAWAN:

```bash
RUN_ID=torch_bundle_smoke ITERATIONS=1 GAMES=2 EVAL_GAMES=2 VALUE_EPOCHS=1 POLICY_EPOCHS=1 BOOTSTRAP_POLICY_EPOCHS=1 bash scripts/run_torch_full_loop.sh
```

Run the full loop:

```bash
RUN_ID=phase6_torch_full ITERATIONS=5 GAMES=10000 EVAL_GAMES=100 VALUE_EPOCHS=5 POLICY_EPOCHS=5 bash scripts/run_torch_full_loop.sh
```

Copy the bundle to ERAWAN:

```powershell
tar -czf .\experiments\full_training_bundle.tar.gz -C .\experiments\full_training_bundle .
scp .\experiments\full_training_bundle.tar.gz erawan:~/pokemon_rl_full.tar.gz
ssh erawan "mkdir -p ~/pokemon_rl_full && tar -xzf ~/pokemon_rl_full.tar.gz -C ~/pokemon_rl_full"
```

Create the server venv over SSH:

```powershell
ssh erawan "cd ~/pokemon_rl_full && python3 -m venv .venv_torch"
ssh erawan "cd ~/pokemon_rl_full && bash -lc 'source .venv_torch/bin/activate && python -m pip install --upgrade pip && python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 && python -m pip install numpy'"
```

Start the full loop on ERAWAN:

```bash
cd ~/pokemon_rl_full
sbatch --partition=gpu --gpus=1 --cpus-per-task=4 --time=72:00:00 --wrap="cd $PWD && source .venv_torch/bin/activate && RUN_ID=phase6_erawan_torch ITERATIONS=5 GAMES=10000 EVAL_GAMES=100 VALUE_EPOCHS=5 POLICY_EPOCHS=5 PYTHON_BIN=$PWD/.venv_torch/bin/python NODE_BIN=${NODE_BIN:-node} bash scripts/run_torch_full_loop.sh"
```

## Local GPU PC

Create a CUDA PyTorch environment on Windows:

```powershell
python -m venv .venv_torch_cuda
.\.venv_torch_cuda\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.\.venv_torch_cuda\Scripts\python.exe -m pip install numpy
.\.venv_torch_cuda\Scripts\python.exe scripts\check_torch_gpu.py --require-cuda
```

If `python` is not on PATH inside this project shell, use the Codex bundled Python:

```powershell
C:\Users\thaip\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m venv .venv_torch_cuda
.\.venv_torch_cuda\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121
.\.venv_torch_cuda\Scripts\python.exe -m pip install numpy
.\.venv_torch_cuda\Scripts\python.exe scripts\check_torch_gpu.py --require-cuda
```

Run a tiny local CUDA architecture smoke test:

```powershell
.\.venv_torch_cuda\Scripts\python.exe scripts\train_value_alphastar_torch.py --dataset data\datasets\value\phase4_mixed_q.jsonl --out-dir experiments\torch_local_cuda_smoke\value --device cuda --limit 128 --epochs 1 --batch-size 32 --vocab-size 4096 --d-model 32 --n-heads 4 --n-layers 1 --max-state-tokens 128 --max-action-tokens 32 --overwrite
```

```powershell
.\.venv_torch_cuda\Scripts\python.exe scripts\train_policy_alphastar_torch.py --dataset data\datasets\search\phase6_search_improved.jsonl --out-dir experiments\torch_local_cuda_smoke\policy --device cuda --limit 128 --epochs 1 --batch-size 8 --vocab-size 4096 --d-model 32 --n-heads 4 --n-layers 1 --max-state-tokens 128 --max-action-tokens 32 --overwrite
```

Generate local training games and value dataset:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\run_large_scale_training.js --run-id phase6_gpu_loop_iter_001 --iterations 1 --games 10000 --stages generate,value-dataset --compact-train-logs --overwrite
```

Build search labels from the generated value dataset using existing Node-compatible policy/value models:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\build_search_improved_dataset.js --dataset data\datasets\value\phase6_gpu_loop_iter_001_value.jsonl --policy-model models\bc_policy\phase6_search_improved\model.json --value-model models\value_model\phase4_mixed_q\model.json --out-dir data\datasets\search --name phase6_gpu_loop_iter_001_search --progress-every 500 --overwrite
```

Train iteration 1 with the PyTorch architecture on local CUDA:

```powershell
.\.venv_torch_cuda\Scripts\python.exe scripts\train_value_alphastar_torch.py --dataset data\datasets\value\phase6_gpu_loop_iter_001_value.jsonl --out-dir models\torch\phase6_gpu_loop\iter_001\value --device cuda --epochs 5 --batch-size 256 --overwrite
```

```powershell
.\.venv_torch_cuda\Scripts\python.exe scripts\train_policy_alphastar_torch.py --dataset data\datasets\search\phase6_gpu_loop_iter_001_search.jsonl --out-dir models\torch\phase6_gpu_loop\iter_001\policy --device cuda --epochs 5 --batch-size 64 --overwrite
```

Train iteration 2 incrementally:

```powershell
.\.venv_torch_cuda\Scripts\python.exe scripts\train_value_alphastar_torch.py --dataset data\datasets\value\phase6_gpu_loop_iter_002_value.jsonl --out-dir models\torch\phase6_gpu_loop\iter_002\value --init-checkpoint models\torch\phase6_gpu_loop\iter_001\value\checkpoint.pt --resume-optimizer --device cuda --epochs 5 --batch-size 256 --overwrite
```

```powershell
.\.venv_torch_cuda\Scripts\python.exe scripts\train_policy_alphastar_torch.py --dataset data\datasets\search\phase6_gpu_loop_iter_002_search.jsonl --out-dir models\torch\phase6_gpu_loop\iter_002\policy --init-checkpoint models\torch\phase6_gpu_loop\iter_001\policy\checkpoint.pt --resume-optimizer --device cuda --epochs 5 --batch-size 64 --overwrite
```

## ERAWAN Setup

ERAWAN jobs should be submitted through Slurm rather than run directly on the login node.

Check the cluster:

```bash
sinfo
scontrol show partition gpu
```

Current ERAWAN regular GPU defaults for this project:

```text
partition: gpu
gpus: 1
cpus-per-task: 4
```

Check whether Node is available:

```bash
node --version
```

If Node is not available, put a Linux Node binary in your home directory and export `NODE_BIN`:

```bash
export NODE_BIN=$HOME/node-v24.14.0-linux-x64/bin/node
export PATH=$HOME/node-v24.14.0-linux-x64/bin:$PATH
node --version
npm --version
```

If the server cannot download Node itself, download the Linux x64 Node archive on your PC and beam it over:

```powershell
scp C:\Users\thaip\Downloads\node-v24.14.0-linux-x64.tar.xz erawan:~/
ssh erawan "tar -xf ~/node-v24.14.0-linux-x64.tar.xz -C ~/"
ssh erawan 'echo "export NODE_BIN=$HOME/node-v24.14.0-linux-x64/bin/node" >> ~/.bashrc'
ssh erawan 'echo "export PATH=$HOME/node-v24.14.0-linux-x64/bin:$PATH" >> ~/.bashrc'
```

Install or repair Pokemon Showdown's runtime Node dependencies:

```bash
cd ~/pokemon_rl_full/vendor/pokemon-showdown
if [ -f package-lock.json ]; then npm ci --omit=dev --omit=optional; else npm install ts-chacha20@1.2.0 --no-save --package-lock=false --no-audit --no-fund; fi
test -d node_modules/ts-chacha20 && echo ts-chacha20_ok
cd ~/pokemon_rl_full
```

Create the PyTorch venv:

```bash
python3 -m venv .venv_torch
source .venv_torch/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy
python scripts/check_torch_gpu.py --require-cuda
```

## Copy Files With SCP

ERAWAN's SSH host is:

```text
erawan.cmu.ac.th
```

CMU account usernames may contain `@`, so the least annoying setup is to add an SSH alias on your PC.

Create or edit this file:

```powershell
notepad $env:USERPROFILE\.ssh\config
```

Add:

```text
Host erawan
  HostName erawan.cmu.ac.th
  User your_cmu_account@cmu.ac.th
```

Test login:

```powershell
ssh erawan
```

Create a project folder on ERAWAN:

```powershell
ssh erawan "mkdir -p ~/pokemon_rl"
```

Copy the whole repo to ERAWAN:

```powershell
cd C:\Users\thaip\Documents\pokemon_rl
tar -czf C:\Users\thaip\Documents\pokemon_rl_repo.tar.gz -C C:\Users\thaip\Documents\pokemon_rl .
scp C:\Users\thaip\Documents\pokemon_rl_repo.tar.gz erawan:~/pokemon_rl_repo.tar.gz
ssh erawan "mkdir -p ~/pokemon_rl && tar -xzf ~/pokemon_rl_repo.tar.gz -C ~/pokemon_rl"
```

Copy only a prepared GPU bundle to ERAWAN:

```powershell
cd C:\Users\thaip\Documents\pokemon_rl
tar -czf phase6_gpu_loop_iter_001.tar.gz -C experiments\gpu_bundles\phase6_gpu_loop_iter_001 .
scp phase6_gpu_loop_iter_001.tar.gz erawan:~/phase6_gpu_loop_iter_001.tar.gz
ssh erawan "mkdir -p ~/pokemon_rl/experiments/gpu_bundles/phase6_gpu_loop_iter_001 && tar -xzf ~/phase6_gpu_loop_iter_001.tar.gz -C ~/pokemon_rl/experiments/gpu_bundles/phase6_gpu_loop_iter_001"
```

Copy only code and small model/data scaffolding, skipping large generated logs, by packaging first:

```powershell
tar -czf pokemon_rl_code.tar.gz AGENTS.md notes scripts src data/teams models/bc_policy/phase6_search_improved models/value_model/phase4_mixed_q
scp pokemon_rl_code.tar.gz erawan:~/pokemon_rl/
ssh erawan "cd ~/pokemon_rl && tar -xzf pokemon_rl_code.tar.gz"
```

Pull trained PyTorch checkpoints back to the PC:

```powershell
ssh erawan "cd ~/pokemon_rl && tar -czf ~/phase6_erawan_loop_models.tar.gz models/torch/phase6_erawan_loop"
scp erawan:~/phase6_erawan_loop_models.tar.gz C:\Users\thaip\Documents\pokemon_rl\experiments\
tar -xzf C:\Users\thaip\Documents\pokemon_rl\experiments\phase6_erawan_loop_models.tar.gz -C C:\Users\thaip\Documents\pokemon_rl
```

Pull one ERAWAN training output folder back:

```powershell
ssh erawan "cd ~/pokemon_rl && tar -czf ~/phase6_erawan_loop_iter_001.tar.gz models/torch/phase6_erawan_loop/iter_001"
scp erawan:~/phase6_erawan_loop_iter_001.tar.gz C:\Users\thaip\Documents\pokemon_rl\experiments\
tar -xzf C:\Users\thaip\Documents\pokemon_rl\experiments\phase6_erawan_loop_iter_001.tar.gz -C C:\Users\thaip\Documents\pokemon_rl
```

If you do not want an SSH config alias, use `-o User=...`:

```powershell
scp -o User=your_cmu_account@cmu.ac.th phase6_gpu_loop_iter_001.tar.gz erawan.cmu.ac.th:~/phase6_gpu_loop_iter_001.tar.gz
ssh -o User=your_cmu_account@cmu.ac.th erawan.cmu.ac.th "mkdir -p ~/pokemon_rl/experiments/gpu_bundles/phase6_gpu_loop_iter_001 && tar -xzf ~/phase6_gpu_loop_iter_001.tar.gz -C ~/pokemon_rl/experiments/gpu_bundles/phase6_gpu_loop_iter_001"
```

For external access through the ERAWAN tunnel, keep this tunnel open in one terminal:

```powershell
ssh -N -L 2222:erawan.cmu.ac.th:22 user_from_email@tunnel.hpc.cmu.ac.th
```

Then use `scp` through localhost port `2222` in another terminal:

```powershell
scp -P 2222 -o User=your_erawan_user phase6_gpu_loop_iter_001.tar.gz localhost:~/phase6_gpu_loop_iter_001.tar.gz
ssh -p 2222 your_erawan_user@localhost "mkdir -p ~/pokemon_rl/experiments/gpu_bundles/phase6_gpu_loop_iter_001 && tar -xzf ~/phase6_gpu_loop_iter_001.tar.gz -C ~/pokemon_rl/experiments/gpu_bundles/phase6_gpu_loop_iter_001"
```

Pull checkpoints back through the same tunnel:

```powershell
ssh -p 2222 your_erawan_user@localhost "cd ~/pokemon_rl && tar -czf ~/phase6_erawan_loop_models.tar.gz models/torch/phase6_erawan_loop"
scp -P 2222 -o User=your_erawan_user localhost:~/phase6_erawan_loop_models.tar.gz C:\Users\thaip\Documents\pokemon_rl\experiments\
tar -xzf C:\Users\thaip\Documents\pokemon_rl\experiments\phase6_erawan_loop_models.tar.gz -C C:\Users\thaip\Documents\pokemon_rl
```

## ERAWAN-Native Loop

This avoids copying large JSONL datasets between the PC and ERAWAN.

Run the Node smoke check through Slurm:

```bash
sbatch --export=ALL,NODE_BIN=${NODE_BIN:-node} scripts/submit_erawan_node_check.sbatch
```

Generate 10,000 games as ten CPU-array shards, two shards running at a time:

```bash
sbatch --array=1-10%2 --export=ALL,RUN_ID=phase6_erawan_loop_iter_001,GAMES_PER_SHARD=1000,NODE_BIN=${NODE_BIN:-node} scripts/submit_erawan_generate_shard.sbatch
```

Build value/search datasets after the array job completes:

```bash
sbatch --dependency=afterok:<GEN_JOB_ID> --export=ALL,RUN_ID=phase6_erawan_loop_iter_001,NODE_BIN=${NODE_BIN:-node} scripts/submit_erawan_build_datasets.sbatch
```

Train iteration 1 with the tested PyTorch architecture:

```bash
sbatch --dependency=afterok:<DATA_JOB_ID> --export=ALL,TRAIN_RUN_ID=phase6_erawan_loop,DATASET_ID=phase6_erawan_loop_iter_001,ITERATION=1,VENV_DIR=$PWD/.venv_torch,VALUE_EPOCHS=5,POLICY_EPOCHS=5,VALUE_BATCH_SIZE=256,POLICY_BATCH_SIZE=64 scripts/submit_erawan_train_repo.sbatch
```

Run a smaller ERAWAN architecture smoke job:

```bash
sbatch --export=ALL,TRAIN_RUN_ID=phase6_erawan_smoke,ITERATION=1,VENV_DIR=$PWD/.venv_torch,VALUE_DATASET=$PWD/data/datasets/value/phase4_mixed_q.jsonl,POLICY_DATASET=$PWD/data/datasets/search/phase6_search_improved.jsonl,OUT_DIR=$PWD/experiments/torch_erawan_smoke,VALUE_EPOCHS=1,POLICY_EPOCHS=1,VALUE_BATCH_SIZE=32,POLICY_BATCH_SIZE=8,VOCAB_SIZE=4096,D_MODEL=32,N_HEADS=4,N_LAYERS=1,MAX_STATE_TOKENS=128,MAX_ACTION_TOKENS=32 scripts/submit_erawan_train_repo.sbatch
```

Expected iteration 1 outputs:

```text
logs/battles/phase6_erawan_loop_iter_001_train/shard_*/
data/datasets/value/phase6_erawan_loop_iter_001_value.jsonl
data/datasets/search/phase6_erawan_loop_iter_001_search.jsonl
models/torch/phase6_erawan_loop/iter_001/value/checkpoint.pt
models/torch/phase6_erawan_loop/iter_001/policy/checkpoint.pt
```

For iteration 2, generate a new dataset:

```bash
sbatch --array=1-10%2 --export=ALL,RUN_ID=phase6_erawan_loop_iter_002,GAMES_PER_SHARD=1000,NODE_BIN=${NODE_BIN:-node} scripts/submit_erawan_generate_shard.sbatch
```

Build iteration 2 datasets:

```bash
sbatch --dependency=afterok:<GEN_JOB_ID> --export=ALL,RUN_ID=phase6_erawan_loop_iter_002,NODE_BIN=${NODE_BIN:-node} scripts/submit_erawan_build_datasets.sbatch
```

Train iteration 2 incrementally. The script automatically resumes from `models/torch/phase6_erawan_loop/iter_001/{value,policy}/checkpoint.pt`:

```bash
sbatch --dependency=afterok:<DATA_JOB_ID> --export=ALL,TRAIN_RUN_ID=phase6_erawan_loop,DATASET_ID=phase6_erawan_loop_iter_002,ITERATION=2,VENV_DIR=$PWD/.venv_torch,VALUE_EPOCHS=5,POLICY_EPOCHS=5,VALUE_BATCH_SIZE=256,POLICY_BATCH_SIZE=64 scripts/submit_erawan_train_repo.sbatch
```

Check job status:

```bash
squeue -u $USER
```

## PC-To-ERAWAN Bundle Fallback

Use this only when games/datasets are generated on the PC but training happens on ERAWAN.

Prepare a bundle from PC datasets:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\prepare_gpu_training_bundle.js --run-id phase6_gpu_loop --iteration 1 --value-dataset data\datasets\value\phase6_gpu_loop_iter_001_value.jsonl --search-dataset data\datasets\search\phase6_gpu_loop_iter_001_search.jsonl --out-dir experiments\gpu_bundles\phase6_gpu_loop_iter_001 --overwrite
```

Submit uploaded bundle on ERAWAN:

```bash
sbatch --export=ALL,BUNDLE_DIR=$PWD,OUT_DIR=$PWD/outputs,DEVICE=cuda,VENV_DIR=$PWD/.venv_torch,VALUE_EPOCHS=5,POLICY_EPOCHS=5 scripts/submit_hpc_train.sbatch
```

Import downloaded GPU results back on the PC:

```powershell
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\import_gpu_training_results.js --run-id phase6_gpu_loop --iteration 1 --hpc-output-dir path\to\downloaded\outputs --overwrite
```

Expected local imported outputs:

```text
models/torch/phase6_gpu_loop/iter_001/value/checkpoint.pt
models/torch/phase6_gpu_loop/iter_001/value/metrics.json
models/torch/phase6_gpu_loop/iter_001/policy/checkpoint.pt
models/torch/phase6_gpu_loop/iter_001/policy/metrics.json
models/torch/phase6_gpu_loop/latest.json
```
