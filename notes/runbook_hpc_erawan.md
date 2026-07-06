# HPC ERAWAN Runbook

Last updated: 2026-06-12

Use this for CMU HPC ERAWAN.

This path avoids conda. It uses:

- `scp` and `ssh` for transfer and login.
- A project-local Python venv named `.venv_torch`.
- A Linux Node binary through `NODE_BIN`.
- Slurm jobs for generation, dataset building, training, and eval.

## SSH Alias

On the Windows PC, create or edit:

```powershell
notepad $env:USERPROFILE\.ssh\config
```

Add:

```text
Host erawan
  HostName erawan.cmu.ac.th
  User your_cmu_account@cmu.ac.th
```

Test:

```powershell
ssh erawan
```

If you need the ERAWAN tunnel, keep this open in one terminal:

```powershell
ssh -N -L 2222:erawan.cmu.ac.th:22 user_from_email@tunnel.hpc.cmu.ac.th
```

Then use `localhost` port `2222` in another terminal:

```powershell
ssh -p 2222 your_erawan_user@localhost
```

## Copy Project Bundle

On the Windows PC:

```powershell
cd C:\Users\thaip\Documents\pokemon_rl
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\prepare_full_training_bundle.js --out-dir experiments\full_training_bundle --overwrite
tar -czf .\experiments\full_training_bundle.tar.gz -C .\experiments\full_training_bundle .
scp .\experiments\full_training_bundle.tar.gz erawan:~/pokemon_rl_full.tar.gz
ssh erawan "mkdir -p ~/pokemon_rl_full && tar -xzf ~/pokemon_rl_full.tar.gz -C ~/pokemon_rl_full"
```

Through the tunnel:

```powershell
scp -P 2222 -o User=your_erawan_user .\experiments\full_training_bundle.tar.gz localhost:~/pokemon_rl_full.tar.gz
ssh -p 2222 your_erawan_user@localhost "mkdir -p ~/pokemon_rl_full && tar -xzf ~/pokemon_rl_full.tar.gz -C ~/pokemon_rl_full"
```

## Install Node Binary If Needed

Check Node on ERAWAN:

```powershell
ssh erawan "node --version"
```

If Node is unavailable, send a Linux x64 Node archive from your PC:

```powershell
scp C:\Users\thaip\Downloads\node-v24.14.0-linux-x64.tar.xz erawan:~/
ssh erawan "tar -xf ~/node-v24.14.0-linux-x64.tar.xz -C ~/"
ssh erawan 'echo "export NODE_BIN=$HOME/node-v24.14.0-linux-x64/bin/node" >> ~/.bashrc'
ssh erawan 'echo "export PATH=$HOME/node-v24.14.0-linux-x64/bin:$PATH" >> ~/.bashrc'
```

Then verify:

```powershell
ssh erawan 'bash -lc "source ~/.bashrc && ${NODE_BIN:-node} --version"'
```

If ERAWAN has internet access from the login node, install or repair Pokemon Showdown's runtime Node dependencies:

```powershell
ssh erawan 'bash -lc "source ~/.bashrc && cd ~/pokemon_rl_full/vendor/pokemon-showdown && if [ -f package-lock.json ]; then npm ci --omit=dev --omit=optional; else npm install ts-chacha20@1.2.0 --no-save --package-lock=false --no-audit --no-fund; fi && test -d node_modules/ts-chacha20 && echo ts-chacha20_ok"'
```

## Create PyTorch Venv

Create the server venv over SSH:

```powershell
ssh erawan "cd ~/pokemon_rl_full && python3 -m venv .venv_torch"
ssh erawan "cd ~/pokemon_rl_full && bash -lc 'source .venv_torch/bin/activate && python -m pip install --upgrade pip && python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 && python -m pip install numpy'"
```

Verify CUDA on a GPU node with Slurm:

```bash
cd ~/pokemon_rl_full
sbatch --partition=gpu --gpus=1 --cpus-per-task=4 --time=00:20:00 --wrap="cd $PWD && source .venv_torch/bin/activate && python scripts/check_torch_gpu.py --require-cuda"
```

Check job status:

```bash
squeue -u $USER
```

## One-Job Full Loop

This is the simplest ERAWAN command. It holds the GPU allocation while CPU game generation is running, so use it for smoke tests or when simplicity matters more than scheduling efficiency.

```bash
cd ~/pokemon_rl_full
sbatch --partition=gpu --gpus=1 --cpus-per-task=4 --time=72:00:00 --wrap="cd $PWD && source .venv_torch/bin/activate && source ~/.bashrc && RUN_ID=phase6_erawan_torch ITERATIONS=5 GAMES=10000 EVAL_GAMES=100 VALUE_EPOCHS=5 POLICY_EPOCHS=5 PYTHON_BIN=$PWD/.venv_torch/bin/python NODE_BIN=${NODE_BIN:-node} bash scripts/run_torch_full_loop.sh"
```

Smoke version:

```bash
cd ~/pokemon_rl_full
sbatch --partition=gpu --gpus=1 --cpus-per-task=4 --time=02:00:00 --wrap="cd $PWD && source .venv_torch/bin/activate && source ~/.bashrc && RUN_ID=phase6_erawan_smoke ITERATIONS=1 GAMES=2 EVAL_GAMES=2 VALUE_EPOCHS=1 POLICY_EPOCHS=1 BOOTSTRAP_POLICY_EPOCHS=1 PYTHON_BIN=$PWD/.venv_torch/bin/python NODE_BIN=${NODE_BIN:-node} bash scripts/run_torch_full_loop.sh"
```

## Scheduled ERAWAN-Native Loop

This avoids holding a GPU while CPU game generation runs.

Run a Node smoke check on a CPU job:

```bash
cd ~/pokemon_rl_full
sbatch --export=ALL,NODE_BIN=${NODE_BIN:-node} scripts/submit_erawan_node_check.sbatch
```

Generate 10,000 games as ten CPU-array shards, two shards running at a time:

```bash
cd ~/pokemon_rl_full
sbatch --array=1-10%2 --export=ALL,RUN_ID=phase6_erawan_loop_iter_001,GAMES_PER_SHARD=1000,NODE_BIN=${NODE_BIN:-node} scripts/submit_erawan_generate_shard.sbatch
```

Build value/search datasets after generation finishes:

```bash
cd ~/pokemon_rl_full
sbatch --dependency=afterok:<GEN_JOB_ID> --export=ALL,RUN_ID=phase6_erawan_loop_iter_001,NODE_BIN=${NODE_BIN:-node} scripts/submit_erawan_build_datasets.sbatch
```

Train iteration 1 on GPU:

```bash
cd ~/pokemon_rl_full
sbatch --dependency=afterok:<DATA_JOB_ID> --export=ALL,TRAIN_RUN_ID=phase6_erawan_loop,DATASET_ID=phase6_erawan_loop_iter_001,ITERATION=1,VENV_DIR=$PWD/.venv_torch,VALUE_EPOCHS=5,POLICY_EPOCHS=5,VALUE_BATCH_SIZE=256,POLICY_BATCH_SIZE=64 scripts/submit_erawan_train_repo.sbatch
```

Run a small training smoke job:

```bash
cd ~/pokemon_rl_full
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

For iteration 2, repeat generation and dataset building with `RUN_ID=phase6_erawan_loop_iter_002`, then train with `ITERATION=2`:

```bash
cd ~/pokemon_rl_full
sbatch --array=1-10%2 --export=ALL,RUN_ID=phase6_erawan_loop_iter_002,GAMES_PER_SHARD=1000,NODE_BIN=${NODE_BIN:-node} scripts/submit_erawan_generate_shard.sbatch
```

```bash
cd ~/pokemon_rl_full
sbatch --dependency=afterok:<GEN_JOB_ID> --export=ALL,RUN_ID=phase6_erawan_loop_iter_002,NODE_BIN=${NODE_BIN:-node} scripts/submit_erawan_build_datasets.sbatch
```

```bash
cd ~/pokemon_rl_full
sbatch --dependency=afterok:<DATA_JOB_ID> --export=ALL,TRAIN_RUN_ID=phase6_erawan_loop,DATASET_ID=phase6_erawan_loop_iter_002,ITERATION=2,VENV_DIR=$PWD/.venv_torch,VALUE_EPOCHS=5,POLICY_EPOCHS=5,VALUE_BATCH_SIZE=256,POLICY_BATCH_SIZE=64 scripts/submit_erawan_train_repo.sbatch
```

Iteration 2 resumes from:

```text
models/torch/phase6_erawan_loop/iter_001/value/checkpoint.pt
models/torch/phase6_erawan_loop/iter_001/policy/checkpoint.pt
```

## Eval With Replays

Evaluate a trained PyTorch policy checkpoint on a CPU job:

```bash
cd ~/pokemon_rl_full
sbatch --partition=cpu --cpus-per-task=4 --time=24:00:00 --wrap="cd $PWD && source .venv_torch/bin/activate && source ~/.bashrc && ${NODE_BIN:-node} scripts/evaluate_selectors.js --model models/torch/phase6_erawan_loop/iter_001/policy/checkpoint.pt --out-dir experiments/selectors/phase6_erawan_iter_001_eval --log-dir logs/battles/phase6_erawan_iter_001_eval --games 100 --seed phase6_erawan_iter_001_eval --matchups torch_policy_vs_random,torch_policy_vs_maxdamage,torch_policy_vs_heuristic --python $PWD/.venv_torch/bin/python --torch-device cpu --overwrite"
```

Validate traces:

```bash
cd ~/pokemon_rl_full
${NODE_BIN:-node} scripts/validate_battle_traces.js --log-dir logs/battles/phase6_erawan_iter_001_eval
```

Eval outputs:

```text
experiments/selectors/phase6_erawan_iter_001_eval/summary.json
logs/battles/phase6_erawan_iter_001_eval/*.replay.html
logs/battles/phase6_erawan_iter_001_eval/*.summary.json
logs/battles/phase6_erawan_iter_001_eval/*.trace.jsonl
```

## Pull Results Back To PC

From Windows PowerShell:

```powershell
ssh erawan "cd ~/pokemon_rl_full && tar -czf ~/phase6_erawan_results.tar.gz models/torch/phase6_erawan_loop experiments/selectors/phase6_erawan_iter_001_eval logs/battles/phase6_erawan_iter_001_eval"
scp erawan:~/phase6_erawan_results.tar.gz C:\Users\thaip\Documents\pokemon_rl\experiments\
tar -xzf C:\Users\thaip\Documents\pokemon_rl\experiments\phase6_erawan_results.tar.gz -C C:\Users\thaip\Documents\pokemon_rl
```

Through the tunnel:

```powershell
ssh -p 2222 your_erawan_user@localhost "cd ~/pokemon_rl_full && tar -czf ~/phase6_erawan_results.tar.gz models/torch/phase6_erawan_loop experiments/selectors/phase6_erawan_iter_001_eval logs/battles/phase6_erawan_iter_001_eval"
scp -P 2222 -o User=your_erawan_user localhost:~/phase6_erawan_results.tar.gz C:\Users\thaip\Documents\pokemon_rl\experiments\
tar -xzf C:\Users\thaip\Documents\pokemon_rl\experiments\phase6_erawan_results.tar.gz -C C:\Users\thaip\Documents\pokemon_rl
```
