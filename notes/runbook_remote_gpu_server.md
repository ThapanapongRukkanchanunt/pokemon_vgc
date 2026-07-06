# Remote GPU Server Runbook

Last updated: 2026-06-12

Use this for a normal Linux GPU server reached with `scp` and `ssh`.

This path avoids conda. It uses a project-local Python venv named `.venv_torch`.

Replace `gpu-server` in the commands below with your SSH alias or `user@host`.

## Prepare Bundle On Local PC

On the Windows PC:

```powershell
cd C:\Users\thaip\Documents\pokemon_rl
vendor\node-v24.14.0\node-v24.14.0-win-x64\node.exe scripts\prepare_full_training_bundle.js --out-dir experiments\full_training_bundle --overwrite
```

Copy the bundle to the server:

```powershell
tar -czf .\experiments\full_training_bundle.tar.gz -C .\experiments\full_training_bundle .
scp .\experiments\full_training_bundle.tar.gz gpu-server:~/pokemon_rl_full.tar.gz
ssh gpu-server "mkdir -p ~/pokemon_rl_full && tar -xzf ~/pokemon_rl_full.tar.gz -C ~/pokemon_rl_full"
```

Open a shell on the server:

```powershell
ssh gpu-server
```

## Server Setup

On the server:

```bash
cd ~/pokemon_rl_full
```

Check Node:

```bash
node --version
```

If Node is not on PATH, put a Linux Node binary in your home directory and export `NODE_BIN`:

```bash
export NODE_BIN=$HOME/node-v24.14.0-linux-x64/bin/node
export PATH=$HOME/node-v24.14.0-linux-x64/bin:$PATH
node --version
npm --version
```

If the server cannot download Node itself, send a Linux x64 Node archive from the PC:

```powershell
scp C:\Users\thaip\Downloads\node-v24.14.0-linux-x64.tar.xz gpu-server:~/
ssh gpu-server "tar -xf ~/node-v24.14.0-linux-x64.tar.xz -C ~/"
ssh gpu-server 'echo "export NODE_BIN=$HOME/node-v24.14.0-linux-x64/bin/node" >> ~/.bashrc'
ssh gpu-server 'echo "export PATH=$HOME/node-v24.14.0-linux-x64/bin:$PATH" >> ~/.bashrc'
```

Install or repair Pokemon Showdown's runtime Node dependencies:

```bash
cd ~/pokemon_rl_full/vendor/pokemon-showdown
if [ -f package-lock.json ]; then npm ci --omit=dev --omit=optional; else npm install ts-chacha20@1.2.0 --no-save --package-lock=false --no-audit --no-fund; fi
node -e "console.log(require.resolve('ts-chacha20'))"
cd ~/pokemon_rl_full
```

Create the venv:

```bash
python3 -m venv .venv_torch
source .venv_torch/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy
python scripts/check_torch_gpu.py --require-cuda
```

The same setup can be run directly from the PC:

```powershell
ssh gpu-server "cd ~/pokemon_rl_full && python3 -m venv .venv_torch"
ssh gpu-server "cd ~/pokemon_rl_full && bash -lc 'cd vendor/pokemon-showdown && if [ -f package-lock.json ]; then npm ci --omit=dev --omit=optional; else npm install ts-chacha20@1.2.0 --no-save --package-lock=false --no-audit --no-fund; fi && cd ../.. && source .venv_torch/bin/activate && python -m pip install --upgrade pip && python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 && python -m pip install numpy && python scripts/check_torch_gpu.py --require-cuda'"
```

## Smoke Run

Run a tiny full loop:

```bash
cd ~/pokemon_rl_full
source .venv_torch/bin/activate
RUN_ID=torch_server_smoke ITERATIONS=1 GAMES=2 EVAL_GAMES=2 VALUE_EPOCHS=1 POLICY_EPOCHS=1 BOOTSTRAP_POLICY_EPOCHS=1 PYTHON_BIN=$PWD/.venv_torch/bin/python NODE_BIN=${NODE_BIN:-node} bash scripts/run_torch_full_loop.sh
```

Validate eval traces:

```bash
${NODE_BIN:-node} scripts/validate_battle_traces.js --log-dir logs/battles/torch_server_smoke_iter_001_post_train_eval
```

## Full Training Run

Run the full loop:

```bash
cd ~/pokemon_rl_full
source .venv_torch/bin/activate
RUN_ID=phase6_remote_torch ITERATIONS=5 GAMES=10000 EVAL_GAMES=100 VALUE_EPOCHS=5 POLICY_EPOCHS=5 PYTHON_BIN=$PWD/.venv_torch/bin/python NODE_BIN=${NODE_BIN:-node} bash scripts/run_torch_full_loop.sh
```

The loop performs:

```text
bootstrap PyTorch policy if needed
play train games with torch_policy
build value dataset
build search-improved policy dataset
train PyTorch value and policy checkpoints
run eval games with torch_policy
repeat
```

Expected outputs:

```text
models/torch/phase6_remote_torch/bootstrap/policy/checkpoint.pt
models/torch/phase6_remote_torch/iter_001/value/checkpoint.pt
models/torch/phase6_remote_torch/iter_001/policy/checkpoint.pt
experiments/large_scale/phase6_remote_torch_iter_001/
logs/battles/phase6_remote_torch_iter_001_post_train_eval/*.replay.html
logs/battles/phase6_remote_torch_iter_001_post_train_eval/*.summary.json
logs/battles/phase6_remote_torch_iter_001_post_train_eval/*.trace.jsonl
```

## Pull Results Back To PC

From Windows PowerShell:

```powershell
ssh gpu-server "cd ~/pokemon_rl_full && tar -czf ~/phase6_remote_torch_results.tar.gz models/torch/phase6_remote_torch experiments/large_scale/phase6_remote_torch_iter_001_post_train logs/battles/phase6_remote_torch_iter_001_post_train_eval"
scp gpu-server:~/phase6_remote_torch_results.tar.gz C:\Users\thaip\Documents\pokemon_rl\experiments\
tar -xzf C:\Users\thaip\Documents\pokemon_rl\experiments\phase6_remote_torch_results.tar.gz -C C:\Users\thaip\Documents\pokemon_rl
```

## Resume Later

Run the same full command with a larger `ITERATIONS` value. The loop resumes each new iteration from:

```text
models/torch/<RUN_ID>/iter_<previous>/value/checkpoint.pt
models/torch/<RUN_ID>/iter_<previous>/policy/checkpoint.pt
```

Use a new `RUN_ID` when you want a clean experiment lane.
