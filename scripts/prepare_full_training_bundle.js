const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.join(__dirname, '..');

function parseArgs(argv) {
  const args = {
    outDir: path.join(repoRoot, 'experiments', 'full_training_bundle'),
    includeWindowsNode: false,
    overwrite: false,
    dryRun: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--out-dir') {
      args.outDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--include-windows-node') {
      args.includeWindowsNode = true;
    } else if (arg === '--overwrite') {
      args.overwrite = true;
    } else if (arg === '--dry-run') {
      args.dryRun = true;
    } else if (arg === '--help' || arg === '-h') {
      printUsage();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  return args;
}

function printUsage() {
  console.log(`Usage:
node scripts/prepare_full_training_bundle.js [options]

Options:
  --out-dir <path>          Bundle directory to create. Default: experiments/full_training_bundle
  --include-windows-node    Also copy the vendored Windows Node runtime.
  --overwrite               Replace files already present in the bundle.
  --dry-run                 Print planned copies without writing files.
`);
}

function relativeDisplay(filePath) {
  const relative = path.relative(repoRoot, filePath);
  return relative && !relative.startsWith('..') && !path.isAbsolute(relative) ?
    relative.replace(/\\/g, '/') :
    filePath.replace(/\\/g, '/');
}

function ensureInsideRepo(filePath, label) {
  const resolved = path.resolve(filePath);
  const relative = path.relative(repoRoot, resolved);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`${label} must stay inside the repo: ${resolved}`);
  }
  return resolved;
}

function shouldSkip(relativePath) {
  const normalized = relativePath.replace(/\\/g, '/');
  const parts = normalized.split('/');
  if (parts.some(part => ['.git', 'logs', '.cache', '__pycache__'].includes(part))) return true;
  if (normalized.endsWith('.pyc')) return true;
  if (normalized.startsWith('vendor/pokemon-showdown/test/')) return true;
  if (normalized.startsWith('vendor/pokemon-showdown/.github/')) return true;
  if (normalized.startsWith('vendor/npm-cache/')) return true;
  return false;
}

function copyFile(src, dest, args, label) {
  if (args.dryRun) {
    console.log(`[dry-run] copy ${label}: ${relativeDisplay(src)} -> ${relativeDisplay(dest)}`);
    return;
  }
  fs.mkdirSync(path.dirname(dest), {recursive: true});
  if (fs.existsSync(dest) && !args.overwrite) {
    throw new Error(`Refusing to overwrite ${relativeDisplay(dest)}; pass --overwrite`);
  }
  fs.copyFileSync(src, dest);
}

function copyRecursive(src, dest, args, label, sourceRoot = src) {
  if (!fs.existsSync(src)) throw new Error(`${label} does not exist: ${src}`);
  const stat = fs.statSync(src);
  const relativeSource = path.relative(repoRoot, src);
  if (shouldSkip(relativeSource)) return;
  if (stat.isFile()) {
    copyFile(src, dest, args, label);
    return;
  }
  if (!stat.isDirectory()) return;
  if (!args.dryRun) fs.mkdirSync(dest, {recursive: true});
  for (const entry of fs.readdirSync(src, {withFileTypes: true})) {
    const childSrc = path.join(src, entry.name);
    const childDest = path.join(dest, entry.name);
    const childRelative = path.relative(repoRoot, childSrc);
    if (shouldSkip(childRelative)) continue;
    copyRecursive(childSrc, childDest, args, `${label}/${path.relative(sourceRoot, childSrc)}`, sourceRoot);
  }
}

function writeText(filePath, text, args, label) {
  if (args.dryRun) {
    console.log(`[dry-run] write ${label}: ${relativeDisplay(filePath)}`);
    return;
  }
  fs.mkdirSync(path.dirname(filePath), {recursive: true});
  if (fs.existsSync(filePath) && !args.overwrite) {
    throw new Error(`Refusing to overwrite ${relativeDisplay(filePath)}; pass --overwrite`);
  }
  fs.writeFileSync(filePath, text, 'utf8');
}

function bundleReadme() {
  return `# Pokemon RL Full PyTorch Training Bundle

This bundle is for running:

\`\`\`text
play games -> build datasets -> train PyTorch value/policy -> eval with replay -> repeat
\`\`\`

The live battle agent uses PyTorch policy checkpoints through \`torch_policy\`.
PyTorch value checkpoints are trained incrementally, but live search-label building still uses the included Node JSON value model.

## Detailed Runbooks

\`\`\`text
notes/runbook_local_pc.md
notes/runbook_remote_gpu_server.md
notes/runbook_hpc_erawan.md
\`\`\`

## Setup With Venv

Install Node.js 20+ or 24+ and Python with PyTorch CUDA.

On the server:

\`\`\`bash
python3 -m venv .venv_torch
source .venv_torch/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy
python scripts/check_torch_gpu.py --require-cuda
node --version
\`\`\`

If Node is not on PATH, put a Linux Node binary somewhere under your home directory and export \`NODE_BIN\`:

\`\`\`bash
export NODE_BIN=$HOME/node-v24.14.0-linux-x64/bin/node
export PATH=$HOME/node-v24.14.0-linux-x64/bin:$PATH
node --version
npm --version
\`\`\`

If the server cannot download Node itself, download the Linux x64 Node archive on your PC and send it with \`scp\`:

\`\`\`powershell
scp C:\\Users\\thaip\\Downloads\\node-v24.14.0-linux-x64.tar.xz erawan:~/
ssh erawan "tar -xf ~/node-v24.14.0-linux-x64.tar.xz -C ~/"
ssh erawan 'echo "export NODE_BIN=$HOME/node-v24.14.0-linux-x64/bin/node" >> ~/.bashrc'
ssh erawan 'echo "export PATH=$HOME/node-v24.14.0-linux-x64/bin:$PATH" >> ~/.bashrc'
\`\`\`

Install or repair Pokemon Showdown's runtime Node dependencies:

\`\`\`bash
cd vendor/pokemon-showdown
if [ -f package-lock.json ]; then npm ci --omit=dev --omit=optional; else npm install ts-chacha20@1.2.0 --no-save --package-lock=false --no-audit --no-fund; fi
node -e "console.log(require.resolve('ts-chacha20'))"
cd ../..
\`\`\`

## Run A Small Smoke Loop

\`\`\`bash
source .venv_torch/bin/activate
RUN_ID=torch_bundle_smoke ITERATIONS=1 GAMES=2 EVAL_GAMES=2 VALUE_EPOCHS=1 POLICY_EPOCHS=1 BOOTSTRAP_POLICY_EPOCHS=1 PYTHON_BIN=$PWD/.venv_torch/bin/python NODE_BIN=\${NODE_BIN:-node} bash scripts/run_torch_full_loop.sh
\`\`\`

## Run A Full Loop

\`\`\`bash
source .venv_torch/bin/activate
RUN_ID=phase6_torch_full ITERATIONS=5 GAMES=10000 EVAL_GAMES=100 VALUE_EPOCHS=5 POLICY_EPOCHS=5 PYTHON_BIN=$PWD/.venv_torch/bin/python NODE_BIN=\${NODE_BIN:-node} bash scripts/run_torch_full_loop.sh
\`\`\`

Useful overrides:

\`\`\`bash
TRAIN_DEVICE=cuda
TORCH_INFERENCE_DEVICE=cpu
VALUE_BATCH_SIZE=256
POLICY_BATCH_SIZE=64
TRAIN_PARTICIPANTS=torch_current_policy,random,maxdamage,heuristic
EVAL_PARTICIPANTS=torch_current_policy,random,maxdamage,heuristic
\`\`\`

Outputs:

\`\`\`text
models/torch/<RUN_ID>/bootstrap/policy/checkpoint.pt
models/torch/<RUN_ID>/iter_001/value/checkpoint.pt
models/torch/<RUN_ID>/iter_001/policy/checkpoint.pt
experiments/large_scale/<RUN_ID>_iter_001/
logs/battles/<RUN_ID>_iter_001_post_train_eval/*.replay.html
logs/battles/<RUN_ID>_iter_001_post_train_eval/*.summary.json
logs/battles/<RUN_ID>_iter_001_post_train_eval/*.trace.jsonl
\`\`\`

## ERAWAN Submit Example

\`\`\`bash
sbatch --partition=gpu --gpus=1 --cpus-per-task=4 --time=72:00:00 --wrap="cd $PWD && source .venv_torch/bin/activate && RUN_ID=phase6_erawan_torch ITERATIONS=5 GAMES=10000 EVAL_GAMES=100 VALUE_EPOCHS=5 POLICY_EPOCHS=5 PYTHON_BIN=$PWD/.venv_torch/bin/python NODE_BIN=\${NODE_BIN:-node} bash scripts/run_torch_full_loop.sh"
\`\`\`

For better ERAWAN scheduling, use the separate shard/build/train Slurm templates in \`scripts/submit_erawan_*.sbatch\`.
The single full-loop command is simplest, but it holds the GPU allocation while CPU game generation is running.

## Copy To ERAWAN

From Windows PowerShell:

\`\`\`powershell
cd C:\\Users\\thaip\\Documents\\pokemon_rl
tar -czf .\\experiments\\full_training_bundle.tar.gz -C .\\experiments\\full_training_bundle .
scp .\\experiments\\full_training_bundle.tar.gz erawan:~/pokemon_rl_full.tar.gz
ssh erawan "mkdir -p ~/pokemon_rl_full && tar -xzf ~/pokemon_rl_full.tar.gz -C ~/pokemon_rl_full"
ssh erawan
cd ~/pokemon_rl_full
\`\`\`

Create the venv over SSH:

\`\`\`powershell
ssh erawan "cd ~/pokemon_rl_full && python3 -m venv .venv_torch"
ssh erawan "cd ~/pokemon_rl_full && bash -lc 'source .venv_torch/bin/activate && python -m pip install --upgrade pip && python -m pip install torch --index-url https://download.pytorch.org/whl/cu121 && python -m pip install numpy'"
\`\`\`

With the ERAWAN tunnel:

\`\`\`powershell
ssh -N -L 2222:erawan.cmu.ac.th:22 user_from_email@tunnel.hpc.cmu.ac.th
scp -P 2222 -o User=your_erawan_user .\\experiments\\full_training_bundle.tar.gz localhost:~/pokemon_rl_full.tar.gz
ssh -p 2222 your_erawan_user@localhost "mkdir -p ~/pokemon_rl_full && tar -xzf ~/pokemon_rl_full.tar.gz -C ~/pokemon_rl_full"
\`\`\`

## Validate Eval Traces

\`\`\`bash
node scripts/validate_battle_traces.js --log-dir logs/battles/phase6_torch_full_iter_001_post_train_eval
\`\`\`
`;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = ensureInsideRepo(args.outDir, '--out-dir');

  const entries = [
    'AGENTS.md',
    'data/teams',
    'data/datasets/search/phase6_search_improved.jsonl',
    'data/datasets/search/phase6_search_improved.summary.json',
    'data/datasets/value/phase4_mixed_q.jsonl',
    'data/datasets/value/phase4_mixed_q.summary.json',
    'models/bc_policy/phase6_search_improved',
    'models/value_model/phase4_mixed_q',
    'notes/phase6_1_gpu_training.md',
    'notes/runbook_local_pc.md',
    'notes/runbook_remote_gpu_server.md',
    'notes/runbook_hpc_erawan.md',
    'scripts',
    'src',
    'vendor/pokemon-showdown',
  ];
  if (args.includeWindowsNode) entries.push('vendor/node-v24.14.0');

  if (!args.dryRun) fs.mkdirSync(outDir, {recursive: true});
  for (const entry of entries) {
    const src = path.join(repoRoot, entry);
    if (!fs.existsSync(src)) {
      console.log(`Skipping missing optional entry: ${entry}`);
      continue;
    }
    copyRecursive(src, path.join(outDir, entry), args, entry);
    console.log(`${args.dryRun ? '[dry-run] planned' : 'Copied'} ${entry}`);
  }

  const manifest = {
    created_at: new Date().toISOString(),
    bundle_type: 'full_torch_training_loop',
    live_policy_agent: 'torch_policy',
    training_loop_script: 'scripts/run_torch_full_loop.sh',
    includes_windows_node: args.includeWindowsNode,
    entries,
  };
  writeText(path.join(outDir, 'FULL_TRAINING_RUNBOOK.md'), bundleReadme(), args, 'runbook');
  writeText(path.join(outDir, 'bundle_manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`, args, 'manifest');
  console.log(`Full training bundle ready: ${relativeDisplay(outDir)}`);
}

try {
  main();
} catch (error) {
  console.error(`FAIL ${error.stack || error.message}`);
  process.exitCode = 1;
}
