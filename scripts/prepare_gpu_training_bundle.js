const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.join(__dirname, '..');

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`${name} must be a positive integer`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    runId: 'phase6_gpu_loop',
    iteration: null,
    valueDataset: null,
    searchDataset: null,
    valueCheckpoint: null,
    policyCheckpoint: null,
    outDir: null,
    overwrite: false,
    dryRun: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--run-id') {
      args.runId = argv[++i];
    } else if (arg === '--iteration') {
      args.iteration = parseInteger(argv[++i], '--iteration');
    } else if (arg === '--value-dataset') {
      args.valueDataset = argv[++i];
    } else if (arg === '--search-dataset') {
      args.searchDataset = argv[++i];
    } else if (arg === '--value-checkpoint') {
      args.valueCheckpoint = argv[++i];
    } else if (arg === '--policy-checkpoint') {
      args.policyCheckpoint = argv[++i];
    } else if (arg === '--out-dir') {
      args.outDir = argv[++i];
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
node scripts/prepare_gpu_training_bundle.js [options]

Options:
  --run-id <id>                 Run identifier. Default: phase6_gpu_loop
  --iteration <n>               Iteration number used for inferred dataset paths.
  --value-dataset <path>        Value JSONL to copy into the bundle.
  --search-dataset <path>       Search/BC JSONL to copy into the bundle.
  --value-checkpoint <path>     Previous PyTorch value checkpoint for incremental training.
  --policy-checkpoint <path>    Previous PyTorch policy checkpoint for incremental training.
  --out-dir <path>              Bundle directory to create.
  --overwrite                   Replace files already present in the bundle.
  --dry-run                     Print planned copies without writing files.
`);
}

function resolveRepoPath(filePath) {
  if (!filePath) return null;
  return path.isAbsolute(filePath) ? path.normalize(filePath) : path.resolve(repoRoot, filePath);
}

function relativeDisplay(filePath) {
  const relative = path.relative(repoRoot, filePath);
  return relative && !relative.startsWith('..') && !path.isAbsolute(relative) ?
    relative.replace(/\\/g, '/') :
    filePath.replace(/\\/g, '/');
}

function iterationTag(iteration) {
  return `iter_${String(iteration).padStart(3, '0')}`;
}

function inferDatasetPath(args, kind) {
  const explicit = kind === 'value' ? args.valueDataset : args.searchDataset;
  if (explicit) return resolveRepoPath(explicit);

  const outputId = args.iteration == null ? args.runId : `${args.runId}_${iterationTag(args.iteration)}`;
  const suffix = kind === 'value' ? 'value' : 'search';
  return path.join(repoRoot, 'data', 'datasets', suffix, `${outputId}_${suffix}.jsonl`);
}

function ensureReadable(filePath, label, required = true) {
  if (!filePath) return null;
  if (!fs.existsSync(filePath)) {
    if (required) throw new Error(`${label} does not exist: ${filePath}`);
    return null;
  }
  const stat = fs.statSync(filePath);
  if (!stat.isFile()) throw new Error(`${label} is not a file: ${filePath}`);
  return filePath;
}

function copyFile(src, dest, args, label) {
  if (!src) return null;
  if (args.dryRun) {
    console.log(`[dry-run] copy ${label}: ${relativeDisplay(src)} -> ${relativeDisplay(dest)}`);
    return dest;
  }
  fs.mkdirSync(path.dirname(dest), {recursive: true});
  if (fs.existsSync(dest) && !args.overwrite) {
    throw new Error(`Refusing to overwrite ${dest}; pass --overwrite`);
  }
  fs.copyFileSync(src, dest);
  const bytes = fs.statSync(dest).size;
  console.log(`Copied ${label}: ${relativeDisplay(dest)} (${bytes} bytes)`);
  return dest;
}

function writeText(filePath, text, args, label) {
  if (args.dryRun) {
    console.log(`[dry-run] write ${label}: ${relativeDisplay(filePath)}`);
    return;
  }
  fs.mkdirSync(path.dirname(filePath), {recursive: true});
  if (fs.existsSync(filePath) && !args.overwrite) {
    throw new Error(`Refusing to overwrite ${filePath}; pass --overwrite`);
  }
  fs.writeFileSync(filePath, text, 'utf8');
  console.log(`Wrote ${label}: ${relativeDisplay(filePath)}`);
}

function writeJson(filePath, data, args, label) {
  writeText(filePath, `${JSON.stringify(data, null, 2)}\n`, args, label);
}

function commandText(hasValueCheckpoint, hasPolicyCheckpoint) {
  const valueInit = hasValueCheckpoint ? ' --init-checkpoint checkpoints/value_checkpoint.pt --resume-optimizer' : '';
  const policyInit = hasPolicyCheckpoint ? ' --init-checkpoint checkpoints/policy_checkpoint.pt --resume-optimizer' : '';
  return `# From the bundle root on a GPU-enabled system:
python3 scripts/check_torch_gpu.py --require-cuda

python3 scripts/train_value_alphastar_torch.py --dataset data/value_dataset.jsonl --out-dir outputs/value${valueInit} --device cuda --epochs 5 --batch-size 256 --overwrite

python3 scripts/train_policy_alphastar_torch.py --dataset data/search_dataset.jsonl --out-dir outputs/policy${policyInit} --device cuda --epochs 5 --batch-size 64 --overwrite

# Equivalent helper:
bash scripts/hpc_train_iteration.sh . outputs cuda

# Larger architecture override example:
D_MODEL=256 N_HEADS=8 N_LAYERS=4 MAX_STATE_TOKENS=512 MAX_ACTION_TOKENS=96 VALUE_BATCH_SIZE=256 POLICY_BATCH_SIZE=64 bash scripts/hpc_train_iteration.sh . outputs cuda

# CMU HPC ERAWAN SLURM template using a project-local venv:
python3 -m venv .venv_torch
source .venv_torch/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
sbatch --export=ALL,BUNDLE_DIR=$PWD,OUT_DIR=$PWD/outputs,DEVICE=cuda,VENV_DIR=$PWD/.venv_torch scripts/submit_hpc_train.sbatch
`;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const valueDataset = ensureReadable(inferDatasetPath(args, 'value'), 'value dataset');
  const searchDataset = ensureReadable(inferDatasetPath(args, 'search'), 'search dataset');
  const valueCheckpoint = ensureReadable(resolveRepoPath(args.valueCheckpoint), 'value checkpoint', false);
  const policyCheckpoint = ensureReadable(resolveRepoPath(args.policyCheckpoint), 'policy checkpoint', false);

  const tag = args.iteration == null ? 'manual' : iterationTag(args.iteration);
  const outDir = resolveRepoPath(args.outDir || path.join('experiments', 'gpu_bundles', `${args.runId}_${tag}`));

  if (!args.dryRun) fs.mkdirSync(outDir, {recursive: true});

  copyFile(valueDataset, path.join(outDir, 'data', 'value_dataset.jsonl'), args, 'value dataset');
  copyFile(searchDataset, path.join(outDir, 'data', 'search_dataset.jsonl'), args, 'search dataset');
  copyFile(valueCheckpoint, path.join(outDir, 'checkpoints', 'value_checkpoint.pt'), args, 'value checkpoint');
  copyFile(policyCheckpoint, path.join(outDir, 'checkpoints', 'policy_checkpoint.pt'), args, 'policy checkpoint');

  const scriptFiles = [
    'check_torch_gpu.py',
    'torch_alpha_model.py',
    'train_value_alphastar_torch.py',
    'train_policy_alphastar_torch.py',
    'hpc_train_iteration.sh',
    'submit_hpc_train.sbatch',
  ];
  for (const fileName of scriptFiles) {
    copyFile(
      ensureReadable(path.join(repoRoot, 'scripts', fileName), `script ${fileName}`),
      path.join(outDir, 'scripts', fileName),
      args,
      `script ${fileName}`,
    );
  }

  const manifest = {
    created_at: new Date().toISOString(),
    run_id: args.runId,
    iteration: args.iteration,
    bundle_dir: relativeDisplay(outDir),
    value_dataset: relativeDisplay(valueDataset),
    search_dataset: relativeDisplay(searchDataset),
    value_checkpoint: valueCheckpoint ? relativeDisplay(valueCheckpoint) : null,
    policy_checkpoint: policyCheckpoint ? relativeDisplay(policyCheckpoint) : null,
    incremental_training: {
      value: Boolean(valueCheckpoint),
      policy: Boolean(policyCheckpoint),
      note: 'If checkpoints are present, the GPU scripts load them and resume optimizer state. If absent, this is the first checkpoint for that model lane.',
    },
  };
  writeJson(path.join(outDir, 'manifest.json'), manifest, args, 'manifest');
  writeText(
    path.join(outDir, 'README_commands.txt'),
    commandText(Boolean(valueCheckpoint), Boolean(policyCheckpoint)),
    args,
    'command README',
  );

  console.log(`Bundle ready: ${relativeDisplay(outDir)}`);
}

try {
  main();
} catch (error) {
  console.error(`FAIL ${error.message}`);
  process.exitCode = 1;
}
