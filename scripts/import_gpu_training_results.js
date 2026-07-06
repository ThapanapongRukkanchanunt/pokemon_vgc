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
    hpcOutputDir: null,
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
    } else if (arg === '--hpc-output-dir') {
      args.hpcOutputDir = argv[++i];
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

  if (!args.hpcOutputDir) throw new Error('--hpc-output-dir is required');
  return args;
}

function printUsage() {
  console.log(`Usage:
node scripts/import_gpu_training_results.js --hpc-output-dir <path> [options]

Options:
  --run-id <id>              Run identifier. Default: phase6_gpu_loop
  --iteration <n>            Iteration number for the imported checkpoint.
  --hpc-output-dir <path>    Directory containing value/ and policy/ outputs.
  --out-dir <path>           Local destination root. Default: models/torch/<run-id>/<iter>
  --overwrite                Replace existing imported files.
  --dry-run                  Print planned copies without writing files.
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
  return iteration == null ? 'manual' : `iter_${String(iteration).padStart(3, '0')}`;
}

function optionalFile(filePath) {
  return fs.existsSync(filePath) && fs.statSync(filePath).isFile() ? filePath : null;
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
  console.log(`Copied ${label}: ${relativeDisplay(dest)}`);
  return relativeDisplay(dest);
}

function writeJson(filePath, data, args, label) {
  if (args.dryRun) {
    console.log(`[dry-run] write ${label}: ${relativeDisplay(filePath)}`);
    return;
  }
  fs.mkdirSync(path.dirname(filePath), {recursive: true});
  if (fs.existsSync(filePath) && !args.overwrite) {
    throw new Error(`Refusing to overwrite ${filePath}; pass --overwrite`);
  }
  fs.writeFileSync(filePath, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
  console.log(`Wrote ${label}: ${relativeDisplay(filePath)}`);
}

function importKind(kind, hpcOutputDir, destRoot, args) {
  const sourceDir = path.join(hpcOutputDir, kind);
  const checkpoint = optionalFile(path.join(sourceDir, 'checkpoint.pt'));
  const metrics = optionalFile(path.join(sourceDir, 'metrics.json'));
  if (!checkpoint && !metrics) return null;

  const destDir = path.join(destRoot, kind);
  const imported = {
    source_dir: relativeDisplay(sourceDir),
    checkpoint: null,
    metrics: null,
  };
  imported.checkpoint = copyFile(checkpoint, path.join(destDir, 'checkpoint.pt'), args, `${kind} checkpoint`);
  imported.metrics = copyFile(metrics, path.join(destDir, 'metrics.json'), args, `${kind} metrics`);
  return imported;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const hpcOutputDir = resolveRepoPath(args.hpcOutputDir);
  if (!fs.existsSync(hpcOutputDir) || !fs.statSync(hpcOutputDir).isDirectory()) {
    throw new Error(`--hpc-output-dir is not a directory: ${hpcOutputDir}`);
  }

  const tag = iterationTag(args.iteration);
  const destRoot = resolveRepoPath(args.outDir || path.join('models', 'torch', args.runId, tag));
  const imported = {
    created_at: new Date().toISOString(),
    run_id: args.runId,
    iteration: args.iteration,
    source_output_dir: relativeDisplay(hpcOutputDir),
    destination: relativeDisplay(destRoot),
    value: importKind('value', hpcOutputDir, destRoot, args),
    policy: importKind('policy', hpcOutputDir, destRoot, args),
  };

  if (!imported.value && !imported.policy) {
    throw new Error(`No value/ or policy/ checkpoint outputs found under ${hpcOutputDir}`);
  }

  writeJson(path.join(destRoot, 'manifest.json'), imported, args, 'import manifest');
  writeJson(path.join(repoRoot, 'models', 'torch', args.runId, 'latest.json'), imported, {...args, overwrite: true}, 'latest pointer');
  console.log(`Imported GPU results: ${relativeDisplay(destRoot)}`);
}

try {
  main();
} catch (error) {
  console.error(`FAIL ${error.message}`);
  process.exitCode = 1;
}
