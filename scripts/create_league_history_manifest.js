const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.join(__dirname, '..');

function parseArgs(argv) {
  const args = {out: null, snapshots: []};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--out') args.out = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--snapshot') args.snapshots.push(argv[++i]);
    else throw new Error(`Unexpected argument: ${arg}`);
  }
  if (!args.out) throw new Error('--out is required');
  if (!args.snapshots.length) throw new Error('Pass at least one --snapshot');
  return args;
}

function parseSnapshot(value) {
  const equals = value.indexOf('=');
  const comma = value.lastIndexOf(',');
  if (equals <= 0 || comma <= equals + 1 || comma === value.length - 1) {
    throw new Error(`Invalid --snapshot ${value}; expected ID=MODELS_DIR,PREVIEW_CHECKPOINT`);
  }
  return {
    id: value.slice(0, equals),
    models_dir: value.slice(equals + 1, comma),
    team_preview_model: value.slice(comma + 1),
  };
}

function run(args) {
  const snapshots = args.snapshots.map(parseSnapshot);
  const ids = new Set();
  for (const snapshot of snapshots) {
    if (ids.has(snapshot.id)) throw new Error(`Duplicate snapshot ID: ${snapshot.id}`);
    ids.add(snapshot.id);
    const modelsDir = path.resolve(repoRoot, snapshot.models_dir);
    const previewModel = path.resolve(repoRoot, snapshot.team_preview_model);
    if (!fs.existsSync(modelsDir)) throw new Error(`Missing models directory: ${modelsDir}`);
    if (!fs.existsSync(previewModel)) throw new Error(`Missing preview checkpoint: ${previewModel}`);
  }
  const manifest = {created_at: new Date().toISOString(), snapshots};
  fs.mkdirSync(path.dirname(args.out), {recursive: true});
  fs.writeFileSync(args.out, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  console.log(`Wrote history manifest with ${snapshots.length} snapshots: ${path.relative(repoRoot, args.out)}`);
}

try {
  run(parseArgs(process.argv.slice(2)));
} catch (error) {
  console.error(`FAIL ${error.stack || error.message}`);
  process.exit(1);
}
