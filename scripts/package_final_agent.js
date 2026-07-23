const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {canonicalFormatId, validateAndPackTeam} = require('../src/battle/showdown_protocol');
const {findTeam, loadTeamPool} = require('../src/battle/run_battle');

const repoRoot = path.join(__dirname, '..');

function parseArgs(argv) {
  const args = {
    teamSelection: null,
    previewSelection: null,
    outDir: path.join(repoRoot, 'models', 'torch', 'final_mb_agent'),
    overwrite: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--team-selection') args.teamSelection = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--preview-selection') args.previewSelection = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--out-dir') args.outDir = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--overwrite') args.overwrite = true;
    else throw new Error(`Unexpected argument: ${arg}`);
  }
  if (!args.teamSelection || !args.previewSelection) {
    throw new Error('--team-selection and --preview-selection are required');
  }
  return args;
}

function assertInsideRepo(filePath) {
  const relative = path.relative(repoRoot, path.resolve(filePath));
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`Final package must stay inside the repository: ${filePath}`);
  }
}

function readJson(filePath) {
  if (!fs.existsSync(filePath)) throw new Error(`Missing JSON: ${filePath}`);
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function resolveSource(checkpoint) {
  if (!checkpoint) throw new Error('Selection JSON has no checkpoint');
  return path.isAbsolute(checkpoint) ? checkpoint : path.resolve(repoRoot, checkpoint);
}

function sha256(filePath) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const input = fs.createReadStream(filePath);
    input.on('error', reject);
    input.on('data', chunk => hash.update(chunk));
    input.on('end', () => resolve(hash.digest('hex')));
  });
}

async function run(args) {
  assertInsideRepo(args.outDir);
  const teamSelection = readJson(args.teamSelection);
  const previewSelection = readJson(args.previewSelection);
  const battleSource = resolveSource(teamSelection.checkpoint);
  const previewSource = resolveSource(previewSelection.checkpoint);
  for (const [label, filePath] of [['battle checkpoint', battleSource], ['preview checkpoint', previewSource]]) {
    if (!fs.existsSync(filePath)) throw new Error(`Missing ${label}: ${filePath}`);
  }
  if (fs.existsSync(args.outDir)) {
    if (!args.overwrite) throw new Error(`Package exists at ${args.outDir}; pass --overwrite`);
    fs.rmSync(args.outDir, {recursive: true, force: true});
  }
  fs.mkdirSync(args.outDir, {recursive: true});

  const pool = loadTeamPool();
  const team = findTeam(pool, teamSelection.team_id, {pick: items => items[0]});
  const importSource = path.resolve(repoRoot, team.import_file);
  validateAndPackTeam({formatId: pool.format_id, importText: fs.readFileSync(importSource, 'utf8')});
  const battleName = 'battle_checkpoint.pt';
  const previewName = 'preview_checkpoint.pt';
  const teamName = 'team.txt';
  fs.copyFileSync(battleSource, path.join(args.outDir, battleName));
  fs.copyFileSync(previewSource, path.join(args.outDir, previewName));
  fs.copyFileSync(importSource, path.join(args.outDir, teamName));

  const manifest = {
    created_at: new Date().toISOString(),
    package_version: 1,
    format_id: canonicalFormatId(pool.format_id),
    format_name: pool.format_name,
    regulation: pool.regulation,
    team_id: team.id,
    team_name: team.name,
    battle_checkpoint: battleName,
    preview_checkpoint: previewName,
    team_import: teamName,
    inference: {epsilon: 0, top_k: 1, sample_actions: false},
    source: {
      team_selection: path.relative(repoRoot, args.teamSelection).replace(/\\/g, '/'),
      preview_selection: path.relative(repoRoot, args.previewSelection).replace(/\\/g, '/'),
      battle_checkpoint: path.relative(repoRoot, battleSource).replace(/\\/g, '/'),
      preview_checkpoint: path.relative(repoRoot, previewSource).replace(/\\/g, '/'),
    },
    sha256: {
      battle_checkpoint: await sha256(path.join(args.outDir, battleName)),
      preview_checkpoint: await sha256(path.join(args.outDir, previewName)),
      team_import: await sha256(path.join(args.outDir, teamName)),
    },
  };
  fs.writeFileSync(path.join(args.outDir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  console.log(`Final agent package: ${path.relative(repoRoot, args.outDir)}`);
  console.log(`Team: ${team.id} (${team.name})`);
}

run(parseArgs(process.argv.slice(2))).catch(error => {
  console.error(`FAIL ${error.stack || error.message}`);
  process.exit(1);
});
