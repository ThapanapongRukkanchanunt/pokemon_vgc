const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {closeTorchPolicyScorers, createAgent} = require('../src/agents');
const {
  findLeadMode,
  findTeam,
  loadTeamPool,
  makeRng,
  runBattle,
} = require('../src/battle/run_battle');
const {canonicalFormatId, validateAndPackTeam} = require('../src/battle/showdown_protocol');
const {loadFinalAgentPackage} = require('../src/final_agent_package');

const repoRoot = path.join(__dirname, '..');

function parseArgs(argv) {
  const args = {
    packageDir: path.join(repoRoot, 'models', 'torch', 'final_mb_agent'),
    opponentTeam: null,
    pythonPath: process.env.POKEMON_RL_PYTHON || null,
    torchDevice: process.env.POKEMON_RL_TORCH_DEVICE || 'cpu',
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--package') args.packageDir = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--opponent-team') args.opponentTeam = argv[++i];
    else if (arg === '--python') args.pythonPath = argv[++i];
    else if (arg === '--torch-device') args.torchDevice = argv[++i];
    else throw new Error(`Unexpected argument: ${arg}`);
  }
  return args;
}

async function run(args) {
  const packageData = loadFinalAgentPackage(args.packageDir);
  const pool = loadTeamPool();
  const poolFormatId = canonicalFormatId(pool.format_id);
  if (canonicalFormatId(packageData.manifest.format_id) !== poolFormatId) {
    throw new Error(`Package format ${packageData.manifest.format_id} does not match ${poolFormatId}`);
  }
  validateAndPackTeam({formatId: pool.format_id, importText: packageData.teamImportText});

  const sourceTeam = findTeam(pool, packageData.manifest.team_id, {pick: items => items[0]});
  const packageTeam = {
    ...sourceTeam,
    name: packageData.manifest.team_name || sourceTeam.name,
    import_file: path.relative(repoRoot, packageData.teamImportPath),
  };
  const opponentTeam = args.opponentTeam ?
    findTeam(pool, args.opponentTeam, {pick: items => items[0]}) :
    pool.teams.find(team => team.id !== packageTeam.id);
  if (!opponentTeam) throw new Error('No opponent team is available for the package smoke battle');

  const rng = makeRng(`final-package-smoke:${packageTeam.id}:${opponentTeam.id}`);
  const logDir = fs.mkdtempSync(path.join(os.tmpdir(), 'pokemon-vgc-final-smoke-'));
  try {
    const result = await runBattle({
      pool,
      seed: `final-package-smoke:${packageTeam.id}:${opponentTeam.id}`,
      p1Team: packageTeam,
      p2Team: opponentTeam,
      p1Lead: findLeadMode(packageTeam, null, rng),
      p2Lead: findLeadMode(opponentTeam, null, rng),
      p1Agent: createAgent('final_rl', {
        formatId: pool.format_id,
        modelPath: packageData.battleModelPath,
        teamPreviewModelPath: packageData.previewModelPath,
        pythonPath: args.pythonPath,
        torchDevice: args.torchDevice,
        epsilon: 0,
        topK: 1,
        megaPolicy: packageData.manifest.inference?.mega_policy || 'model',
        sampleActions: false,
      }),
      p2Agent: createAgent('heuristic', {formatId: pool.format_id}),
      logDir,
      rng,
    });
    if (!result.winner || !(result.turns > 0)) throw new Error('Smoke battle did not complete normally');
    console.log(JSON.stringify({
      status: 'pass',
      format_id: pool.format_id,
      team_id: packageTeam.id,
      opponent_team: opponentTeam.id,
      winner: result.winner,
      turns: result.turns,
      package_manifest: path.relative(repoRoot, packageData.manifestPath).replace(/\\/g, '/'),
    }, null, 2));
  } finally {
    await closeTorchPolicyScorers();
    fs.rmSync(logDir, {recursive: true, force: true});
  }
}

run(parseArgs(process.argv.slice(2))).catch(error => {
  console.error(`FAIL ${error.stack || error.message}`);
  process.exit(1);
});
