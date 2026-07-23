const fs = require('node:fs');
const path = require('node:path');
const {closeTorchPolicyScorers, createAgent} = require('../src/agents');
const {findTeam, loadTeamPool} = require('../src/battle/run_battle');
const {canonicalFormatId, validateAndPackTeam} = require('../src/battle/showdown_protocol');
const {loadFinalAgentPackage} = require('../src/final_agent_package');
const {ShowdownLadderClient, teamSummaryFromSets} = require('../src/showdown_ladder');

const repoRoot = path.join(__dirname, '..');

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    team: null,
    packageDir: null,
    modelPath: null,
    modelManifest: null,
    teamPreviewModel: null,
    credentials: path.join(repoRoot, 'showdown.env'),
    username: process.env.SHOWDOWN_USERNAME || null,
    password: process.env.SHOWDOWN_PASSWORD || null,
    games: 1,
    mode: 'ladder',
    megaPolicy: null,
    pythonPath: process.env.POKEMON_RL_PYTHON || null,
    torchDevice: process.env.POKEMON_RL_TORCH_DEVICE || 'cpu',
    websocketUrl: 'wss://sim3.psim.us/showdown/websocket',
    loginUrl: 'https://play.pokemonshowdown.com/api/login',
    logDir: path.join(repoRoot, 'logs', 'ladder'),
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--team') args.team = argv[++i];
    else if (arg === '--package') args.packageDir = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--model-path') args.modelPath = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--model-manifest') args.modelManifest = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--team-preview-model') args.teamPreviewModel = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--credentials') args.credentials = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--username') args.username = argv[++i];
    else if (arg === '--games') args.games = parseInteger(argv[++i], '--games');
    else if (arg === '--mode') args.mode = argv[++i];
    else if (arg === '--mega-policy') args.megaPolicy = argv[++i].toLowerCase().replace(/-/g, '_');
    else if (arg === '--python') args.pythonPath = argv[++i];
    else if (arg === '--torch-device') args.torchDevice = argv[++i];
    else if (arg === '--websocket-url') args.websocketUrl = argv[++i];
    else if (arg === '--login-url') args.loginUrl = argv[++i];
    else if (arg === '--log-dir') args.logDir = path.resolve(repoRoot, argv[++i]);
    else throw new Error(`Unexpected argument: ${arg}`);
  }
  if (args.packageDir && (args.team || args.modelPath || args.modelManifest || args.teamPreviewModel)) {
    throw new Error('--package cannot be combined with team/model/preview arguments');
  }
  if (!args.packageDir) {
    if (!args.team) throw new Error('--team is required');
    if (Boolean(args.modelPath) === Boolean(args.modelManifest)) {
      throw new Error('Pass exactly one of --model-path or --model-manifest');
    }
    if (!args.teamPreviewModel) throw new Error('--team-preview-model is required');
  }
  if (!['ladder', 'challenge'].includes(args.mode)) {
    throw new Error('--mode must be ladder or challenge');
  }
  if (args.games < 0 || (args.mode === 'ladder' && args.games === 0)) {
    throw new Error('--games must be > 0 in ladder mode or >= 0 in challenge mode');
  }
  if (args.megaPolicy && !['model', 'sole_usable'].includes(args.megaPolicy)) {
    throw new Error('--mega-policy must be model or sole_usable');
  }
  return args;
}

function parseEnvFile(filePath) {
  if (!fs.existsSync(filePath)) return {};
  const values = {};
  for (const rawLine of fs.readFileSync(filePath, 'utf8').split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const equals = line.indexOf('=');
    if (equals <= 0) continue;
    const key = line.slice(0, equals).trim();
    let value = line.slice(equals + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    values[key] = value;
  }
  return values;
}

function checkpointFromManifest(filePath, teamId) {
  const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  const entry = (parsed.models || parsed)[teamId];
  const checkpoint = typeof entry === 'string' ? entry : entry?.checkpoint;
  if (!checkpoint) throw new Error(`Model manifest has no checkpoint for ${teamId}`);
  return path.isAbsolute(checkpoint) ? checkpoint : path.resolve(repoRoot, checkpoint);
}

async function main(args) {
  let packageManifest = null;
  let packageData = null;
  let team = null;
  let importText = null;
  let formatId = null;
  if (args.packageDir) {
    packageData = loadFinalAgentPackage(args.packageDir);
    packageManifest = packageData.manifest;
    args.team = packageManifest.team_id;
    args.modelPath = packageData.battleModelPath;
    args.teamPreviewModel = packageData.previewModelPath;
    args.megaPolicy = args.megaPolicy || packageManifest.inference?.mega_policy || 'model';
    team = {
      id: packageManifest.team_id,
      name: packageManifest.team_name || packageManifest.team_id,
      team_summary: teamSummaryFromSets(
        packageData.sets,
        packageManifest.team_id,
        packageManifest.team_name || packageManifest.team_id
      ),
    };
    importText = packageData.teamImportText;
    formatId = canonicalFormatId(packageManifest.format_id);
  }
  const credentials = parseEnvFile(args.credentials);
  args.username = args.username || credentials.SHOWDOWN_USERNAME;
  args.password = args.password || credentials.SHOWDOWN_PASSWORD;
  if (!args.username || !args.password) {
    throw new Error('Set SHOWDOWN_USERNAME and SHOWDOWN_PASSWORD in showdown.env or the environment');
  }

  const pool = packageData ? null : loadTeamPool();
  args.megaPolicy = args.megaPolicy || 'model';
  team = team || findTeam(pool, args.team, {pick: items => items[0]});
  const modelPath = args.modelPath || checkpointFromManifest(args.modelManifest, team.id);
  for (const [label, filePath] of [['agent checkpoint', modelPath], ['preview checkpoint', args.teamPreviewModel]]) {
    if (!fs.existsSync(filePath)) throw new Error(`Missing ${label}: ${filePath}`);
  }
  importText = importText || fs.readFileSync(path.join(repoRoot, team.import_file), 'utf8');
  formatId = formatId || canonicalFormatId(pool.format_id);
  const packedTeam = validateAndPackTeam({formatId, importText});
  if (packageManifest?.format_id && packageManifest.format_id !== formatId) {
    throw new Error(`Package format ${packageManifest.format_id} does not match ${formatId}`);
  }
  const runId = `${args.mode}_${team.id}_${new Date().toISOString().replace(/[:.]/g, '-')}`;
  fs.mkdirSync(args.logDir, {recursive: true});
  const logPath = path.join(args.logDir, `${runId}.jsonl`);
  const summaryPath = path.join(args.logDir, `${runId}.summary.json`);
  const replayDir = path.join(args.logDir, runId, 'replays');

  const agentOptions = {
    formatId,
    modelPath,
    teamPreviewModelPath: args.teamPreviewModel,
    pythonPath: args.pythonPath,
    torchDevice: args.torchDevice,
    epsilon: 0,
    topK: 1,
    megaPolicy: args.megaPolicy,
    sampleActions: false,
  };
  const client = new ShowdownLadderClient({
    username: args.username,
    password: args.password,
    packedTeam,
    formatId,
    ownTeam: team,
    agentFactory: () => createAgent('final_rl', agentOptions),
    maxBattles: args.games,
    mode: args.mode,
    websocketUrl: args.websocketUrl,
    loginUrl: args.loginUrl,
    logPath,
    replayDir,
  });
  const stop = signal => client.stop(signal);
  process.once('SIGINT', stop);
  process.once('SIGTERM', stop);
  const result = await client.run();
  process.removeListener('SIGINT', stop);
  process.removeListener('SIGTERM', stop);
  const summary = {
    created_at: new Date().toISOString(),
    run_id: runId,
    mode: args.mode,
    mega_policy: args.megaPolicy,
    format_id: formatId,
    team_id: team.id,
    model_path: path.relative(repoRoot, modelPath).replace(/\\/g, '/'),
    team_preview_model: path.relative(repoRoot, args.teamPreviewModel).replace(/\\/g, '/'),
    package_manifest: packageData ? path.relative(repoRoot, packageData.manifestPath).replace(/\\/g, '/') : null,
    log_path: path.relative(repoRoot, logPath).replace(/\\/g, '/'),
    replay_dir: path.relative(repoRoot, replayDir).replace(/\\/g, '/'),
    ...result,
  };
  fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  console.log(`Ladder summary: ${path.relative(repoRoot, summaryPath)}`);
}

main(parseArgs(process.argv.slice(2)))
  .finally(() => closeTorchPolicyScorers())
  .catch(error => {
    console.error(`FAIL ${error.stack || error.message}`);
    process.exit(1);
  });
