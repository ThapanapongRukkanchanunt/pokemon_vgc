const fs = require('node:fs');
const path = require('node:path');
const {closeTorchPolicyScorers, createAgent} = require('../src/agents');
const {findTeam, loadTeamPool} = require('../src/battle/run_battle');
const {dexForFormat, validateAndPackTeam} = require('../src/battle/showdown_protocol');
const {ShowdownLadderClient} = require('../src/showdown_ladder');

const repoRoot = path.join(__dirname, '..');

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    team: null,
    modelPath: null,
    modelManifest: null,
    teamPreviewModel: null,
    credentials: path.join(repoRoot, 'showdown.env'),
    username: process.env.SHOWDOWN_USERNAME || null,
    password: process.env.SHOWDOWN_PASSWORD || null,
    games: 1,
    pythonPath: process.env.POKEMON_RL_PYTHON || null,
    torchDevice: process.env.POKEMON_RL_TORCH_DEVICE || 'cpu',
    websocketUrl: 'wss://sim3.psim.us/showdown/websocket',
    loginUrl: 'https://play.pokemonshowdown.com/api/login',
    logDir: path.join(repoRoot, 'logs', 'ladder'),
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--team') args.team = argv[++i];
    else if (arg === '--model-path') args.modelPath = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--model-manifest') args.modelManifest = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--team-preview-model') args.teamPreviewModel = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--credentials') args.credentials = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--username') args.username = argv[++i];
    else if (arg === '--games') args.games = parseInteger(argv[++i], '--games');
    else if (arg === '--python') args.pythonPath = argv[++i];
    else if (arg === '--torch-device') args.torchDevice = argv[++i];
    else if (arg === '--websocket-url') args.websocketUrl = argv[++i];
    else if (arg === '--login-url') args.loginUrl = argv[++i];
    else if (arg === '--log-dir') args.logDir = path.resolve(repoRoot, argv[++i]);
    else throw new Error(`Unexpected argument: ${arg}`);
  }
  if (!args.team) throw new Error('--team is required');
  if (Boolean(args.modelPath) === Boolean(args.modelManifest)) {
    throw new Error('Pass exactly one of --model-path or --model-manifest');
  }
  if (!args.teamPreviewModel) throw new Error('--team-preview-model is required');
  if (args.games <= 0) throw new Error('--games must be > 0');
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
  const credentials = parseEnvFile(args.credentials);
  args.username = args.username || credentials.SHOWDOWN_USERNAME;
  args.password = args.password || credentials.SHOWDOWN_PASSWORD;
  if (!args.username || !args.password) {
    throw new Error('Set SHOWDOWN_USERNAME and SHOWDOWN_PASSWORD in showdown.env or the environment');
  }

  const pool = loadTeamPool();
  const team = findTeam(pool, args.team, {pick: items => items[0]});
  const modelPath = args.modelPath || checkpointFromManifest(args.modelManifest, team.id);
  for (const [label, filePath] of [['agent checkpoint', modelPath], ['preview checkpoint', args.teamPreviewModel]]) {
    if (!fs.existsSync(filePath)) throw new Error(`Missing ${label}: ${filePath}`);
  }
  const importText = fs.readFileSync(path.join(repoRoot, team.import_file), 'utf8');
  const packedTeam = validateAndPackTeam({formatId: pool.format_id, importText});
  const formatId = dexForFormat(pool.format_id).id;
  const runId = `ladder_${team.id}_${new Date().toISOString().replace(/[:.]/g, '-')}`;
  fs.mkdirSync(args.logDir, {recursive: true});
  const logPath = path.join(args.logDir, `${runId}.jsonl`);
  const summaryPath = path.join(args.logDir, `${runId}.summary.json`);

  const agent = createAgent('final_rl', {
    formatId: pool.format_id,
    modelPath,
    teamPreviewModelPath: args.teamPreviewModel,
    pythonPath: args.pythonPath,
    torchDevice: args.torchDevice,
    epsilon: 0,
    topK: 1,
    sampleActions: false,
  });
  const client = new ShowdownLadderClient({
    username: args.username,
    password: args.password,
    packedTeam,
    formatId,
    ownTeam: team,
    agent,
    maxBattles: args.games,
    websocketUrl: args.websocketUrl,
    loginUrl: args.loginUrl,
    logPath,
  });
  const result = await client.run();
  const summary = {
    created_at: new Date().toISOString(),
    run_id: runId,
    format_id: formatId,
    team_id: team.id,
    model_path: path.relative(repoRoot, modelPath).replace(/\\/g, '/'),
    team_preview_model: path.relative(repoRoot, args.teamPreviewModel).replace(/\\/g, '/'),
    log_path: path.relative(repoRoot, logPath).replace(/\\/g, '/'),
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
