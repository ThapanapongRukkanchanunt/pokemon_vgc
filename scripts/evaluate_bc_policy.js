const fs = require('node:fs');
const path = require('node:path');
const {closeTorchPolicyScorers, createAgent} = require('../src/agents');
const {
  findLeadMode,
  findTeam,
  loadTeamPool,
  makeRng,
  runBattle,
} = require('../src/battle/run_battle');

const repoRoot = path.join(__dirname, '..');

const DEFAULT_MATCHUPS = {
  bc_vs_random: {p1: 'bc_policy', p2: 'random'},
  bc_vs_maxdamage: {p1: 'bc_policy', p2: 'maxdamage'},
  torch_vs_random: {p1: 'torch_policy', p2: 'random'},
  torch_vs_maxdamage: {p1: 'torch_policy', p2: 'maxdamage'},
  maxdamage_vs_random: {p1: 'maxdamage', p2: 'random'},
};

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    model: path.join(repoRoot, 'models', 'bc_policy', 'trace_test_maxdamage', 'model.json'),
    outDir: path.join(repoRoot, 'experiments', 'bc_policy', 'eval_trace_test_maxdamage'),
    logDir: path.join(repoRoot, 'logs', 'battles', 'bc_policy_eval'),
    games: 1,
    seed: 'phase2_eval',
    matchups: Object.keys(DEFAULT_MATCHUPS),
    pythonPath: process.env.POKEMON_RL_PYTHON || null,
    torchDevice: process.env.POKEMON_RL_TORCH_DEVICE || null,
    overwrite: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--model') {
      args.model = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--out-dir') {
      args.outDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--log-dir') {
      args.logDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--games') {
      args.games = parseInteger(argv[++i], '--games');
    } else if (arg === '--seed') {
      args.seed = argv[++i];
    } else if (arg === '--matchups') {
      args.matchups = argv[++i].split(',').map(value => value.trim()).filter(Boolean);
    } else if (arg === '--python') {
      args.pythonPath = argv[++i];
    } else if (arg === '--torch-device') {
      args.torchDevice = argv[++i];
    } else if (arg === '--overwrite') {
      args.overwrite = true;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (args.games <= 0) throw new Error('--games must be > 0');
  for (const matchup of args.matchups) {
    if (!DEFAULT_MATCHUPS[matchup]) {
      throw new Error(`Unknown matchup ${matchup}; choose from ${Object.keys(DEFAULT_MATCHUPS).join(', ')}`);
    }
  }
  return args;
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function winnerSide(winner) {
  if (typeof winner !== 'string') return null;
  const match = winner.match(/\b(P[12])$/i);
  return match ? match[1].toLowerCase() : null;
}

function createEvalAgent(name, args, formatId) {
  const options = {formatId};
  if (['bc', 'bc_policy', 'bc_policy_agent', 'torch_policy'].includes(name)) options.modelPath = args.model;
  if (name === 'torch_policy') {
    options.pythonPath = args.pythonPath;
    options.torchDevice = args.torchDevice;
  }
  return createAgent(name, options);
}

function countRecoveryRows(tracePath) {
  if (!fs.existsSync(tracePath)) return 0;
  return fs.readFileSync(tracePath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .reduce((count, line) => {
      try {
        return count + (JSON.parse(line).error_recovery ? 1 : 0);
      } catch (error) {
        return count;
      }
    }, 0);
}

function ensureOutput(args) {
  const summaryPath = path.join(args.outDir, 'summary.json');
  if (!args.overwrite && fs.existsSync(summaryPath)) {
    throw new Error(`Summary exists at ${relativePath(summaryPath)}; pass --overwrite to replace it`);
  }
  fs.mkdirSync(args.outDir, {recursive: true});
  fs.mkdirSync(args.logDir, {recursive: true});
  return summaryPath;
}

async function evaluate(args) {
  if (!fs.existsSync(args.model)) throw new Error(`Model not found: ${args.model}`);
  const summaryPath = ensureOutput(args);
  const pool = loadTeamPool();
  const summary = {
    created_at: new Date().toISOString(),
    model_path: relativePath(args.model),
    games_per_matchup: args.games,
    seed: args.seed,
    log_dir: relativePath(args.logDir),
    matchups: [],
  };

  for (const matchupId of args.matchups) {
    const matchup = DEFAULT_MATCHUPS[matchupId];
    const matchupSummary = {
      id: matchupId,
      p1_agent: matchup.p1,
      p2_agent: matchup.p2,
      games: [],
      wins: {
        p1: 0,
        p2: 0,
        unknown: 0,
      },
      recovery_rows: 0,
    };

    for (let game = 1; game <= args.games; game++) {
      const battleSeed = `${args.seed}:${matchupId}:${game}`;
      const rng = makeRng(battleSeed);
      const p1Team = findTeam(pool, null, rng);
      const p2Team = findTeam(pool, null, rng);
      const p1Lead = findLeadMode(p1Team, null, rng);
      const p2Lead = findLeadMode(p2Team, null, rng);
      const p1Agent = createEvalAgent(matchup.p1, args, pool.format_id);
      const p2Agent = createEvalAgent(matchup.p2, args, pool.format_id);

      const result = await runBattle({
        pool,
        seed: battleSeed,
        p1Team,
        p2Team,
        p1Lead,
        p2Lead,
        p1Agent,
        p2Agent,
        logDir: args.logDir,
        rng,
      });

      const side = winnerSide(result.winner) || 'unknown';
      matchupSummary.wins[side] += 1;
      const tracePath = path.resolve(repoRoot, result.trace_jsonl_path);
      const recoveryRows = countRecoveryRows(tracePath);
      matchupSummary.recovery_rows += recoveryRows;
      matchupSummary.games.push({
        game,
        seed: battleSeed,
        winner: result.winner,
        winner_side: side,
        turns: result.turns,
        p1_team: result.p1.team_id,
        p2_team: result.p2.team_id,
        p1_lead: result.p1.lead_id,
        p2_lead: result.p2.lead_id,
        recovery_rows: recoveryRows,
        summary_json_path: result.summary_json_path,
        trace_jsonl_path: result.trace_jsonl_path,
      });
      console.log(`${matchupId} game=${game} winner=${result.winner} turns=${result.turns} recovery=${recoveryRows}`);
    }

    summary.matchups.push(matchupSummary);
  }

  fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  console.log(`Wrote summary: ${relativePath(summaryPath)}`);
  return {summaryPath, summary};
}

function main() {
  evaluate(parseArgs(process.argv.slice(2)))
    .finally(() => closeTorchPolicyScorers())
    .catch(error => {
    console.error(`FAIL ${error.stack || error.message}`);
    process.exit(1);
  });
}

main();
