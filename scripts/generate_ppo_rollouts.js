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

const OPPONENTS = {
  random: {agent: 'random'},
  maxdamage: {agent: 'maxdamage'},
  heuristic: {agent: 'heuristic_selector'},
  search_balanced: {agent: 'search_balanced', policy: true, value: true},
  hmm_belief: {agent: 'hmm_belief', policy: true, value: true},
};

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseList(value) {
  return String(value || '').split(',').map(item => item.trim()).filter(Boolean);
}

function parseArgs(argv) {
  const args = {
    runId: 'phase8_ppo_rollouts',
    model: path.join(repoRoot, 'experiments', 'torch_smoke', 'policy', 'checkpoint.pt'),
    teacherModel: path.join(repoRoot, 'models', 'bc_policy', 'phase6_search_improved', 'model.json'),
    valueModel: path.join(repoRoot, 'models', 'value_model', 'phase4_mixed_q', 'model.json'),
    outDir: path.join(repoRoot, 'data', 'datasets', 'rl'),
    logDir: null,
    games: 2,
    seed: null,
    opponents: ['random', 'maxdamage', 'heuristic'],
    sideSwaps: true,
    pythonPath: process.env.POKEMON_RL_PYTHON || null,
    torchDevice: process.env.POKEMON_RL_TORCH_DEVICE || null,
    overwrite: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--run-id') {
      args.runId = argv[++i];
    } else if (arg === '--model') {
      args.model = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--teacher-model') {
      args.teacherModel = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--value-model') {
      args.valueModel = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--out-dir') {
      args.outDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--log-dir') {
      args.logDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--games') {
      args.games = parseInteger(argv[++i], '--games');
    } else if (arg === '--seed') {
      args.seed = argv[++i];
    } else if (arg === '--opponents') {
      args.opponents = parseList(argv[++i]);
    } else if (arg === '--python') {
      args.pythonPath = argv[++i];
    } else if (arg === '--torch-device') {
      args.torchDevice = argv[++i];
    } else if (arg === '--no-side-swaps') {
      args.sideSwaps = false;
    } else if (arg === '--overwrite') {
      args.overwrite = true;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (!args.seed) args.seed = args.runId;
  if (!args.runId || /[\\/:*?"<>|]/.test(args.runId)) {
    throw new Error('--run-id must be filename-safe');
  }
  if (args.games <= 0) throw new Error('--games must be > 0');
  for (const opponent of args.opponents) {
    if (!OPPONENTS[opponent]) {
      throw new Error(`Unknown opponent ${opponent}; choose from ${Object.keys(OPPONENTS).join(', ')}`);
    }
  }
  return args;
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function winnerSide(winner) {
  if (typeof winner !== 'string') return 'unknown';
  const match = winner.match(/\b(P[12])$/i);
  return match ? match[1].toLowerCase() : 'unknown';
}

function createOpponentAgent(id, args, formatId) {
  const opponent = OPPONENTS[id];
  const options = {formatId};
  if (opponent.policy) options.modelPath = args.teacherModel;
  if (opponent.value) options.valueModelPath = args.valueModel;
  return createAgent(opponent.agent, options);
}

function createRlAgent(args, formatId) {
  return createAgent('final_rl', {
    formatId,
    modelPath: args.model,
    pythonPath: args.pythonPath,
    torchDevice: args.torchDevice,
    sampleActions: true,
  });
}

function ensureOutput(args) {
  fs.mkdirSync(args.outDir, {recursive: true});
  const datasetPath = path.join(args.outDir, `${args.runId}_rollouts.jsonl`);
  const summaryPath = path.join(args.outDir, `${args.runId}_rollouts.summary.json`);
  const logDir = args.logDir || path.join(repoRoot, 'logs', 'battles', `${args.runId}_rollouts`);
  if (!args.overwrite && fs.existsSync(datasetPath)) {
    throw new Error(`Rollout dataset exists at ${relativePath(datasetPath)}; pass --overwrite`);
  }
  fs.mkdirSync(logDir, {recursive: true});
  return {datasetPath, summaryPath, logDir};
}

function traceRows(tracePath) {
  return fs.readFileSync(tracePath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map(line => JSON.parse(line));
}

function buildRolloutRows({result, tracePath, opponentId, rlSide}) {
  const rows = traceRows(tracePath).filter(row => row.agent === 'ppo_policy_agent' && row.side === rlSide);
  const finalWinnerSide = winnerSide(result.winner);
  return rows.map((row, index) => {
    const ppo = row.agent_diagnostics?.ppo_policy || {};
    const done = index === rows.length - 1;
    const reward = done ? (finalWinnerSide === row.side ? 1 : -1) : 0;
    const actionIndex = Number.isInteger(ppo.action_index) ?
      ppo.action_index :
      (row.legal_actions || []).indexOf(row.chosen_action);
    return {
      run_id: null,
      battle_id: row.battle_id,
      trajectory_id: `${row.battle_id}:${row.side}`,
      step_index: index,
      turn: row.turn,
      side: row.side,
      opponent_id: opponentId,
      team: row.team,
      lead: row.lead,
      p1_team: result.p1.team_id,
      p2_team: result.p2.team_id,
      state: {
        request: row.request,
        public_state: row.public_state,
        agent_diagnostics: row.agent_diagnostics || null,
      },
      request_type: row.request?.teamPreview ? 'team_preview' :
        (Array.isArray(row.request?.forceSwitch) && row.request.forceSwitch.some(Boolean) ? 'force_switch' :
          (Array.isArray(row.request?.active) ? 'move' : 'other')),
      legal_actions: row.legal_actions || [],
      action: row.chosen_action,
      action_index: actionIndex,
      log_prob: Number(ppo.log_prob),
      value_prediction: Number(ppo.value_prediction),
      entropy: Number(ppo.entropy),
      selected_probability: Number(ppo.selected_probability),
      epsilon: Number(ppo.epsilon || 0),
      reward,
      done,
      winner: result.winner,
      winner_side: finalWinnerSide,
      trace_jsonl_path: relativePath(tracePath),
    };
  });
}

function pairings(opponents, sideSwaps) {
  const rows = [];
  for (const opponent of opponents) {
    rows.push({rlSide: 'p1', opponent});
    if (sideSwaps) rows.push({rlSide: 'p2', opponent});
  }
  return rows;
}

async function generate(args) {
  if (!fs.existsSync(args.model)) throw new Error(`PPO/warm-start checkpoint not found: ${args.model}`);
  const outputs = ensureOutput(args);
  const pool = loadTeamPool();
  const rows = [];
  const games = [];

  for (const pairing of pairings(args.opponents, args.sideSwaps)) {
    for (let game = 1; game <= args.games; game++) {
      const seed = `${args.seed}:${pairing.opponent}:${pairing.rlSide}:${game}`;
      const rng = makeRng(seed);
      const p1Team = findTeam(pool, null, rng);
      const p2Team = findTeam(pool, null, rng);
      const p1Lead = findLeadMode(p1Team, null, rng);
      const p2Lead = findLeadMode(p2Team, null, rng);
      const rlAgent = createRlAgent(args, pool.format_id);
      const opponentAgent = createOpponentAgent(pairing.opponent, args, pool.format_id);
      const result = await runBattle({
        pool,
        seed,
        p1Team,
        p2Team,
        p1Lead,
        p2Lead,
        p1Agent: pairing.rlSide === 'p1' ? rlAgent : opponentAgent,
        p2Agent: pairing.rlSide === 'p2' ? rlAgent : opponentAgent,
        logDir: outputs.logDir,
        rng,
      });
      const tracePath = path.resolve(repoRoot, result.trace_jsonl_path);
      const rolloutRows = buildRolloutRows({
        result,
        tracePath,
        opponentId: pairing.opponent,
        rlSide: pairing.rlSide,
      }).map(row => ({...row, run_id: args.runId}));
      rows.push(...rolloutRows);
      games.push({
        seed,
        opponent: pairing.opponent,
        rl_side: pairing.rlSide,
        winner: result.winner,
        winner_side: winnerSide(result.winner),
        turns: result.turns,
        rollout_rows: rolloutRows.length,
        trace_jsonl_path: result.trace_jsonl_path,
      });
      console.log(`${pairing.rlSide} final_rl vs ${pairing.opponent} game=${game} winner=${result.winner} turns=${result.turns} rollout_rows=${rolloutRows.length}`);
    }
  }

  fs.writeFileSync(outputs.datasetPath, rows.map(row => JSON.stringify(row)).join('\n') + '\n', 'utf8');
  const summary = {
    created_at: new Date().toISOString(),
    run_id: args.runId,
    model_path: relativePath(args.model),
    dataset_path: relativePath(outputs.datasetPath),
    log_dir: relativePath(outputs.logDir),
    games,
    rows: rows.length,
    opponents: args.opponents,
    side_swaps: args.sideSwaps,
  };
  fs.writeFileSync(outputs.summaryPath, JSON.stringify(summary, null, 2) + '\n', 'utf8');
  console.log(`Wrote rollouts: ${relativePath(outputs.datasetPath)} rows=${rows.length}`);
  console.log(`Wrote summary: ${relativePath(outputs.summaryPath)}`);
  return {datasetPath: outputs.datasetPath, summaryPath: outputs.summaryPath};
}

generate(parseArgs(process.argv.slice(2)))
  .finally(() => closeTorchPolicyScorers())
  .catch(error => {
    console.error(error.stack || error.message || error);
    process.exit(1);
  });
