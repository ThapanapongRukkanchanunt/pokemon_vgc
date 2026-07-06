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

const PARTICIPANTS = {
  phase2_bc: {
    label: 'Phase2BC',
    agent: 'bc_policy',
    policy: 'baseline',
  },
  phase6_search_policy: {
    label: 'Phase6SearchPolicy',
    agent: 'bc_policy',
    policy: 'current',
  },
  torch_current_policy: {
    label: 'TorchCurrentPolicy',
    agent: 'torch_policy',
    policy: 'current',
  },
  torch_baseline_policy: {
    label: 'TorchBaselinePolicy',
    agent: 'torch_policy',
    policy: 'baseline',
  },
  heuristic: {
    label: 'Heuristic',
    agent: 'heuristic_selector',
  },
  risk_balanced: {
    label: 'RiskBalanced',
    agent: 'risk_balanced',
    policy: 'baseline',
    value: true,
  },
  shallow_search: {
    label: 'ShallowSearch',
    agent: 'search_balanced',
    policy: 'current',
    value: true,
  },
};

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    model: path.join(repoRoot, 'models', 'bc_policy', 'phase6_search_improved', 'model.json'),
    baselineModel: path.join(repoRoot, 'models', 'bc_policy', 'trace_test_maxdamage', 'model.json'),
    valueModel: path.join(repoRoot, 'models', 'value_model', 'phase4_mixed_q', 'model.json'),
    outDir: path.join(repoRoot, 'experiments', 'self_play', 'phase6_league'),
    logDir: path.join(repoRoot, 'logs', 'battles', 'phase6_self_play_league'),
    games: 1,
    seed: 'phase6_self_play_league',
    participants: ['phase2_bc', 'phase6_search_policy', 'heuristic', 'shallow_search'],
    includeSideSwaps: true,
    pythonPath: process.env.POKEMON_RL_PYTHON || null,
    torchDevice: process.env.POKEMON_RL_TORCH_DEVICE || null,
    overwrite: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--model') {
      args.model = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--baseline-model') {
      args.baselineModel = path.resolve(repoRoot, argv[++i]);
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
    } else if (arg === '--participants') {
      args.participants = argv[++i].split(',').map(value => value.trim()).filter(Boolean);
    } else if (arg === '--python') {
      args.pythonPath = argv[++i];
    } else if (arg === '--torch-device') {
      args.torchDevice = argv[++i];
    } else if (arg === '--no-side-swaps') {
      args.includeSideSwaps = false;
    } else if (arg === '--overwrite') {
      args.overwrite = true;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (args.games <= 0) throw new Error('--games must be > 0');
  if (args.participants.length < 2) throw new Error('--participants must include at least two ids');
  for (const id of args.participants) {
    if (!PARTICIPANTS[id]) {
      throw new Error(`Unknown participant ${id}; choose from ${Object.keys(PARTICIPANTS).join(', ')}`);
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

function ensureOutput(args) {
  const summaryPath = path.join(args.outDir, 'summary.json');
  if (!args.overwrite && fs.existsSync(summaryPath)) {
    throw new Error(`Summary exists at ${relativePath(summaryPath)}; pass --overwrite to replace it`);
  }
  fs.mkdirSync(args.outDir, {recursive: true});
  fs.mkdirSync(args.logDir, {recursive: true});
  return summaryPath;
}

function modelPathForParticipant(participant, args) {
  if (participant.policy === 'current') return args.model;
  if (participant.policy === 'baseline') return args.baselineModel;
  return null;
}

function assertInputs(args) {
  const needed = new Set();
  for (const id of args.participants) {
    const participant = PARTICIPANTS[id];
    const modelPath = modelPathForParticipant(participant, args);
    if (modelPath) needed.add(modelPath);
    if (participant.value) needed.add(args.valueModel);
  }
  for (const filePath of needed) {
    if (!fs.existsSync(filePath)) throw new Error(`Required model not found: ${filePath}`);
  }
}

function createLeagueAgent(id, args, formatId) {
  const participant = PARTICIPANTS[id];
  const options = {formatId};
  const modelPath = modelPathForParticipant(participant, args);
  if (modelPath) options.modelPath = modelPath;
  if (participant.value) options.valueModelPath = args.valueModel;
  if (participant.agent === 'torch_policy') {
    options.pythonPath = args.pythonPath;
    options.torchDevice = args.torchDevice;
  }
  const agent = createAgent(participant.agent, options);
  return {
    ...agent,
    name: `${id}_agent`,
    displayName: participant.label,
    participantId: id,
  };
}

function countRecoveryRows(tracePath) {
  const recovery = {total: 0, by_side: {p1: 0, p2: 0, unknown: 0}};
  if (!fs.existsSync(tracePath)) return recovery;
  for (const line of fs.readFileSync(tracePath, 'utf8').split(/\r?\n/).filter(Boolean)) {
    try {
      const row = JSON.parse(line);
      if (!row.error_recovery) continue;
      const side = row.side === 'p1' || row.side === 'p2' ? row.side : 'unknown';
      recovery.total += 1;
      recovery.by_side[side] += 1;
    } catch (error) {
      recovery.total += 1;
      recovery.by_side.unknown += 1;
    }
  }
  return recovery;
}

function ensureAgentRow(agentTable, participantId) {
  if (!agentTable.has(participantId)) {
    agentTable.set(participantId, {
      participant: participantId,
      games: 0,
      wins: 0,
      losses: 0,
      unknown: 0,
      recovery_rows: 0,
      win_rate: 0,
    });
  }
  return agentTable.get(participantId);
}

function recordAgentResult({agentTable, participantId, side, winner, recovery}) {
  const row = ensureAgentRow(agentTable, participantId);
  row.games += 1;
  row.recovery_rows += recovery.by_side[side] || 0;
  if (winner === side) row.wins += 1;
  else if (winner === 'unknown') row.unknown += 1;
  else row.losses += 1;
}

function finalizeAgentTable(agentTable) {
  return [...agentTable.values()]
    .map(row => ({
      ...row,
      win_rate: row.wins + row.losses ? row.wins / (row.wins + row.losses) : 0,
    }))
    .sort((a, b) => b.win_rate - a.win_rate || b.wins - a.wins || a.participant.localeCompare(b.participant));
}

function schedulePairings(participants, includeSideSwaps) {
  const pairings = [];
  for (let i = 0; i < participants.length; i++) {
    for (let j = i + 1; j < participants.length; j++) {
      pairings.push({p1: participants[i], p2: participants[j]});
      if (includeSideSwaps) pairings.push({p1: participants[j], p2: participants[i]});
    }
  }
  return pairings;
}

function printAgentTable(rows) {
  console.log('participant,games,wins,losses,unknown,win_rate,recovery_rows');
  for (const row of rows) {
    console.log([
      row.participant,
      row.games,
      row.wins,
      row.losses,
      row.unknown,
      row.win_rate.toFixed(3),
      row.recovery_rows,
    ].join(','));
  }
}

async function runLeague(args) {
  assertInputs(args);
  const summaryPath = ensureOutput(args);
  const pool = loadTeamPool();
  const agentTable = new Map();
  const pairings = schedulePairings(args.participants, args.includeSideSwaps);
  const summary = {
    created_at: new Date().toISOString(),
    seed: args.seed,
    games_per_pairing: args.games,
    participants: args.participants,
    participant_config: Object.fromEntries(args.participants.map(id => [id, PARTICIPANTS[id]])),
    model_path: relativePath(args.model),
    baseline_model_path: relativePath(args.baselineModel),
    value_model_path: relativePath(args.valueModel),
    log_dir: relativePath(args.logDir),
    pairings: [],
    agent_table: [],
  };

  for (const pairing of pairings) {
    const pairingSummary = {
      p1: pairing.p1,
      p2: pairing.p2,
      games: [],
      wins: {p1: 0, p2: 0, unknown: 0},
      recovery_rows: 0,
    };
    for (let game = 1; game <= args.games; game++) {
      const battleSeed = `${args.seed}:${pairing.p1}_vs_${pairing.p2}:${game}`;
      const rng = makeRng(battleSeed);
      const p1Team = findTeam(pool, null, rng);
      const p2Team = findTeam(pool, null, rng);
      const p1Lead = findLeadMode(p1Team, null, rng);
      const p2Lead = findLeadMode(p2Team, null, rng);
      const p1Agent = createLeagueAgent(pairing.p1, args, pool.format_id);
      const p2Agent = createLeagueAgent(pairing.p2, args, pool.format_id);
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
      pairingSummary.wins[side] += 1;
      const recovery = countRecoveryRows(path.resolve(repoRoot, result.trace_jsonl_path));
      pairingSummary.recovery_rows += recovery.total;
      recordAgentResult({
        agentTable,
        participantId: pairing.p1,
        side: 'p1',
        winner: side,
        recovery,
      });
      recordAgentResult({
        agentTable,
        participantId: pairing.p2,
        side: 'p2',
        winner: side,
        recovery,
      });
      pairingSummary.games.push({
        game,
        seed: battleSeed,
        winner: result.winner,
        winner_side: side,
        turns: result.turns,
        p1_team: result.p1.team_id,
        p2_team: result.p2.team_id,
        p1_lead: result.p1.lead_id,
        p2_lead: result.p2.lead_id,
        recovery_rows: recovery.total,
        summary_json_path: result.summary_json_path,
        trace_jsonl_path: result.trace_jsonl_path,
      });
      console.log(
        `${pairing.p1}_vs_${pairing.p2} game=${game} ` +
        `winner=${result.winner} turns=${result.turns} recovery=${recovery.total}`
      );
    }
    summary.pairings.push(pairingSummary);
  }

  summary.agent_table = finalizeAgentTable(agentTable);
  printAgentTable(summary.agent_table);
  fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  console.log(`Wrote summary: ${relativePath(summaryPath)}`);
  return {summaryPath, summary};
}

function main() {
  runLeague(parseArgs(process.argv.slice(2)))
    .finally(() => closeTorchPolicyScorers())
    .catch(error => {
    console.error(`FAIL ${error.stack || error.message}`);
    process.exit(1);
  });
}

main();
