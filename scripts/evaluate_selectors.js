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

const ALL_MATCHUPS = {
  maxdamage_vs_random: {p1: 'maxdamage', p2: 'random'},
  bc_model_vs_random: {p1: 'bc_policy', p2: 'random'},
  heuristic_vs_random: {p1: 'heuristic_selector', p2: 'random'},
  hybrid_vs_random: {p1: 'hybrid_selector', p2: 'random'},
  value_vs_random: {p1: 'value_selector', p2: 'random'},
  risk_balanced_vs_random: {p1: 'risk_balanced', p2: 'random'},
  search_balanced_vs_random: {p1: 'search_balanced', p2: 'random'},
  heuristic_vs_maxdamage: {p1: 'heuristic_selector', p2: 'maxdamage'},
  hybrid_vs_maxdamage: {p1: 'hybrid_selector', p2: 'maxdamage'},
  value_vs_maxdamage: {p1: 'value_selector', p2: 'maxdamage'},
  risk_balanced_vs_maxdamage: {p1: 'risk_balanced', p2: 'maxdamage'},
  search_balanced_vs_maxdamage: {p1: 'search_balanced', p2: 'maxdamage'},
  hmm_belief_vs_random: {p1: 'hmm_belief', p2: 'random'},
  hmm_belief_vs_maxdamage: {p1: 'hmm_belief', p2: 'maxdamage'},
  hmm_belief_vs_heuristic: {p1: 'hmm_belief', p2: 'heuristic_selector'},
  hybrid_vs_bc_model: {p1: 'hybrid_selector', p2: 'bc_policy'},
  risk_balanced_vs_bc_model: {p1: 'risk_balanced', p2: 'bc_policy'},
  search_balanced_vs_bc_model: {p1: 'search_balanced', p2: 'bc_policy'},
  risk_balanced_vs_heuristic: {p1: 'risk_balanced', p2: 'heuristic_selector'},
  search_balanced_vs_heuristic: {p1: 'search_balanced', p2: 'heuristic_selector'},
  search_balanced_vs_risk_balanced: {p1: 'search_balanced', p2: 'risk_balanced'},
  torch_policy_vs_random: {p1: 'torch_policy', p2: 'random'},
  torch_policy_vs_maxdamage: {p1: 'torch_policy', p2: 'maxdamage'},
  torch_policy_vs_heuristic: {p1: 'torch_policy', p2: 'heuristic_selector'},
  final_rl_vs_random: {p1: 'final_rl', p2: 'random'},
  final_rl_vs_maxdamage: {p1: 'final_rl', p2: 'maxdamage'},
  final_rl_vs_heuristic: {p1: 'final_rl', p2: 'heuristic_selector'},
  final_rl_vs_search_balanced: {p1: 'final_rl', p2: 'search_balanced'},
  final_rl_vs_hmm_belief: {p1: 'final_rl', p2: 'hmm_belief'},
  risk_stable_vs_comeback: {p1: 'risk_stable', p2: 'risk_comeback'},
  risk_closing_vs_comeback: {p1: 'risk_closing', p2: 'risk_comeback'},
};

const DEFAULT_MATCHUP_IDS = [
  'maxdamage_vs_random',
  'bc_model_vs_random',
  'heuristic_vs_random',
  'hybrid_vs_random',
  'heuristic_vs_maxdamage',
  'hybrid_vs_maxdamage',
  'hybrid_vs_bc_model',
];

const POLICY_MODEL_AGENT_IDS = new Set([
  'bc_policy',
  'bc_policy_agent',
  'bc',
  'torch',
  'torch_policy',
  'torch_policy_agent',
  'pytorch',
  'pytorch_policy',
  'ppo_policy',
  'rl_policy',
  'final_rl',
  'final_rl_agent',
  'policy_selector',
  'policy_selector_agent',
  'model_only_selector',
  'hybrid',
  'hybrid_selector',
  'hybrid_selector_agent',
  'model_plus_heuristic',
  'policy_heuristic',
  'policy_value',
  'policy_value_risk',
  'policy_value_risk_selector',
  'risk_selector',
  'risk_balanced',
  'risk_stable',
  'risk_comeback',
  'risk_closing',
  'policy_value_risk_balanced',
  'policy_value_risk_stable',
  'policy_value_risk_comeback',
  'policy_value_risk_closing',
  'shallow_search',
  'search',
  'search_selector',
  'search_balanced',
  'search_stable',
  'search_comeback',
  'search_closing',
  'shallow_search_balanced',
  'shallow_search_stable',
  'shallow_search_comeback',
  'shallow_search_closing',
  'hmm_belief',
  'hmm_search',
  'belief_search',
  'hmm_belief_agent',
]);

const VALUE_MODEL_AGENT_IDS = new Set([
  'value',
  'value_selector',
  'value_selector_agent',
  'q_selector',
  'policy_value',
  'policy_value_risk',
  'policy_value_risk_selector',
  'risk_selector',
  'risk_balanced',
  'risk_stable',
  'risk_comeback',
  'risk_closing',
  'policy_value_risk_balanced',
  'policy_value_risk_stable',
  'policy_value_risk_comeback',
  'policy_value_risk_closing',
  'shallow_search',
  'search',
  'search_selector',
  'search_balanced',
  'search_stable',
  'search_comeback',
  'search_closing',
  'shallow_search_balanced',
  'shallow_search_stable',
  'shallow_search_comeback',
  'shallow_search_closing',
  'hmm_belief',
  'hmm_search',
  'belief_search',
  'hmm_belief_agent',
]);

const PPO_MODEL_AGENT_IDS = new Set([
  'ppo_policy',
  'rl_policy',
  'final_rl',
  'final_rl_agent',
]);

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    model: path.join(repoRoot, 'models', 'bc_policy', 'trace_test_maxdamage', 'model.json'),
    rlModel: null,
    valueModel: path.join(repoRoot, 'models', 'value_model', 'phase4_mixed_q', 'model.json'),
    outDir: path.join(repoRoot, 'experiments', 'selectors', 'phase3_eval'),
    logDir: path.join(repoRoot, 'logs', 'battles', 'selector_eval'),
    games: 1,
    seed: 'phase3_selectors',
    matchups: DEFAULT_MATCHUP_IDS,
    riskMode: 'balanced',
    pythonPath: process.env.POKEMON_RL_PYTHON || null,
    torchDevice: process.env.POKEMON_RL_TORCH_DEVICE || null,
    overwrite: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--model') {
      args.model = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--rl-model') {
      args.rlModel = path.resolve(repoRoot, argv[++i]);
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
    } else if (arg === '--matchups') {
      args.matchups = argv[++i].split(',').map(value => value.trim()).filter(Boolean);
    } else if (arg === '--risk-mode') {
      args.riskMode = argv[++i];
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
  if (!['stable', 'balanced', 'comeback', 'closing'].includes(args.riskMode)) {
    throw new Error('--risk-mode must be one of stable, balanced, comeback, closing');
  }
  for (const matchup of args.matchups) {
    if (!ALL_MATCHUPS[matchup]) {
      throw new Error(`Unknown matchup ${matchup}; choose from ${Object.keys(ALL_MATCHUPS).join(', ')}`);
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

function canonicalAgentId(name) {
  return String(name || '').toLowerCase();
}

function createEvalAgent(name, args, formatId) {
  const options = {formatId};
  const agentId = canonicalAgentId(name);
  if (PPO_MODEL_AGENT_IDS.has(agentId) && args.rlModel) {
    options.modelPath = args.rlModel;
  } else if (POLICY_MODEL_AGENT_IDS.has(agentId)) {
    options.modelPath = args.model;
  }
  if (VALUE_MODEL_AGENT_IDS.has(agentId)) options.valueModelPath = args.valueModel;
  if ([
    'torch',
    'torch_policy',
    'torch_policy_agent',
    'pytorch',
    'pytorch_policy',
    'ppo_policy',
    'rl_policy',
    'final_rl',
    'final_rl_agent',
  ].includes(agentId)) {
    options.pythonPath = args.pythonPath;
    options.torchDevice = args.torchDevice;
  }
  if (agentId === 'policy_value_risk' ||
      agentId === 'policy_value_risk_selector' ||
      agentId === 'policy_value' ||
      agentId === 'risk_selector' ||
      agentId === 'hmm_belief' ||
      agentId === 'hmm_search' ||
      agentId === 'belief_search' ||
      agentId === 'hmm_belief_agent') {
    options.riskMode = args.riskMode;
  }
  return createAgent(name, options);
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

function ensureOutput(args) {
  const summaryPath = path.join(args.outDir, 'summary.json');
  if (!args.overwrite && fs.existsSync(summaryPath)) {
    throw new Error(`Summary exists at ${relativePath(summaryPath)}; pass --overwrite to replace it`);
  }
  fs.mkdirSync(args.outDir, {recursive: true});
  fs.mkdirSync(args.logDir, {recursive: true});
  return summaryPath;
}

function ensureAgentRow(agentTable, agentId) {
  if (!agentTable.has(agentId)) {
    agentTable.set(agentId, {
      agent: agentId,
      games: 0,
      wins: 0,
      losses: 0,
      unknown: 0,
      recovery_rows: 0,
      win_rate: 0,
    });
  }
  return agentTable.get(agentId);
}

function recordAgentResult({agentTable, agentId, side, winner, recovery}) {
  const row = ensureAgentRow(agentTable, agentId);
  row.games += 1;
  row.recovery_rows += recovery.by_side[side] || 0;
  if (winner === side) {
    row.wins += 1;
  } else if (winner === 'unknown') {
    row.unknown += 1;
  } else {
    row.losses += 1;
  }
}

function finalizeAgentTable(agentTable) {
  return [...agentTable.values()]
    .map(row => ({
      ...row,
      win_rate: row.wins + row.losses ? row.wins / (row.wins + row.losses) : 0,
    }))
    .sort((a, b) => b.win_rate - a.win_rate || b.wins - a.wins || a.agent.localeCompare(b.agent));
}

function printAgentTable(rows) {
  console.log('agent,games,wins,losses,unknown,win_rate,recovery_rows');
  for (const row of rows) {
    console.log([
      row.agent,
      row.games,
      row.wins,
      row.losses,
      row.unknown,
      row.win_rate.toFixed(3),
      row.recovery_rows,
    ].join(','));
  }
}

async function evaluate(args) {
  const needsPolicyModel = args.matchups.some(matchupId => {
    const matchup = ALL_MATCHUPS[matchupId];
    return [matchup.p1, matchup.p2].some(agentId => {
      return POLICY_MODEL_AGENT_IDS.has(agentId) && !PPO_MODEL_AGENT_IDS.has(agentId);
    });
  });
  const needsRlModel = args.matchups.some(matchupId => {
    const matchup = ALL_MATCHUPS[matchupId];
    return PPO_MODEL_AGENT_IDS.has(matchup.p1) || PPO_MODEL_AGENT_IDS.has(matchup.p2);
  });
  const needsValueModel = args.matchups.some(matchupId => {
    const matchup = ALL_MATCHUPS[matchupId];
    return VALUE_MODEL_AGENT_IDS.has(matchup.p1) || VALUE_MODEL_AGENT_IDS.has(matchup.p2);
  });
  if (needsPolicyModel && !fs.existsSync(args.model)) throw new Error(`Policy model not found: ${args.model}`);
  const rlModelPath = args.rlModel || args.model;
  if (needsRlModel && !fs.existsSync(rlModelPath)) throw new Error(`RL model not found: ${rlModelPath}`);
  if (needsValueModel && !fs.existsSync(args.valueModel)) {
    throw new Error(`Value model not found: ${args.valueModel}`);
  }

  const summaryPath = ensureOutput(args);
  const pool = loadTeamPool();
  const agentTable = new Map();
  const summary = {
    created_at: new Date().toISOString(),
    model_path: needsPolicyModel ? relativePath(args.model) : (needsRlModel ? relativePath(rlModelPath) : null),
    policy_model_path: needsPolicyModel ? relativePath(args.model) : null,
    rl_model_path: needsRlModel ? relativePath(rlModelPath) : null,
    value_model_path: needsValueModel ? relativePath(args.valueModel) : null,
    games_per_matchup: args.games,
    seed: args.seed,
    risk_mode: args.riskMode,
    log_dir: relativePath(args.logDir),
    matchups: [],
    agent_table: [],
  };

  for (const matchupId of args.matchups) {
    const matchup = ALL_MATCHUPS[matchupId];
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
      const recovery = countRecoveryRows(tracePath);
      matchupSummary.recovery_rows += recovery.total;
      recordAgentResult({
        agentTable,
        agentId: matchup.p1,
        side: 'p1',
        winner: side,
        recovery,
      });
      recordAgentResult({
        agentTable,
        agentId: matchup.p2,
        side: 'p2',
        winner: side,
        recovery,
      });
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
        recovery_rows: recovery.total,
        summary_json_path: result.summary_json_path,
        trace_jsonl_path: result.trace_jsonl_path,
      });
      console.log(`${matchupId} game=${game} winner=${result.winner} turns=${result.turns} recovery=${recovery.total}`);
    }

    summary.matchups.push(matchupSummary);
  }

  summary.agent_table = finalizeAgentTable(agentTable);
  printAgentTable(summary.agent_table);
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
