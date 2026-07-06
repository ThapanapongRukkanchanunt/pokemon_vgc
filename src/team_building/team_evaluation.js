const fs = require('node:fs');
const path = require('node:path');
const {closeTorchPolicyScorers, createAgent} = require('../agents');
const {
  findLeadMode,
  makeRng,
  runBattle,
} = require('../battle/run_battle');
const {
  hashString,
  relativePath,
  resolveRepoPath,
} = require('./team_candidates');

const repoRoot = path.join(__dirname, '..', '..');

const POLICY_MODEL_AGENT_IDS = new Set([
  'bc_policy',
  'bc_policy_agent',
  'bc',
  'policy_selector',
  'policy_selector_agent',
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
  'shallow_search',
  'search',
  'search_selector',
  'search_balanced',
  'search_stable',
  'search_comeback',
  'search_closing',
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
  'shallow_search',
  'search',
  'search_selector',
  'search_balanced',
  'search_stable',
  'search_comeback',
  'search_closing',
  'hmm_belief',
  'hmm_search',
  'belief_search',
  'hmm_belief_agent',
]);

function canonicalAgentId(name) {
  return String(name || '').toLowerCase();
}

function agentNeedsPolicyModel(name) {
  return POLICY_MODEL_AGENT_IDS.has(canonicalAgentId(name));
}

function agentNeedsValueModel(name) {
  return VALUE_MODEL_AGENT_IDS.has(canonicalAgentId(name));
}

function winnerSide(winner) {
  if (typeof winner !== 'string') return null;
  const match = winner.match(/\b(P[12])$/i);
  return match ? match[1].toLowerCase() : null;
}

function countRecoveryRows(tracePath) {
  const recovery = {total: 0, by_side: {p1: 0, p2: 0, unknown: 0}};
  if (!tracePath || !fs.existsSync(tracePath)) return recovery;
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

function createTeamEvalAgent(name, args, formatId) {
  const options = {formatId};
  const agentId = canonicalAgentId(name);
  if (POLICY_MODEL_AGENT_IDS.has(agentId)) options.modelPath = args.modelPath;
  if (VALUE_MODEL_AGENT_IDS.has(agentId)) options.valueModelPath = args.valueModelPath;
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
  if ([
    'policy_value',
    'policy_value_risk',
    'policy_value_risk_selector',
    'risk_selector',
    'hmm_belief',
    'hmm_search',
    'belief_search',
    'hmm_belief_agent',
  ].includes(agentId)) {
    options.riskMode = args.riskMode || 'balanced';
  }
  return createAgent(name, options);
}

function assertAgentInputs(args) {
  const agents = [args.agent, args.opponentAgent];
  if (agents.some(agentNeedsPolicyModel) && !fs.existsSync(resolveRepoPath(args.modelPath))) {
    throw new Error(`Policy model not found: ${args.modelPath}`);
  }
  if (agents.some(agentNeedsValueModel) && !fs.existsSync(resolveRepoPath(args.valueModelPath))) {
    throw new Error(`Value model not found: ${args.valueModelPath}`);
  }
}

function loadCache(cachePath) {
  const resolved = resolveRepoPath(cachePath);
  const rows = new Map();
  if (!fs.existsSync(resolved)) return rows;
  for (const line of fs.readFileSync(resolved, 'utf8').split(/\r?\n/).filter(Boolean)) {
    const row = JSON.parse(line);
    if (row.key) rows.set(row.key, row);
  }
  return rows;
}

function appendCacheRow(cachePath, row) {
  const resolved = resolveRepoPath(cachePath);
  fs.mkdirSync(path.dirname(resolved), {recursive: true});
  fs.appendFileSync(resolved, `${JSON.stringify(row)}\n`, 'utf8');
}

function modelCachePath(filePath, needed) {
  if (!needed) return null;
  return relativePath(resolveRepoPath(filePath));
}

function battleCacheKey({
  formatId,
  candidateHash,
  metagameTeamId,
  candidateSide,
  battleSeed,
  agent,
  opponentAgent,
  modelPath,
  valueModelPath,
  riskMode,
}) {
  return hashString(JSON.stringify({
    version: 1,
    formatId,
    candidateHash,
    metagameTeamId,
    candidateSide,
    battleSeed,
    agent,
    opponentAgent,
    modelPath,
    valueModelPath,
    riskMode,
  }));
}

function ensureCandidateRow(candidateRows, candidate) {
  if (!candidateRows.has(candidate.id)) {
    candidateRows.set(candidate.id, {
      candidate_id: candidate.id,
      candidate_name: candidate.name,
      candidate_hash: candidate.hash || null,
      species: candidate.species || [],
      source_team_ids: candidate.source_team_ids || [],
      games: 0,
      wins: 0,
      losses: 0,
      unknown: 0,
      recovery_rows: 0,
      matchups: new Map(),
      game_rows: [],
    });
  }
  return candidateRows.get(candidate.id);
}

function ensureMatchupRow(candidateRow, metagameTeam) {
  if (!candidateRow.matchups.has(metagameTeam.id)) {
    candidateRow.matchups.set(metagameTeam.id, {
      metagame_team_id: metagameTeam.id,
      metagame_team_name: metagameTeam.name,
      games: 0,
      wins: 0,
      losses: 0,
      unknown: 0,
      recovery_rows: 0,
      win_rate: 0,
    });
  }
  return candidateRow.matchups.get(metagameTeam.id);
}

function recordCandidateGame({candidateRows, candidate, metagameTeam, gameRow}) {
  const candidateRow = ensureCandidateRow(candidateRows, candidate);
  const matchupRow = ensureMatchupRow(candidateRow, metagameTeam);
  const candidateWon = gameRow.candidate_won;
  candidateRow.games += 1;
  matchupRow.games += 1;
  candidateRow.recovery_rows += gameRow.recovery_rows || 0;
  matchupRow.recovery_rows += gameRow.recovery_rows || 0;

  if (candidateWon === true) {
    candidateRow.wins += 1;
    matchupRow.wins += 1;
  } else if (gameRow.winner_side === 'unknown') {
    candidateRow.unknown += 1;
    matchupRow.unknown += 1;
  } else {
    candidateRow.losses += 1;
    matchupRow.losses += 1;
  }
  candidateRow.game_rows.push(gameRow);
}

function finalizeCandidateRow(row) {
  const matchups = [...row.matchups.values()].map(matchup => {
    const decisive = matchup.wins + matchup.losses;
    return {
      ...matchup,
      win_rate: decisive ? matchup.wins / decisive : 0,
    };
  }).sort((a, b) => a.metagame_team_id.localeCompare(b.metagame_team_id));
  const decisive = row.wins + row.losses;
  const winRate = decisive ? row.wins / decisive : 0;
  const matchupWinRates = matchups.length ? matchups.map(matchup => matchup.win_rate) : [0];
  const worst = Math.min(...matchupWinRates);
  const best = Math.max(...matchupWinRates);
  const mean = matchupWinRates.reduce((sum, value) => sum + value, 0) / matchupWinRates.length;
  const spread = best - worst;
  const recoveryPenalty = Math.min(0.25, row.recovery_rows * 0.01);
  const score = (0.6 * mean) + (0.4 * worst) - (0.1 * spread) - recoveryPenalty;
  return {
    candidate_id: row.candidate_id,
    candidate_name: row.candidate_name,
    candidate_hash: row.candidate_hash,
    species: row.species,
    source_team_ids: row.source_team_ids,
    games: row.games,
    wins: row.wins,
    losses: row.losses,
    unknown: row.unknown,
    recovery_rows: row.recovery_rows,
    win_rate: winRate,
    matchup_mean_win_rate: mean,
    worst_matchup_win_rate: worst,
    best_matchup_win_rate: best,
    matchup_spread: spread,
    score,
    matchups,
    games_detail: row.game_rows,
  };
}

async function runOrReadCachedBattle({
  args,
  cache,
  candidatePool,
  candidate,
  metagameTeam,
  candidateSide,
  battleSeed,
}) {
  const needsPolicy = [args.agent, args.opponentAgent].some(agentNeedsPolicyModel);
  const needsValue = [args.agent, args.opponentAgent].some(agentNeedsValueModel);
  const key = battleCacheKey({
    formatId: candidatePool.format_id,
    candidateHash: candidate.hash,
    metagameTeamId: metagameTeam.id,
    candidateSide,
    battleSeed,
    agent: args.agent,
    opponentAgent: args.opponentAgent,
    modelPath: modelCachePath(args.modelPath, needsPolicy),
    valueModelPath: modelCachePath(args.valueModelPath, needsValue),
    riskMode: args.riskMode,
  });

  if (cache.has(key)) {
    return {...cache.get(key), cache_hit: true};
  }

  const rng = makeRng(battleSeed);
  const candidateLead = findLeadMode(candidate, null, rng);
  const metagameLead = findLeadMode(metagameTeam, null, rng);
  const candidateAgent = createTeamEvalAgent(args.agent, args, candidatePool.format_id);
  const opponentAgent = createTeamEvalAgent(args.opponentAgent, args, candidatePool.format_id);
  const p1IsCandidate = candidateSide === 'p1';
  const result = await runBattle({
    pool: candidatePool,
    seed: battleSeed,
    p1Team: p1IsCandidate ? candidate : metagameTeam,
    p2Team: p1IsCandidate ? metagameTeam : candidate,
    p1Lead: p1IsCandidate ? candidateLead : metagameLead,
    p2Lead: p1IsCandidate ? metagameLead : candidateLead,
    p1Agent: p1IsCandidate ? candidateAgent : opponentAgent,
    p2Agent: p1IsCandidate ? opponentAgent : candidateAgent,
    logDir: args.logDir,
    rng,
  });
  const side = winnerSide(result.winner) || 'unknown';
  const recovery = countRecoveryRows(path.resolve(repoRoot, result.trace_jsonl_path));
  const row = {
    key,
    created_at: new Date().toISOString(),
    candidate_id: candidate.id,
    candidate_hash: candidate.hash,
    metagame_team_id: metagameTeam.id,
    candidate_side: candidateSide,
    battle_seed: battleSeed,
    agent: args.agent,
    opponent_agent: args.opponentAgent,
    winner: result.winner,
    winner_side: side,
    candidate_won: side === candidateSide ? true : (side === 'unknown' ? null : false),
    turns: result.turns,
    recovery_rows: recovery.total,
    p1_team: result.p1.team_id,
    p2_team: result.p2.team_id,
    p1_lead: result.p1.lead_id,
    p2_lead: result.p2.lead_id,
    summary_json_path: result.summary_json_path,
    trace_jsonl_path: result.trace_jsonl_path,
    cache_hit: false,
  };
  cache.set(key, row);
  appendCacheRow(args.cachePath, row);
  return row;
}

async function evaluateCandidatePool({
  candidatePool,
  metagamePool,
  candidates,
  metagameTeams,
  args,
}) {
  assertAgentInputs(args);
  fs.mkdirSync(resolveRepoPath(args.logDir), {recursive: true});
  const cache = loadCache(args.cachePath);
  const candidateRows = new Map();
  const cacheStats = {path: relativePath(resolveRepoPath(args.cachePath)), hits: 0, misses: 0};
  const sides = args.includeSideSwaps ? ['p1', 'p2'] : ['p1'];

  for (const candidate of candidates) {
    for (const metagameTeam of metagameTeams) {
      for (const candidateSide of sides) {
        for (let game = 1; game <= args.gamesPerMatchup; game++) {
          const battleSeed = [
            args.seed,
            candidate.id,
            metagameTeam.id,
            candidateSide,
            game,
          ].join(':');
          const row = await runOrReadCachedBattle({
            args,
            cache,
            candidatePool,
            candidate,
            metagameTeam,
            candidateSide,
            battleSeed,
          });
          if (row.cache_hit) cacheStats.hits += 1;
          else cacheStats.misses += 1;
          recordCandidateGame({candidateRows, candidate, metagameTeam, gameRow: row});
          console.log(
            `${candidate.id} vs ${metagameTeam.id} side=${candidateSide} game=${game} ` +
            `winner=${row.winner} recovery=${row.recovery_rows} cache=${row.cache_hit ? 'hit' : 'miss'}`
          );
        }
      }
    }
  }

  const candidateTable = [...candidateRows.values()]
    .map(finalizeCandidateRow)
    .sort((a, b) => b.score - a.score || b.worst_matchup_win_rate - a.worst_matchup_win_rate ||
      b.win_rate - a.win_rate || a.candidate_id.localeCompare(b.candidate_id));

  return {
    created_at: new Date().toISOString(),
    seed: args.seed,
    agent: args.agent,
    opponent_agent: args.opponentAgent,
    games_per_matchup: args.gamesPerMatchup,
    include_side_swaps: args.includeSideSwaps,
    candidate_count: candidates.length,
    metagame_team_count: metagameTeams.length,
    log_dir: relativePath(resolveRepoPath(args.logDir)),
    cache: cacheStats,
    candidate_table: candidateTable,
  };
}

function printCandidateTable(rows) {
  console.log('candidate,games,wins,losses,unknown,win_rate,worst_matchup,spread,score,recovery_rows');
  for (const row of rows) {
    console.log([
      row.candidate_id,
      row.games,
      row.wins,
      row.losses,
      row.unknown,
      row.win_rate.toFixed(3),
      row.worst_matchup_win_rate.toFixed(3),
      row.matchup_spread.toFixed(3),
      row.score.toFixed(3),
      row.recovery_rows,
    ].join(','));
  }
}

module.exports = {
  agentNeedsPolicyModel,
  agentNeedsValueModel,
  closeTorchPolicyScorers,
  countRecoveryRows,
  createTeamEvalAgent,
  evaluateCandidatePool,
  printCandidateTable,
  winnerSide,
};
