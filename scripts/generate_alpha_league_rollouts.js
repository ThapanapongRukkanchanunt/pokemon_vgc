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

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseNumber(value, name) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`${name} must be numeric`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    runId: 'mb_alpha_league_iter_001',
    modelsDir: path.join(repoRoot, 'models', 'torch', 'mb_alpha_league', 'agents'),
    teamPreviewModel: null,
    outDir: path.join(repoRoot, 'data', 'datasets', 'rl'),
    logDir: null,
    games: 1000,
    seed: null,
    epsilon: 0.1,
    topK: 4,
    rolloutMaxDecisions: 120,
    pythonPath: process.env.POKEMON_RL_PYTHON || null,
    torchDevice: process.env.POKEMON_RL_TORCH_DEVICE || null,
    overwrite: false,
    compactLogs: false,
    deleteBattleLogs: false,
    progressEvery: 25,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--run-id') {
      args.runId = argv[++i];
    } else if (arg === '--models-dir') {
      args.modelsDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--team-preview-model') {
      args.teamPreviewModel = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--out-dir') {
      args.outDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--log-dir') {
      args.logDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--games') {
      args.games = parseInteger(argv[++i], '--games');
    } else if (arg === '--seed') {
      args.seed = argv[++i];
    } else if (arg === '--epsilon') {
      args.epsilon = parseNumber(argv[++i], '--epsilon');
    } else if (arg === '--top-k') {
      args.topK = parseInteger(argv[++i], '--top-k');
    } else if (arg === '--rollout-max-decisions') {
      args.rolloutMaxDecisions = parseInteger(argv[++i], '--rollout-max-decisions');
    } else if (arg === '--python') {
      args.pythonPath = argv[++i];
    } else if (arg === '--torch-device') {
      args.torchDevice = argv[++i];
    } else if (arg === '--progress-every') {
      args.progressEvery = parseInteger(argv[++i], '--progress-every');
    } else if (arg === '--compact-logs') {
      args.compactLogs = true;
    } else if (arg === '--delete-battle-logs') {
      args.deleteBattleLogs = true;
    } else if (arg === '--overwrite') {
      args.overwrite = true;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (!args.runId || /[\\/:*?"<>|]/.test(args.runId)) throw new Error('--run-id must be filename-safe');
  if (args.games <= 0) throw new Error('--games must be > 0');
  if (args.epsilon < 0 || args.epsilon > 1) throw new Error('--epsilon must be between 0 and 1');
  if (args.topK <= 0) throw new Error('--top-k must be > 0');
  if (args.rolloutMaxDecisions <= 0) throw new Error('--rollout-max-decisions must be > 0');
  if (!args.seed) args.seed = args.runId;
  if (!args.logDir) args.logDir = path.join(repoRoot, 'logs', 'battles', `${args.runId}_league`);
  return args;
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function assertInsideRepo(filePath, label) {
  const resolved = path.resolve(filePath);
  const relative = path.relative(repoRoot, resolved);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`${label} must stay inside repo: ${resolved}`);
  }
}

function removeIfExists(filePath, label) {
  if (!fs.existsSync(filePath)) return;
  assertInsideRepo(filePath, label);
  fs.rmSync(filePath, {recursive: true, force: true});
}

function winnerSide(winner) {
  if (typeof winner !== 'string') return 'unknown';
  const match = winner.match(/\b(P[12])$/i);
  return match ? match[1].toLowerCase() : 'unknown';
}

function modelPathForTeam(args, teamId) {
  return path.join(args.modelsDir, teamId, 'checkpoint.pt');
}

function assertModelInputs(args, pool) {
  for (const team of pool.teams) {
    const modelPath = modelPathForTeam(args, team.id);
    if (!fs.existsSync(modelPath)) throw new Error(`Missing model for ${team.id}: ${modelPath}`);
  }
  if (args.teamPreviewModel && !fs.existsSync(args.teamPreviewModel)) {
    throw new Error(`Missing team preview model: ${args.teamPreviewModel}`);
  }
}

function compactResultArtifacts(result, deleteTraces) {
  const artifacts = [result.protocol_log_path, result.replay_html_path];
  if (deleteTraces) artifacts.push(result.trace_jsonl_path, result.summary_json_path);
  for (const relative of artifacts) {
    if (!relative) continue;
    const filePath = path.resolve(repoRoot, relative);
    assertInsideRepo(filePath, 'battle artifact');
    if (fs.existsSync(filePath)) fs.rmSync(filePath, {force: true});
  }
}

function ensureOutput(args, pool) {
  fs.mkdirSync(args.outDir, {recursive: true});
  const allDatasetPath = path.join(args.outDir, `${args.runId}_all_rollouts.jsonl`);
  const summaryPath = path.join(args.outDir, `${args.runId}_summary.json`);
  const perTeamPaths = Object.fromEntries(pool.teams.map(team => [
    team.id,
    path.join(args.outDir, `${args.runId}_${team.id}_rollouts.jsonl`),
  ]));

  if (args.overwrite) {
    removeIfExists(allDatasetPath, 'all rollout dataset');
    removeIfExists(summaryPath, 'rollout summary');
    for (const filePath of Object.values(perTeamPaths)) removeIfExists(filePath, 'team rollout dataset');
    removeIfExists(args.logDir, 'log directory');
  } else {
    const existing = [allDatasetPath, summaryPath, ...Object.values(perTeamPaths)].filter(filePath => fs.existsSync(filePath));
    if (existing.length) throw new Error(`Rollout outputs already exist; pass --overwrite. First: ${existing[0]}`);
  }
  fs.mkdirSync(args.logDir, {recursive: true});
  return {allDatasetPath, summaryPath, perTeamPaths};
}

function traceRows(tracePath) {
  return fs.readFileSync(tracePath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map(line => JSON.parse(line));
}

function appendRows(filePath, rows) {
  if (!rows.length) return;
  fs.mkdirSync(path.dirname(filePath), {recursive: true});
  fs.appendFileSync(filePath, rows.map(row => JSON.stringify(row)).join('\n') + '\n', 'utf8');
}

function rolloutRowsForSide({result, tracePath, side, opponentTeamId, runId}) {
  const rows = traceRows(tracePath).filter(row => row.agent === 'ppo_policy_agent' && row.side === side && !row.error_recovery);
  const finalWinnerSide = winnerSide(result.winner);
  return rows.map((row, index) => {
    const ppo = row.agent_diagnostics?.ppo_policy || {};
    const teamContext = row.agent_diagnostics?.team_context || null;
    const legalActions = row.legal_actions || [];
    const actionIndex = Number.isInteger(ppo.action_index) ? ppo.action_index : legalActions.indexOf(row.chosen_action);
    const done = index === rows.length - 1;
    const reward = done ? (finalWinnerSide === row.side ? 1 : -1) : 0;
    return {
      run_id: runId,
      battle_id: row.battle_id,
      trajectory_id: `${row.battle_id}:${row.side}`,
      step_index: index,
      turn: row.turn,
      side: row.side,
      opponent_id: opponentTeamId,
      agent_team_id: row.team,
      team: row.team,
      lead: row.lead,
      p1_team: result.p1.team_id,
      p2_team: result.p2.team_id,
      request_type: row.request?.teamPreview ? 'team_preview' :
        (Array.isArray(row.request?.forceSwitch) && row.request.forceSwitch.some(Boolean) ? 'force_switch' :
          (Array.isArray(row.request?.active) ? 'move' : 'other')),
      state: {
        request: row.request,
        public_state: row.public_state,
        team_context: teamContext,
        agent_diagnostics: row.agent_diagnostics || null,
      },
      legal_actions: legalActions,
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
  }).filter(row => (
    row.action_index >= 0 &&
    row.action_index < row.legal_actions.length &&
    Number.isFinite(row.log_prob) &&
    Number.isFinite(row.value_prediction)
  ));
}

function createLeagueAgent({args, pool, team}) {
  return createAgent('final_rl', {
    formatId: pool.format_id,
    modelPath: modelPathForTeam(args, team.id),
    teamPreviewModelPath: args.teamPreviewModel,
    pythonPath: args.pythonPath,
    torchDevice: args.torchDevice,
    epsilon: args.epsilon,
    topK: args.topK,
    sampleActions: false,
  });
}

function updateStanding(table, teamId, result) {
  if (!table.has(teamId)) table.set(teamId, {team_id: teamId, games: 0, wins: 0, losses: 0, unknown: 0, rollout_rows: 0});
  const row = table.get(teamId);
  row.games += 1;
  if (result === 'win') row.wins += 1;
  else if (result === 'loss') row.losses += 1;
  else row.unknown += 1;
}

function finalizeStandings(table) {
  return [...table.values()]
    .map(row => ({...row, win_rate: row.wins + row.losses ? row.wins / (row.wins + row.losses) : 0}))
    .sort((a, b) => b.win_rate - a.win_rate || b.wins - a.wins || a.team_id.localeCompare(b.team_id));
}

async function generate(args) {
  const pool = loadTeamPool();
  assertModelInputs(args, pool);
  const outputs = ensureOutput(args, pool);
  const standings = new Map();
  const games = [];
  let totalRows = 0;

  for (let game = 1; game <= args.games; game++) {
    const seed = `${args.seed}:game:${game}`;
    const rng = makeRng(seed);
    const p1Team = findTeam(pool, null, rng);
    let p2Team = findTeam(pool, null, rng);
    if (pool.teams.length > 1) {
      let guard = 0;
      while (p2Team.id === p1Team.id && guard++ < 20) p2Team = findTeam(pool, null, rng);
    }
    const result = await runBattle({
      pool,
      seed,
      p1Team,
      p2Team,
      p1Lead: findLeadMode(p1Team, null, rng),
      p2Lead: findLeadMode(p2Team, null, rng),
      p1Agent: createLeagueAgent({args, pool, team: p1Team}),
      p2Agent: createLeagueAgent({args, pool, team: p2Team}),
      logDir: args.logDir,
      rng,
      rolloutMaxDecisions: args.rolloutMaxDecisions,
    });
    const tracePath = path.resolve(repoRoot, result.trace_jsonl_path);
    const p1Rows = rolloutRowsForSide({result, tracePath, side: 'p1', opponentTeamId: p2Team.id, runId: args.runId});
    const p2Rows = rolloutRowsForSide({result, tracePath, side: 'p2', opponentTeamId: p1Team.id, runId: args.runId});
    appendRows(outputs.allDatasetPath, p1Rows.concat(p2Rows));
    appendRows(outputs.perTeamPaths[p1Team.id], p1Rows);
    appendRows(outputs.perTeamPaths[p2Team.id], p2Rows);
    totalRows += p1Rows.length + p2Rows.length;

    const side = winnerSide(result.winner);
    updateStanding(standings, p1Team.id, side === 'p1' ? 'win' : side === 'p2' ? 'loss' : 'unknown');
    updateStanding(standings, p2Team.id, side === 'p2' ? 'win' : side === 'p1' ? 'loss' : 'unknown');
    standings.get(p1Team.id).rollout_rows += p1Rows.length;
    standings.get(p2Team.id).rollout_rows += p2Rows.length;

    games.push({
      game,
      seed,
      winner: result.winner,
      winner_side: side,
      turns: result.turns,
      p1_team: p1Team.id,
      p2_team: p2Team.id,
      p1_rollout_rows: p1Rows.length,
      p2_rollout_rows: p2Rows.length,
      trace_jsonl_path: args.deleteBattleLogs ? null : result.trace_jsonl_path,
      summary_json_path: args.deleteBattleLogs ? null : result.summary_json_path,
    });
    if (args.compactLogs || args.deleteBattleLogs) compactResultArtifacts(result, args.deleteBattleLogs);
    if (game === 1 || game === args.games || (args.progressEvery > 0 && game % args.progressEvery === 0)) {
      console.log(
        `league rollout game=${game}/${args.games} ${p1Team.id} vs ${p2Team.id} ` +
        `winner=${result.winner} rows=${p1Rows.length + p2Rows.length}`
      );
    }
  }

  const summary = {
    created_at: new Date().toISOString(),
    run_id: args.runId,
    games_requested: args.games,
    games_completed: games.length,
    rows: totalRows,
    seed: args.seed,
    epsilon: args.epsilon,
    top_k: args.topK,
    rollout_max_decisions: args.rolloutMaxDecisions,
    models_dir: relativePath(args.modelsDir),
    team_preview_model: args.teamPreviewModel ? relativePath(args.teamPreviewModel) : null,
    dataset_path: relativePath(outputs.allDatasetPath),
    per_team_dataset_paths: Object.fromEntries(Object.entries(outputs.perTeamPaths).map(([team, filePath]) => [team, relativePath(filePath)])),
    log_dir: args.deleteBattleLogs ? null : relativePath(args.logDir),
    compact_logs: args.compactLogs,
    delete_battle_logs: args.deleteBattleLogs,
    standings: finalizeStandings(standings),
    games,
  };
  fs.writeFileSync(outputs.summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  console.log(`Wrote rollouts: ${relativePath(outputs.allDatasetPath)} rows=${totalRows}`);
  console.log(`Wrote summary: ${relativePath(outputs.summaryPath)}`);
}

generate(parseArgs(process.argv.slice(2)))
  .finally(() => closeTorchPolicyScorers())
  .catch(error => {
    console.error(`FAIL ${error.stack || error.message}`);
    process.exit(1);
  });
