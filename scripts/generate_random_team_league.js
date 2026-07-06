const fs = require('node:fs');
const path = require('node:path');
const {createAgent} = require('../src/agents');
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

function parseArgs(argv) {
  const args = {
    runId: 'mb_random_bootstrap',
    games: 10000,
    outDir: path.join(repoRoot, 'experiments', 'mb_alpha_league'),
    logDir: null,
    seed: null,
    overwrite: false,
    compactLogs: false,
    progressEvery: 100,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--run-id') {
      args.runId = argv[++i];
    } else if (arg === '--games') {
      args.games = parseInteger(argv[++i], '--games');
    } else if (arg === '--out-dir') {
      args.outDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--log-dir') {
      args.logDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--seed') {
      args.seed = argv[++i];
    } else if (arg === '--progress-every') {
      args.progressEvery = parseInteger(argv[++i], '--progress-every');
    } else if (arg === '--compact-logs') {
      args.compactLogs = true;
    } else if (arg === '--overwrite') {
      args.overwrite = true;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (!args.runId || /[\\/:*?"<>|]/.test(args.runId)) {
    throw new Error('--run-id must be filename-safe');
  }
  if (args.games <= 0) throw new Error('--games must be > 0');
  if (!args.seed) args.seed = args.runId;
  if (!args.logDir) args.logDir = path.join(repoRoot, 'logs', 'battles', `${args.runId}_random_bootstrap`);
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

function compactResultArtifacts(result) {
  for (const relative of [result.protocol_log_path, result.replay_html_path]) {
    if (!relative) continue;
    const filePath = path.resolve(repoRoot, relative);
    assertInsideRepo(filePath, 'battle artifact');
    if (fs.existsSync(filePath)) fs.rmSync(filePath, {force: true});
  }
}

function ensureOutput(args) {
  fs.mkdirSync(args.outDir, {recursive: true});
  const manifestPath = path.join(args.outDir, `${args.runId}_manifest.jsonl`);
  const summaryPath = path.join(args.outDir, `${args.runId}_summary.json`);
  if (args.overwrite) {
    removeIfExists(manifestPath, 'manifest');
    removeIfExists(summaryPath, 'summary');
    removeIfExists(args.logDir, 'log directory');
  } else if (fs.existsSync(manifestPath) || fs.existsSync(summaryPath)) {
    throw new Error(`Outputs already exist for ${args.runId}; pass --overwrite`);
  }
  fs.mkdirSync(args.logDir, {recursive: true});
  return {manifestPath, summaryPath};
}

async function generate(args) {
  const outputs = ensureOutput(args);
  const pool = loadTeamPool();
  const standings = new Map(pool.teams.map(team => [team.id, {team_id: team.id, games: 0, wins: 0, losses: 0, unknown: 0}]));

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
      p1Agent: createAgent('random', {formatId: pool.format_id}),
      p2Agent: createAgent('random', {formatId: pool.format_id}),
      logDir: args.logDir,
      rng,
    });
    if (args.compactLogs) compactResultArtifacts(result);
    const side = winnerSide(result.winner);
    const p1Row = standings.get(p1Team.id);
    const p2Row = standings.get(p2Team.id);
    p1Row.games += 1;
    p2Row.games += 1;
    if (side === 'p1') {
      p1Row.wins += 1;
      p2Row.losses += 1;
    } else if (side === 'p2') {
      p2Row.wins += 1;
      p1Row.losses += 1;
    } else {
      p1Row.unknown += 1;
      p2Row.unknown += 1;
    }

    fs.appendFileSync(outputs.manifestPath, `${JSON.stringify({
      game,
      seed,
      winner: result.winner,
      winner_side: side,
      turns: result.turns,
      p1_team: result.p1.team_id,
      p2_team: result.p2.team_id,
      p1_lead: result.p1.lead_id,
      p2_lead: result.p2.lead_id,
      trace_jsonl_path: result.trace_jsonl_path,
      summary_json_path: result.summary_json_path,
    })}\n`, 'utf8');

    if (game === 1 || game === args.games || (args.progressEvery > 0 && game % args.progressEvery === 0)) {
      console.log(`random bootstrap game=${game}/${args.games} winner=${result.winner} turns=${result.turns}`);
    }
  }

  const summary = {
    created_at: new Date().toISOString(),
    run_id: args.runId,
    games: args.games,
    seed: args.seed,
    format_id: pool.format_id,
    team_pool: relativePath(path.join(repoRoot, 'data', 'teams', 'team_pool.json')),
    log_dir: relativePath(args.logDir),
    manifest_path: relativePath(outputs.manifestPath),
    compact_logs: args.compactLogs,
    standings: [...standings.values()].map(row => ({
      ...row,
      win_rate: row.wins + row.losses ? row.wins / (row.wins + row.losses) : 0,
    })).sort((a, b) => b.win_rate - a.win_rate || b.wins - a.wins || a.team_id.localeCompare(b.team_id)),
  };
  fs.writeFileSync(outputs.summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  console.log(`Wrote summary: ${relativePath(outputs.summaryPath)}`);
}

generate(parseArgs(process.argv.slice(2))).catch(error => {
  console.error(`FAIL ${error.stack || error.message}`);
  process.exit(1);
});
