const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.join(__dirname, '..');

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    runId: 'mb_alphastar_league',
    iterations: 10,
    startIteration: 1,
    outDir: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--run-id') {
      args.runId = argv[++i];
    } else if (arg === '--iterations') {
      args.iterations = parseInteger(argv[++i], '--iterations');
    } else if (arg === '--start-iteration') {
      args.startIteration = parseInteger(argv[++i], '--start-iteration');
    } else if (arg === '--out-dir') {
      args.outDir = path.resolve(repoRoot, argv[++i]);
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }
  if (!args.outDir) args.outDir = path.join(repoRoot, 'experiments', 'mb_alpha_league', args.runId);
  return args;
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function readJsonIfExists(filePath) {
  if (!fs.existsSync(filePath)) return null;
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function iterationTag(iteration) {
  return `iter_${String(iteration).padStart(3, '0')}`;
}

function compactStandings(summary) {
  return (summary?.standings || []).map(row => ({
    team_id: row.team_id,
    games: row.games,
    wins: row.wins,
    losses: row.losses,
    unknown: row.unknown,
    win_rate: row.win_rate,
    rollout_rows: row.rollout_rows,
    p1: row.p1,
    p2: row.p2,
  }));
}

function collect(args) {
  const bootstrapSummaryPath = path.join(args.outDir, 'bootstrap', `${args.runId}_bootstrap_random_summary.json`);
  const bootstrapBcSummaryPath = path.join(repoRoot, 'data', 'datasets', 'bc', `${args.runId}_bootstrap_bc.summary.json`);
  const bootstrapPolicyMetricsPath = path.join(repoRoot, 'models', 'torch', args.runId, 'bootstrap', 'policy', 'metrics.json');

  const iterations = [];
  const finalIteration = args.startIteration + args.iterations - 1;
  for (let iteration = args.startIteration; iteration <= finalIteration; iteration++) {
    const tag = iterationTag(iteration);
    const iterationId = `${args.runId}_${tag}`;
    const rolloutSummaryPath = path.join(repoRoot, 'data', 'datasets', 'rl', `${iterationId}_summary.json`);
    const evalSummaryPath = path.join(args.outDir, 'eval', `${iterationId}_eval_summary.json`);
    const universalMetricsPath = path.join(repoRoot, 'models', 'torch', args.runId, tag, 'universal_preview', 'metrics.json');
    const rolloutSummary = readJsonIfExists(rolloutSummaryPath);
    const evalSummary = readJsonIfExists(evalSummaryPath);
    iterations.push({
      iteration,
      tag,
      rollout_summary_path: fs.existsSync(rolloutSummaryPath) ? relativePath(rolloutSummaryPath) : null,
      eval_summary_path: fs.existsSync(evalSummaryPath) ? relativePath(evalSummaryPath) : null,
      universal_preview_metrics_path: fs.existsSync(universalMetricsPath) ? relativePath(universalMetricsPath) : null,
      rollout_rows: rolloutSummary?.rows ?? null,
      rollout_games: rolloutSummary?.games_completed ?? null,
      battle_action_policy: rolloutSummary?.battle_action_policy ?? null,
      team_preview_policy: rolloutSummary?.team_preview_policy ?? null,
      rollout_standings: compactStandings(rolloutSummary),
      eval_preview_mode: evalSummary?.preview_mode ?? null,
      eval_side_swaps: evalSummary?.side_swaps ?? null,
      eval_side_summary: evalSummary?.side_summary ?? null,
      eval_standings: compactStandings(evalSummary),
    });
  }

  const report = {
    created_at: new Date().toISOString(),
    run_id: args.runId,
    start_iteration: args.startIteration,
    iterations_requested: args.iterations,
    team_pool_path: 'data/teams/team_pool.json',
    bootstrap: {
      random_summary_path: fs.existsSync(bootstrapSummaryPath) ? relativePath(bootstrapSummaryPath) : null,
      bc_summary_path: fs.existsSync(bootstrapBcSummaryPath) ? relativePath(bootstrapBcSummaryPath) : null,
      policy_metrics_path: fs.existsSync(bootstrapPolicyMetricsPath) ? relativePath(bootstrapPolicyMetricsPath) : null,
      random_summary: readJsonIfExists(bootstrapSummaryPath),
      bc_summary: readJsonIfExists(bootstrapBcSummaryPath),
    },
    iterations,
  };
  fs.mkdirSync(args.outDir, {recursive: true});
  const reportPath = path.join(args.outDir, 'report.json');
  fs.writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  console.log(`Wrote report: ${relativePath(reportPath)}`);
}

try {
  collect(parseArgs(process.argv.slice(2)));
} catch (error) {
  console.error(`FAIL ${error.stack || error.message}`);
  process.exit(1);
}
