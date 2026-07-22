const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.join(__dirname, '..');

function parseArgs(argv) {
  const args = {outDir: null, runPrefix: null};
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--out-dir') args.outDir = path.resolve(repoRoot, argv[++index]);
    else if (arg === '--run-prefix') args.runPrefix = argv[++index];
    else throw new Error(`Unexpected argument: ${arg}`);
  }
  if (!args.outDir) throw new Error('--out-dir is required');
  if (!args.runPrefix || /[\\/:*?"<>|]/.test(args.runPrefix)) {
    throw new Error('--run-prefix must be filename-safe');
  }
  return args;
}

function loadJson(filePath) {
  if (!fs.existsSync(filePath)) throw new Error(`Missing ablation summary: ${filePath}`);
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function totals(standings) {
  const result = standings.reduce((accumulator, row) => {
    for (const key of ['games', 'wins', 'losses', 'unknown']) accumulator[key] += row[key];
    return accumulator;
  }, {games: 0, wins: 0, losses: 0, unknown: 0});
  result.win_rate = result.wins + result.losses ? result.wins / (result.wins + result.losses) : 0;
  result.wilson_95 = wilsonInterval(result.wins, result.wins + result.losses);
  return result;
}

function wilsonInterval(wins, games) {
  if (!games) return [0, 0];
  const z = 1.959963984540054;
  const rate = wins / games;
  const denominator = 1 + z * z / games;
  const center = (rate + z * z / (2 * games)) / denominator;
  const margin = z * Math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games)) / denominator;
  return [Math.max(0, center - margin), Math.min(1, center + margin)];
}

function gameOutcomes(summary) {
  const outcomes = new Map();
  for (const matchup of summary.matchups || []) {
    for (const game of matchup.games || []) {
      const key = [matchup.agent_team, matchup.opponent_team, game.game, game.rl_side].join('|');
      const won = game.winner_side === game.rl_side;
      outcomes.set(key, game.winner_side === 'unknown' ? null : won);
    }
  }
  return outcomes;
}

function exactTwoSidedBinomial(successes, trials) {
  if (!trials) return 1;
  const tail = Math.min(successes, trials - successes);
  let probability = Math.pow(0.5, trials);
  let cumulative = probability;
  for (let index = 1; index <= tail; index++) {
    probability *= (trials - index + 1) / index;
    cumulative += probability;
  }
  return Math.min(1, 2 * cumulative);
}

function pairedComparison(left, right) {
  const leftGames = gameOutcomes(left);
  const rightGames = gameOutcomes(right);
  let leftOnlyWins = 0;
  let rightOnlyWins = 0;
  let same = 0;
  let unknown = 0;
  for (const [key, leftWon] of leftGames) {
    if (!rightGames.has(key)) continue;
    const rightWon = rightGames.get(key);
    if (leftWon == null || rightWon == null) unknown += 1;
    else if (leftWon === rightWon) same += 1;
    else if (leftWon) leftOnlyWins += 1;
    else rightOnlyWins += 1;
  }
  const discordant = leftOnlyWins + rightOnlyWins;
  return {
    paired_games: leftOnlyWins + rightOnlyWins + same + unknown,
    left_only_wins: leftOnlyWins,
    right_only_wins: rightOnlyWins,
    same_outcome: same,
    unknown,
    mcnemar_exact_p: exactTwoSidedBinomial(leftOnlyWins, discordant),
  };
}

function summarize(args) {
  const modeFiles = {
    learned: path.join(args.outDir, `${args.runPrefix}_learned_summary.json`),
    random: path.join(args.outDir, `${args.runPrefix}_random_summary.json`),
    battle_model: path.join(args.outDir, `${args.runPrefix}_battle_model_summary.json`),
  };
  const summaries = Object.fromEntries(Object.entries(modeFiles).map(([mode, filePath]) => [mode, loadJson(filePath)]));
  const modes = Object.fromEntries(Object.entries(summaries).map(([mode, summary]) => [mode, {
    totals: totals(summary.standings || []),
    side_summary: summary.side_summary,
    standings: summary.standings,
    source_summary: path.relative(repoRoot, modeFiles[mode]).replace(/\\/g, '/'),
  }]));

  const report = {
    created_at: new Date().toISOString(),
    run_prefix: args.runPrefix,
    modes,
    learned_minus_random_win_rate: modes.learned.totals.win_rate - modes.random.totals.win_rate,
    learned_minus_battle_model_win_rate: modes.learned.totals.win_rate - modes.battle_model.totals.win_rate,
    paired_learned_vs_random: pairedComparison(summaries.learned, summaries.random),
    paired_learned_vs_battle_model: pairedComparison(summaries.learned, summaries.battle_model),
  };
  const outputPath = path.join(args.outDir, `${args.runPrefix}_report.json`);
  fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  console.log(`Wrote preview ablation report: ${path.relative(repoRoot, outputPath).replace(/\\/g, '/')}`);
}

try {
  summarize(parseArgs(process.argv.slice(2)));
} catch (error) {
  console.error(`FAIL ${error.stack || error.message}`);
  process.exit(1);
}
