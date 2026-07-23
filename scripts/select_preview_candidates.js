const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.join(__dirname, '..');

function parseArgs(argv) {
  const args = {candidates: [], baselines: [], out: null, selection: null, fallback: null};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--candidate') args.candidates.push(argv[++i]);
    else if (arg === '--baseline') args.baselines.push(argv[++i]);
    else if (arg === '--fallback') args.fallback = argv[++i];
    else if (arg === '--out') args.out = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--selection') args.selection = path.resolve(repoRoot, argv[++i]);
    else throw new Error(`Unexpected argument: ${arg}`);
  }
  if (args.candidates.length < 2) throw new Error('Pass at least two --candidate values');
  if (!args.out || !args.selection) throw new Error('--out and --selection are required');
  return args;
}

function exactTwoSidedBinomial(left, right) {
  const trials = left + right;
  if (!trials) return 1;
  const tail = Math.min(left, right);
  let probability = Math.pow(0.5, trials);
  let cumulative = probability;
  for (let successes = 1; successes <= tail; successes++) {
    probability *= (trials - successes + 1) / successes;
    cumulative += probability;
  }
  return Math.min(1, 2 * cumulative);
}

function parseEntry(value, needsCheckpoint) {
  const equals = value.indexOf('=');
  const comma = value.lastIndexOf(',');
  if (equals <= 0 || (needsCheckpoint && comma <= equals + 1)) {
    throw new Error(`Invalid entry: ${value}`);
  }
  return {
    id: value.slice(0, equals),
    summaryPath: path.resolve(repoRoot, value.slice(equals + 1, needsCheckpoint ? comma : undefined)),
    checkpoint: needsCheckpoint ? value.slice(comma + 1) : null,
    eligible: needsCheckpoint,
  };
}

function wilson(wins, games, z = 1.959963984540054) {
  if (!games) return {low: 0, high: 1};
  const p = wins / games;
  const z2 = z * z;
  const denominator = 1 + z2 / games;
  const center = (p + z2 / (2 * games)) / denominator;
  const half = z * Math.sqrt((p * (1 - p) + z2 / (4 * games)) / games) / denominator;
  return {low: center - half, high: center + half};
}

function outcomes(summary) {
  const result = new Map();
  for (const matchup of summary.matchups || []) {
    for (const game of matchup.games || []) {
      const key = [matchup.agent_team, matchup.opponent_team, game.game, game.rl_side].join('|');
      result.set(key, game.winner_side === game.rl_side ? 'win' :
        (game.winner_side === 'unknown' ? 'unknown' : 'loss'));
    }
  }
  return result;
}

function paired(candidate, reference) {
  let candidateOnly = 0;
  let referenceOnly = 0;
  let same = 0;
  for (const [key, value] of candidate.outcomes) {
    const other = reference.outcomes.get(key);
    if (!other || value === 'unknown' || other === 'unknown') continue;
    if (value === other) same += 1;
    else if (value === 'win') candidateOnly += 1;
    else referenceOnly += 1;
  }
  return {
    same,
    candidate_only_wins: candidateOnly,
    reference_only_wins: referenceOnly,
    net_paired_wins: candidateOnly - referenceOnly,
    discordant_games: candidateOnly + referenceOnly,
    exact_mcnemar_p: exactTwoSidedBinomial(candidateOnly, referenceOnly),
  };
}

function load(entry) {
  if (!fs.existsSync(entry.summaryPath)) throw new Error(`Missing summary: ${entry.summaryPath}`);
  const summary = JSON.parse(fs.readFileSync(entry.summaryPath, 'utf8'));
  const result = outcomes(summary);
  const wins = [...result.values()].filter(value => value === 'win').length;
  const losses = [...result.values()].filter(value => value === 'loss').length;
  return {...entry, summary, outcomes: result, wins, losses};
}

function run(args) {
  const candidates = args.candidates.map(value => load(parseEntry(value, true)));
  const baselines = args.baselines.map(value => load(parseEntry(value, false)));
  const keys = [...candidates[0].outcomes.keys()].sort();
  for (const entry of [...candidates.slice(1), ...baselines]) {
    const other = [...entry.outcomes.keys()].sort();
    if (other.length !== keys.length || other.some((key, index) => key !== keys[index])) {
      throw new Error(`${entry.id} does not contain the same paired games`);
    }
  }
  const rankedCandidates = [...candidates].sort((a, b) => b.wins - a.wins || b.id.localeCompare(a.id));
  let selected = rankedCandidates[0];
  let fallback = null;
  let promotion = null;
  if (args.fallback) {
    fallback = candidates.find(candidate => candidate.id === args.fallback);
    if (!fallback) throw new Error(`Unknown --fallback candidate: ${args.fallback}`);
    const challenger = rankedCandidates.find(candidate => candidate.id !== fallback.id) || fallback;
    const comparison = paired(challenger, fallback);
    const promoted = challenger.wins > fallback.wins && comparison.exact_mcnemar_p < 0.05;
    selected = promoted ? challenger : fallback;
    promotion = {
      challenger: challenger.id,
      fallback: fallback.id,
      promoted,
      rule: 'strictly_more_wins_and_paired_exact_mcnemar_p_below_0.05',
      comparison,
    };
  }
  const rows = [...candidates, ...baselines].map(entry => ({
    id: entry.id,
    eligible: entry.eligible,
    summary: path.relative(repoRoot, entry.summaryPath).replace(/\\/g, '/'),
    games: entry.wins + entry.losses,
    wins: entry.wins,
    losses: entry.losses,
    win_rate: entry.wins / Math.max(1, entry.wins + entry.losses),
    wilson_95: wilson(entry.wins, entry.wins + entry.losses),
    paired_vs_selected: paired(entry, selected),
  })).sort((a, b) => b.win_rate - a.win_rate || b.wins - a.wins || a.id.localeCompare(b.id));
  const report = {
    created_at: new Date().toISOString(),
    method: 'common_seed_universal_preview_selection',
    selected: selected.id,
    selected_checkpoint: selected.checkpoint,
    fallback: fallback?.id || null,
    promotion,
    results: rows,
  };
  const selection = {
    created_at: report.created_at,
    candidate: selected.id,
    checkpoint: selected.checkpoint,
    selection_wins: selected.wins,
    selection_games: selected.wins + selected.losses,
    report: path.relative(repoRoot, args.out).replace(/\\/g, '/'),
  };
  fs.mkdirSync(path.dirname(args.out), {recursive: true});
  fs.mkdirSync(path.dirname(args.selection), {recursive: true});
  fs.writeFileSync(args.out, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  fs.writeFileSync(args.selection, `${JSON.stringify(selection, null, 2)}\n`, 'utf8');
  console.log(`Selected universal preview: ${selected.id} (${selected.wins}/${selected.wins + selected.losses})`);
}

try {
  run(parseArgs(process.argv.slice(2)));
} catch (error) {
  console.error(`FAIL ${error.stack || error.message}`);
  process.exit(1);
}
