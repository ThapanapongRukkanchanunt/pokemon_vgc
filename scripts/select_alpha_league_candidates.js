const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.join(__dirname, '..');

function parseArgs(argv) {
  const args = {summaries: [], out: null, manifest: null, reference: null};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--summary') args.summaries.push(path.resolve(repoRoot, argv[++i]));
    else if (arg === '--out') args.out = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--manifest') args.manifest = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--reference') args.reference = argv[++i];
    else throw new Error(`Unexpected argument: ${arg}`);
  }
  if (args.summaries.length < 2) throw new Error('Pass at least two --summary files');
  if (!args.out) throw new Error('--out is required');
  if (!args.manifest) throw new Error('--manifest is required');
  return args;
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
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

function logGamma(value) {
  const coefficients = [
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7,
  ];
  if (value < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
  let x = 0.9999999999998099;
  let shifted = value - 1;
  for (let i = 0; i < coefficients.length; i++) x += coefficients[i] / (shifted + i + 1);
  const t = shifted + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t) - t + Math.log(x);
}

function binomialProbability(k, n) {
  const logCombination = logGamma(n + 1) - logGamma(k + 1) - logGamma(n - k + 1);
  return Math.exp(logCombination - n * Math.log(2));
}

function exactTwoSidedBinomial(successes, trials) {
  if (!trials) return 1;
  const observed = binomialProbability(successes, trials);
  let total = 0;
  for (let k = 0; k <= trials; k++) {
    const probability = binomialProbability(k, trials);
    if (probability <= observed + 1e-15) total += probability;
  }
  return Math.min(1, total);
}

function candidateId(summary, filePath) {
  const match = String(summary.models_dir || '').match(/(iter_\d+)/);
  return match ? match[1] : path.basename(filePath, path.extname(filePath));
}

function gameOutcomes(summary) {
  const outcomes = new Map();
  for (const matchup of summary.matchups || []) {
    for (const game of matchup.games || []) {
      const key = [matchup.agent_team, matchup.opponent_team, game.game, game.rl_side].join('|');
      const outcome = game.winner_side === 'unknown' ? 'unknown' :
        (game.winner_side === game.rl_side ? 'win' : 'loss');
      outcomes.set(key, outcome);
    }
  }
  return outcomes;
}

function pairedComparison(candidate, reference) {
  let bothWin = 0;
  let bothLoss = 0;
  let candidateOnly = 0;
  let referenceOnly = 0;
  let excluded = 0;
  for (const [key, candidateOutcome] of candidate.outcomes) {
    const referenceOutcome = reference.outcomes.get(key);
    if (!referenceOutcome || candidateOutcome === 'unknown' || referenceOutcome === 'unknown') {
      excluded += 1;
    } else if (candidateOutcome === 'win' && referenceOutcome === 'win') {
      bothWin += 1;
    } else if (candidateOutcome === 'loss' && referenceOutcome === 'loss') {
      bothLoss += 1;
    } else if (candidateOutcome === 'win') {
      candidateOnly += 1;
    } else {
      referenceOnly += 1;
    }
  }
  return {
    both_win: bothWin,
    both_loss: bothLoss,
    candidate_only_wins: candidateOnly,
    reference_only_wins: referenceOnly,
    excluded,
    discordant: candidateOnly + referenceOnly,
    mcnemar_exact_p: exactTwoSidedBinomial(candidateOnly, candidateOnly + referenceOnly),
  };
}

function checkpointFor(summary, teamId) {
  if (summary.model_manifest) {
    const manifestPath = path.resolve(repoRoot, summary.model_manifest);
    const parsed = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const entry = (parsed.models || parsed)[teamId];
    return typeof entry === 'string' ? entry : entry.checkpoint;
  }
  return `${summary.models_dir}/${teamId}/checkpoint.pt`;
}

function run(args) {
  const candidates = args.summaries.map(filePath => {
    const summary = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    return {id: candidateId(summary, filePath), filePath, summary, outcomes: gameOutcomes(summary)};
  });
  const keys = [...candidates[0].outcomes.keys()].sort();
  for (const candidate of candidates.slice(1)) {
    const candidateKeys = [...candidate.outcomes.keys()].sort();
    if (candidateKeys.length !== keys.length || candidateKeys.some((key, index) => key !== keys[index])) {
      throw new Error(`Candidate ${candidate.id} does not contain the same paired games`);
    }
  }
  const reference = args.reference ? candidates.find(candidate => candidate.id === args.reference) : candidates.at(-1);
  if (!reference) throw new Error(`Unknown --reference candidate: ${args.reference}`);

  const teamIds = [...new Set(candidates.flatMap(candidate => candidate.summary.standings.map(row => row.team_id)))].sort();
  const candidateRows = candidates.map(candidate => {
    const wins = [...candidate.outcomes.values()].filter(outcome => outcome === 'win').length;
    const losses = [...candidate.outcomes.values()].filter(outcome => outcome === 'loss').length;
    return {
      id: candidate.id,
      summary: relativePath(candidate.filePath),
      games: wins + losses,
      wins,
      losses,
      win_rate: wins / Math.max(1, wins + losses),
      wilson_95: wilson(wins, wins + losses),
      paired_vs_reference: pairedComparison(candidate, reference),
    };
  }).sort((a, b) => b.win_rate - a.win_rate || b.wins - a.wins || b.id.localeCompare(a.id));

  const selected = {};
  const perTeam = {};
  for (const teamId of teamIds) {
    const rows = candidates.map(candidate => {
      const standing = candidate.summary.standings.find(row => row.team_id === teamId);
      const games = standing.wins + standing.losses;
      return {
        candidate: candidate.id,
        games,
        wins: standing.wins,
        losses: standing.losses,
        win_rate: standing.win_rate,
        wilson_95: wilson(standing.wins, games),
      };
    }).sort((a, b) => b.wins - a.wins || b.win_rate - a.win_rate || b.candidate.localeCompare(a.candidate));
    const winner = rows[0];
    const source = candidates.find(candidate => candidate.id === winner.candidate);
    selected[teamId] = {
      checkpoint: checkpointFor(source.summary, teamId),
      candidate: winner.candidate,
      selection_wins: winner.wins,
      selection_games: winner.games,
    };
    perTeam[teamId] = rows;
  }

  const report = {
    created_at: new Date().toISOString(),
    method: 'common_seed_per_team_checkpoint_selection',
    reference: reference.id,
    candidates: candidateRows,
    per_team: perTeam,
    selected,
  };
  const manifest = {
    created_at: report.created_at,
    selection_report: relativePath(args.out),
    models: selected,
  };
  fs.mkdirSync(path.dirname(args.out), {recursive: true});
  fs.mkdirSync(path.dirname(args.manifest), {recursive: true});
  fs.writeFileSync(args.out, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  fs.writeFileSync(args.manifest, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
  console.log(`Selected checkpoints: ${relativePath(args.manifest)}`);
  for (const [teamId, entry] of Object.entries(selected)) {
    console.log(`${teamId}: ${entry.candidate} (${entry.selection_wins}/${entry.selection_games})`);
  }
}

try {
  run(parseArgs(process.argv.slice(2)));
} catch (error) {
  console.error(`FAIL ${error.stack || error.message}`);
  process.exit(1);
}
