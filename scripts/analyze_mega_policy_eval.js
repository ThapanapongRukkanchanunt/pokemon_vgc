const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.join(__dirname, '..');

function parseModePath(value) {
  const equals = value.indexOf('=');
  if (equals <= 0) throw new Error(`Expected MODE=PATH, got ${value}`);
  return {
    mode: value.slice(0, equals),
    filePath: path.resolve(repoRoot, value.slice(equals + 1)),
  };
}

function parseArgs(argv) {
  const args = {model: [], guarded: [], out: null};
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--model') args.model.push(parseModePath(argv[++index]));
    else if (arg === '--guarded') args.guarded.push(parseModePath(argv[++index]));
    else if (arg === '--out') args.out = path.resolve(repoRoot, argv[++index]);
    else throw new Error(`Unexpected argument: ${arg}`);
  }
  if (!args.model.length || args.model.length !== args.guarded.length) {
    throw new Error('Pass matching --model and --guarded MODE=PATH summaries');
  }
  if (!args.out) throw new Error('--out is required');
  return args;
}

function readJson(filePath) {
  if (!fs.existsSync(filePath)) throw new Error(`Missing JSON: ${filePath}`);
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function totals(summary) {
  const result = summary.standings.reduce((accumulator, row) => {
    accumulator.wins += row.wins;
    accumulator.losses += row.losses;
    accumulator.unknown += row.unknown;
    return accumulator;
  }, {wins: 0, losses: 0, unknown: 0});
  result.games = result.wins + result.losses;
  result.win_rate = result.games ? result.wins / result.games : 0;
  result.mega_policy_stats = summary.mega_policy_stats || null;
  return result;
}

function outcomes(summary) {
  const result = new Map();
  for (const matchup of summary.matchups || []) {
    for (const game of matchup.games || []) {
      const key = [matchup.agent_team, matchup.opponent_team, game.game, game.rl_side].join('|');
      const outcome = game.winner_side === game.rl_side ? 'win' :
        (game.winner_side === 'unknown' ? 'unknown' : 'loss');
      result.set(key, {outcome, seed: game.seed});
    }
  }
  return result;
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

function pairedDifference(guardedSummary, modelSummary) {
  const guarded = outcomes(guardedSummary);
  const model = outcomes(modelSummary);
  let guardedOnly = 0;
  let modelOnly = 0;
  let same = 0;
  let pairedGames = 0;
  for (const [key, guardedGame] of guarded) {
    const modelGame = model.get(key);
    if (!modelGame) continue;
    if (guardedGame.seed !== modelGame.seed) {
      throw new Error(`Seed mismatch for paired game ${key}`);
    }
    if (guardedGame.outcome === 'unknown' || modelGame.outcome === 'unknown') continue;
    pairedGames += 1;
    if (guardedGame.outcome === modelGame.outcome) same += 1;
    else if (guardedGame.outcome === 'win') guardedOnly += 1;
    else modelOnly += 1;
  }
  return {
    paired_games: pairedGames,
    same,
    guarded_only_wins: guardedOnly,
    model_only_wins: modelOnly,
    net_paired_wins: guardedOnly - modelOnly,
    discordant_games: guardedOnly + modelOnly,
    exact_mcnemar_p: exactTwoSidedBinomial(guardedOnly, modelOnly),
  };
}

function addTotals(target, source) {
  for (const key of ['games', 'wins', 'losses', 'unknown']) target[key] += source[key];
}

function addPaired(target, source) {
  for (const key of [
    'paired_games',
    'same',
    'guarded_only_wins',
    'model_only_wins',
    'net_paired_wins',
    'discordant_games',
  ]) {
    target[key] += source[key];
  }
}

function run(args) {
  const modelByMode = new Map(args.model.map(entry => [entry.mode, {
    path: entry.filePath,
    summary: readJson(entry.filePath),
  }]));
  const guardedByMode = new Map(args.guarded.map(entry => [entry.mode, {
    path: entry.filePath,
    summary: readJson(entry.filePath),
  }]));
  const modes = [...modelByMode.keys()].sort();
  if (modes.some(mode => !guardedByMode.has(mode))) {
    throw new Error('Model and guarded modes do not match');
  }

  const aggregate = {
    model: {games: 0, wins: 0, losses: 0, unknown: 0},
    guarded: {games: 0, wins: 0, losses: 0, unknown: 0},
    paired: {
      paired_games: 0,
      same: 0,
      guarded_only_wins: 0,
      model_only_wins: 0,
      net_paired_wins: 0,
      discordant_games: 0,
    },
  };
  const results = {};

  for (const mode of modes) {
    const modelEntry = modelByMode.get(mode);
    const guardedEntry = guardedByMode.get(mode);
    if (modelEntry.summary.seed !== guardedEntry.summary.seed) {
      throw new Error(`Seed mismatch for ${mode}`);
    }
    const model = totals(modelEntry.summary);
    const guarded = totals(guardedEntry.summary);
    const paired = pairedDifference(guardedEntry.summary, modelEntry.summary);
    results[mode] = {
      seed: modelEntry.summary.seed,
      model_path: path.relative(repoRoot, modelEntry.path).replace(/\\/g, '/'),
      guarded_path: path.relative(repoRoot, guardedEntry.path).replace(/\\/g, '/'),
      model,
      guarded,
      win_rate_delta: guarded.win_rate - model.win_rate,
      paired,
    };
    addTotals(aggregate.model, model);
    addTotals(aggregate.guarded, guarded);
    addPaired(aggregate.paired, paired);
  }

  aggregate.model.win_rate = aggregate.model.games ?
    aggregate.model.wins / aggregate.model.games :
    0;
  aggregate.guarded.win_rate = aggregate.guarded.games ?
    aggregate.guarded.wins / aggregate.guarded.games :
    0;
  aggregate.win_rate_delta = aggregate.guarded.win_rate - aggregate.model.win_rate;
  aggregate.paired.exact_mcnemar_p = exactTwoSidedBinomial(
    aggregate.paired.guarded_only_wins,
    aggregate.paired.model_only_wins
  );

  const report = {
    created_at: new Date().toISOString(),
    method: 'common_seed_sole_usable_mega_policy_ablation',
    modes,
    results,
    aggregate,
  };
  fs.mkdirSync(path.dirname(args.out), {recursive: true});
  fs.writeFileSync(args.out, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
  console.log(
    `Mega guard: ${aggregate.guarded.wins}/${aggregate.guarded.games} ` +
    `(${(100 * aggregate.guarded.win_rate).toFixed(2)}%)`
  );
  console.log(
    `Model only: ${aggregate.model.wins}/${aggregate.model.games} ` +
    `(${(100 * aggregate.model.win_rate).toFixed(2)}%)`
  );
  console.log(
    `Paired net=${aggregate.paired.net_paired_wins} ` +
    `p=${aggregate.paired.exact_mcnemar_p.toFixed(6)}`
  );
  console.log(`Wrote report: ${path.relative(repoRoot, args.out).replace(/\\/g, '/')}`);
}

try {
  run(parseArgs(process.argv.slice(2)));
} catch (error) {
  console.error(`FAIL ${error.stack || error.message}`);
  process.exit(1);
}
