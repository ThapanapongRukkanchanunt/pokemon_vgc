const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.join(__dirname, '..');

function parseModePath(value) {
  const equals = value.indexOf('=');
  if (equals <= 0) throw new Error(`Expected MODE=PATH, got ${value}`);
  return {mode: value.slice(0, equals), filePath: path.resolve(repoRoot, value.slice(equals + 1))};
}

function parseArgs(argv) {
  const args = {
    selected: [],
    baseline: [],
    selectedManifest: null,
    baselineModelsDir: null,
    validatedManifest: null,
    out: null,
    finalSelection: null,
    headToHead: null,
    searchTop1: null,
    search: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--selected') args.selected.push(parseModePath(argv[++i]));
    else if (arg === '--baseline') args.baseline.push(parseModePath(argv[++i]));
    else if (arg === '--selected-manifest') args.selectedManifest = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--baseline-models-dir') args.baselineModelsDir = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--validated-manifest') args.validatedManifest = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--out') args.out = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--final-selection') args.finalSelection = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--head-to-head') args.headToHead = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--search-top1') args.searchTop1 = path.resolve(repoRoot, argv[++i]);
    else if (arg === '--search') args.search = path.resolve(repoRoot, argv[++i]);
    else throw new Error(`Unexpected argument: ${arg}`);
  }
  if (!args.selected.length || args.selected.length !== args.baseline.length) {
    throw new Error('Pass matching --selected and --baseline summaries');
  }
  if (!args.selectedManifest || !args.baselineModelsDir) {
    throw new Error('--selected-manifest and --baseline-models-dir are required');
  }
  if (!args.validatedManifest || !args.out || !args.finalSelection) {
    throw new Error('--validated-manifest, --out, and --final-selection are required');
  }
  return args;
}

function readJson(filePath) {
  if (!fs.existsSync(filePath)) throw new Error(`Missing JSON: ${filePath}`);
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
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

function rowFor(summary, teamId) {
  const row = summary.standings.find(entry => entry.team_id === teamId);
  if (!row) throw new Error(`Summary ${summary.run_id} has no standing for ${teamId}`);
  return row;
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

function pairedDifference(selectedSummary, baselineSummary) {
  const selected = outcomes(selectedSummary);
  const baseline = outcomes(baselineSummary);
  let selectedOnly = 0;
  let baselineOnly = 0;
  let same = 0;
  for (const [key, value] of selected) {
    const other = baseline.get(key);
    if (!other || value === 'unknown' || other === 'unknown') continue;
    if (value === other) same += 1;
    else if (value === 'win') selectedOnly += 1;
    else baselineOnly += 1;
  }
  return {
    same,
    selected_only_wins: selectedOnly,
    baseline_only_wins: baselineOnly,
    net_paired_wins: selectedOnly - baselineOnly,
    discordant_games: selectedOnly + baselineOnly,
    exact_mcnemar_p: exactTwoSidedBinomial(selectedOnly, baselineOnly),
  };
}

function compactSummary(filePath) {
  if (!filePath) return null;
  const summary = readJson(filePath);
  const wins = summary.standings.reduce((sum, row) => sum + row.wins, 0);
  const losses = summary.standings.reduce((sum, row) => sum + row.losses, 0);
  return {
    path: path.relative(repoRoot, filePath).replace(/\\/g, '/'),
    run_id: summary.run_id,
    games: wins + losses,
    wins,
    losses,
    win_rate: wins / Math.max(1, wins + losses),
    wilson_95: wilson(wins, wins + losses),
    standings: summary.standings,
  };
}

function run(args) {
  const selectedByMode = new Map(args.selected.map(entry => [entry.mode, readJson(entry.filePath)]));
  const baselineByMode = new Map(args.baseline.map(entry => [entry.mode, readJson(entry.filePath)]));
  const modes = [...selectedByMode.keys()].sort();
  if (modes.some(mode => !baselineByMode.has(mode))) throw new Error('Selected/baseline modes do not match');
  const selectedManifest = readJson(args.selectedManifest);
  const selectedModels = selectedManifest.models || selectedManifest;
  const teamIds = selectedByMode.get(modes[0]).standings.map(row => row.team_id).sort();
  const validatedModels = {};
  const teamResults = {};

  for (const teamId of teamIds) {
    const modeRows = {};
    const totals = {
      selected: {games: 0, wins: 0, losses: 0},
      baseline: {games: 0, wins: 0, losses: 0},
    };
    for (const mode of modes) {
      const selectedRow = rowFor(selectedByMode.get(mode), teamId);
      const baselineRow = rowFor(baselineByMode.get(mode), teamId);
      modeRows[mode] = {selected: selectedRow, baseline: baselineRow};
      for (const [kind, row] of [['selected', selectedRow], ['baseline', baselineRow]]) {
        totals[kind].games += row.wins + row.losses;
        totals[kind].wins += row.wins;
        totals[kind].losses += row.losses;
      }
    }
    for (const total of Object.values(totals)) {
      total.win_rate = total.wins / Math.max(1, total.games);
      total.wilson_95 = wilson(total.wins, total.games);
    }
    const keepSelected = totals.selected.wins > totals.baseline.wins;
    const selectedEntry = selectedModels[teamId];
    const selectedCheckpoint = typeof selectedEntry === 'string' ? selectedEntry : selectedEntry.checkpoint;
    const checkpoint = keepSelected ? selectedCheckpoint :
      `${path.relative(repoRoot, args.baselineModelsDir).replace(/\\/g, '/')}/${teamId}/checkpoint.pt`;
    validatedModels[teamId] = {
      checkpoint,
      policy_source: keepSelected ? 'common_seed_selected' : 'iter_010_fallback',
      holdout_wins: keepSelected ? totals.selected.wins : totals.baseline.wins,
      holdout_games: keepSelected ? totals.selected.games : totals.baseline.games,
    };
    teamResults[teamId] = {
      chosen: keepSelected ? 'selected' : 'baseline_iter_010',
      selected: totals.selected,
      baseline: totals.baseline,
      modes: modeRows,
    };
  }

  const rankings = teamIds.map(teamId => {
    const chosen = teamResults[teamId].chosen === 'selected' ?
      teamResults[teamId].selected : teamResults[teamId].baseline;
    return {team_id: teamId, ...chosen, source: teamResults[teamId].chosen};
  }).sort((a, b) => b.wins - a.wins || b.win_rate - a.win_rate || a.team_id.localeCompare(b.team_id));
  const finalTeam = rankings[0];
  const report = {
    created_at: new Date().toISOString(),
    method: 'independent_stronger_opponent_holdout_with_iter10_fallback',
    modes,
    paired_global: Object.fromEntries(modes.map(mode => [mode, pairedDifference(
      selectedByMode.get(mode),
      baselineByMode.get(mode)
    )])),
    team_results: teamResults,
    final_team_ranking: rankings,
    final_team: finalTeam.team_id,
    head_to_head: compactSummary(args.headToHead),
    top_k_search: args.search ? {
      top_1: compactSummary(args.searchTop1),
      top_4: compactSummary(args.search),
      paired_top4_vs_top1: args.searchTop1 ? pairedDifference(
        readJson(args.search),
        readJson(args.searchTop1)
      ) : null,
    } : null,
  };
  const manifest = {
    created_at: report.created_at,
    holdout_report: path.relative(repoRoot, args.out).replace(/\\/g, '/'),
    models: validatedModels,
  };
  const finalSelection = {
    created_at: report.created_at,
    team_id: finalTeam.team_id,
    checkpoint: validatedModels[finalTeam.team_id].checkpoint,
    policy_source: validatedModels[finalTeam.team_id].policy_source,
    holdout_wins: finalTeam.wins,
    holdout_games: finalTeam.games,
    report: path.relative(repoRoot, args.out).replace(/\\/g, '/'),
  };
  for (const [filePath, value] of [
    [args.validatedManifest, manifest],
    [args.out, report],
    [args.finalSelection, finalSelection],
  ]) {
    fs.mkdirSync(path.dirname(filePath), {recursive: true});
    fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  }
  console.log(`Validated final team: ${finalTeam.team_id} (${finalTeam.wins}/${finalTeam.games})`);
}

try {
  run(parseArgs(process.argv.slice(2)));
} catch (error) {
  console.error(`FAIL ${error.stack || error.message}`);
  process.exit(1);
}
