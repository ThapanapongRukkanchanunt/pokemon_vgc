const fs = require('node:fs');
const path = require('node:path');
const {loadPool, relativePath, resolveRepoPath} = require('../src/team_building/team_candidates');
const {
  closeTorchPolicyScorers,
  evaluateCandidatePool,
  printCandidateTable,
} = require('../src/team_building/team_evaluation');

const repoRoot = path.join(__dirname, '..');

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseIdList(value) {
  return String(value || '').split(',').map(item => item.trim()).filter(Boolean);
}

function parseArgs(argv) {
  const args = {
    candidates: path.join(repoRoot, 'data', 'team_building', 'phase7_candidates', 'candidates.json'),
    metagame: path.join(repoRoot, 'data', 'teams', 'team_pool.json'),
    outDir: path.join(repoRoot, 'experiments', 'team_building', 'phase7_eval'),
    logDir: path.join(repoRoot, 'logs', 'battles', 'phase7_team_eval'),
    cachePath: null,
    agent: 'hmm_belief',
    opponentAgent: null,
    modelPath: path.join(repoRoot, 'models', 'bc_policy', 'phase6_search_improved', 'model.json'),
    valueModelPath: path.join(repoRoot, 'models', 'value_model', 'phase4_mixed_q', 'model.json'),
    riskMode: 'balanced',
    gamesPerMatchup: 1,
    seed: 'phase7_team_eval',
    candidateLimit: null,
    metagameTeamIds: null,
    includeSideSwaps: true,
    overwrite: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--candidates') {
      args.candidates = resolveRepoPath(argv[++i]);
    } else if (arg === '--metagame') {
      args.metagame = resolveRepoPath(argv[++i]);
    } else if (arg === '--out-dir') {
      args.outDir = resolveRepoPath(argv[++i]);
    } else if (arg === '--log-dir') {
      args.logDir = resolveRepoPath(argv[++i]);
    } else if (arg === '--cache') {
      args.cachePath = resolveRepoPath(argv[++i]);
    } else if (arg === '--agent') {
      args.agent = argv[++i];
    } else if (arg === '--opponent-agent') {
      args.opponentAgent = argv[++i];
    } else if (arg === '--model') {
      args.modelPath = resolveRepoPath(argv[++i]);
    } else if (arg === '--value-model') {
      args.valueModelPath = resolveRepoPath(argv[++i]);
    } else if (arg === '--risk-mode') {
      args.riskMode = argv[++i];
    } else if (arg === '--games-per-matchup') {
      args.gamesPerMatchup = parseInteger(argv[++i], '--games-per-matchup');
    } else if (arg === '--seed') {
      args.seed = argv[++i];
    } else if (arg === '--candidate-limit') {
      args.candidateLimit = parseInteger(argv[++i], '--candidate-limit');
    } else if (arg === '--metagame-teams') {
      args.metagameTeamIds = parseIdList(argv[++i]);
    } else if (arg === '--no-side-swaps') {
      args.includeSideSwaps = false;
    } else if (arg === '--overwrite') {
      args.overwrite = true;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (args.gamesPerMatchup <= 0) throw new Error('--games-per-matchup must be > 0');
  if (args.candidateLimit !== null && args.candidateLimit <= 0) {
    throw new Error('--candidate-limit must be > 0');
  }
  args.opponentAgent = args.opponentAgent || args.agent;
  args.cachePath = args.cachePath || path.join(args.outDir, 'result_cache.jsonl');
  return args;
}

function selectTeams(pool, teamIds, limit, label) {
  let teams = pool.teams || [];
  if (teamIds?.length) {
    const byId = new Map(teams.map(team => [team.id, team]));
    teams = teamIds.map(id => {
      const team = byId.get(id);
      if (!team) throw new Error(`Unknown ${label} team id: ${id}`);
      return team;
    });
  }
  if (limit !== null && limit !== undefined) teams = teams.slice(0, limit);
  if (!teams.length) throw new Error(`No ${label} teams selected`);
  return teams;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const summaryPath = path.join(args.outDir, 'summary.json');
  if (!args.overwrite && fs.existsSync(summaryPath)) {
    throw new Error(`Summary exists at ${relativePath(summaryPath)}; pass --overwrite to replace it`);
  }
  fs.mkdirSync(args.outDir, {recursive: true});

  const candidatePool = loadPool(args.candidates);
  const metagamePool = loadPool(args.metagame);
  if (candidatePool.format_id !== metagamePool.format_id) {
    throw new Error(`Format mismatch: candidates=${candidatePool.format_id}, metagame=${metagamePool.format_id}`);
  }
  const candidates = selectTeams(candidatePool, null, args.candidateLimit, 'candidate');
  const metagameTeams = selectTeams(metagamePool, args.metagameTeamIds, null, 'metagame');

  const summary = await evaluateCandidatePool({
    candidatePool,
    metagamePool,
    candidates,
    metagameTeams,
    args,
  });
  summary.candidate_pool_path = relativePath(args.candidates);
  summary.metagame_pool_path = relativePath(args.metagame);
  fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  printCandidateTable(summary.candidate_table);
  console.log(`Wrote summary: ${relativePath(summaryPath)}`);
}

main()
  .finally(() => closeTorchPolicyScorers())
  .catch(error => {
    console.error(`FAIL ${error.stack || error.message}`);
    process.exit(1);
  });
