const fs = require('node:fs');
const path = require('node:path');
const {makeRng} = require('../src/battle/run_battle');
const {
  candidateFromEntries,
  crossoverEntries,
  generateRandomCandidates,
  loadPool,
  loadSetCatalog,
  mutateEntries,
  relativePath,
  resolveRepoPath,
  writeCandidatePool,
} = require('../src/team_building/team_candidates');
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
    sourcePool: path.join(repoRoot, 'data', 'teams', 'team_pool.json'),
    outDir: path.join(repoRoot, 'experiments', 'team_building', 'phase7_evolution'),
    candidateOutDir: path.join(repoRoot, 'data', 'team_building', 'phase7_evolution'),
    logDir: path.join(repoRoot, 'logs', 'battles', 'phase7_evolution'),
    cachePath: null,
    populationSize: 6,
    generations: 2,
    eliteCount: 2,
    gamesPerMatchup: 1,
    seed: 'phase7_evolution',
    metagameTeamIds: ['full-001', 'full-002', 'full-003'],
    includeSideSwaps: true,
    agent: 'heuristic_selector',
    opponentAgent: null,
    modelPath: path.join(repoRoot, 'models', 'bc_policy', 'phase6_search_improved', 'model.json'),
    valueModelPath: path.join(repoRoot, 'models', 'value_model', 'phase4_mixed_q', 'model.json'),
    riskMode: 'balanced',
    minMegas: 1,
    maxMegas: 2,
    uniqueItems: true,
    maxAttempts: 5000,
    overwrite: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--source-pool') {
      args.sourcePool = resolveRepoPath(argv[++i]);
    } else if (arg === '--out-dir') {
      args.outDir = resolveRepoPath(argv[++i]);
    } else if (arg === '--candidate-out-dir') {
      args.candidateOutDir = resolveRepoPath(argv[++i]);
    } else if (arg === '--log-dir') {
      args.logDir = resolveRepoPath(argv[++i]);
    } else if (arg === '--cache') {
      args.cachePath = resolveRepoPath(argv[++i]);
    } else if (arg === '--population-size') {
      args.populationSize = parseInteger(argv[++i], '--population-size');
    } else if (arg === '--generations') {
      args.generations = parseInteger(argv[++i], '--generations');
    } else if (arg === '--elite-count') {
      args.eliteCount = parseInteger(argv[++i], '--elite-count');
    } else if (arg === '--games-per-matchup') {
      args.gamesPerMatchup = parseInteger(argv[++i], '--games-per-matchup');
    } else if (arg === '--seed') {
      args.seed = argv[++i];
    } else if (arg === '--metagame-teams') {
      args.metagameTeamIds = parseIdList(argv[++i]);
    } else if (arg === '--no-side-swaps') {
      args.includeSideSwaps = false;
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
    } else if (arg === '--min-megas') {
      args.minMegas = parseInteger(argv[++i], '--min-megas');
    } else if (arg === '--max-megas') {
      args.maxMegas = parseInteger(argv[++i], '--max-megas');
    } else if (arg === '--max-attempts') {
      args.maxAttempts = parseInteger(argv[++i], '--max-attempts');
    } else if (arg === '--allow-duplicate-items') {
      args.uniqueItems = false;
    } else if (arg === '--overwrite') {
      args.overwrite = true;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (args.populationSize <= 1) throw new Error('--population-size must be > 1');
  if (args.generations <= 0) throw new Error('--generations must be > 0');
  if (args.eliteCount <= 0 || args.eliteCount > args.populationSize) {
    throw new Error('--elite-count must be between 1 and --population-size');
  }
  if (args.gamesPerMatchup <= 0) throw new Error('--games-per-matchup must be > 0');
  args.opponentAgent = args.opponentAgent || args.agent;
  args.cachePath = args.cachePath || path.join(args.outDir, 'result_cache.jsonl');
  return args;
}

function selectMetagameTeams(pool, teamIds) {
  const byId = new Map((pool.teams || []).map(team => [team.id, team]));
  const ids = teamIds?.length ? teamIds : (pool.teams || []).map(team => team.id);
  const teams = ids.map(id => {
    const team = byId.get(id);
    if (!team) throw new Error(`Unknown metagame team id: ${id}`);
    return team;
  });
  if (!teams.length) throw new Error('No metagame teams selected');
  return teams;
}

function generationId(generation) {
  return `generation_${String(generation).padStart(3, '0')}`;
}

function rankPopulation(population, candidateTable) {
  const byId = new Map(population.map(candidate => [candidate.id, candidate]));
  return candidateTable.map(row => ({
    row,
    candidate: byId.get(row.candidate_id),
  })).filter(entry => entry.candidate);
}

function makeNextGeneration({ranked, catalog, rng, generation, args}) {
  const generatorOptions = {
    minMegas: args.minMegas,
    maxMegas: args.maxMegas,
    uniqueItems: args.uniqueItems,
    maxAttempts: args.maxAttempts,
  };
  const next = [];
  const seenHashes = new Set();
  const elites = ranked.slice(0, args.eliteCount);

  for (const elite of elites) {
    const candidate = candidateFromEntries({
      entries: elite.candidate.entries,
      id: `phase7-g${String(generation).padStart(3, '0')}-elite-${String(next.length + 1).padStart(3, '0')}`,
      name: `Phase7 G${generation} Elite ${next.length + 1}`,
      formatId: catalog.format_id,
    });
    next.push(candidate);
    seenHashes.add(candidate.hash);
  }

  let attempts = 0;
  while (next.length < args.populationSize && attempts < args.maxAttempts) {
    attempts += 1;
    const parentA = rng.pick(elites).candidate;
    const parentB = rng.pick(elites).candidate;
    const entries = rng.next() < 0.65 && parentA !== parentB
      ? crossoverEntries({
        leftEntries: parentA.entries,
        rightEntries: parentB.entries,
        catalog,
        rng,
        options: generatorOptions,
      })
      : mutateEntries({
        parentEntries: parentA.entries,
        catalog,
        rng,
        options: generatorOptions,
      });
    if (!entries) continue;
    const candidate = candidateFromEntries({
      entries,
      id: `phase7-g${String(generation).padStart(3, '0')}-cand-${String(next.length + 1).padStart(3, '0')}`,
      name: `Phase7 G${generation} Candidate ${next.length + 1}`,
      formatId: catalog.format_id,
    });
    if (seenHashes.has(candidate.hash)) continue;
    seenHashes.add(candidate.hash);
    next.push(candidate);
  }

  if (next.length < args.populationSize) {
    const generated = generateRandomCandidates({
      catalog,
      rng,
      count: args.populationSize - next.length,
      idPrefix: `phase7-g${String(generation).padStart(3, '0')}-fresh`,
      options: generatorOptions,
    }).candidates;
    for (const candidate of generated) {
      if (next.length >= args.populationSize) break;
      if (seenHashes.has(candidate.hash)) continue;
      next.push(candidate);
      seenHashes.add(candidate.hash);
    }
  }

  if (next.length < args.populationSize) {
    throw new Error(`Could only create ${next.length}/${args.populationSize} candidates for generation ${generation}`);
  }
  return next;
}

async function evaluateGeneration({population, sourcePool, metagamePool, metagameTeams, generation, args}) {
  const id = generationId(generation);
  const candidateDir = path.join(args.candidateOutDir, id);
  const {pool: candidatePool, poolPath} = writeCandidatePool({
    sourcePool,
    sourcePoolPath: args.sourcePool,
    candidates: population,
    outDir: candidateDir,
    seed: `${args.seed}:${id}`,
    generator: {
      strategy: generation === 1 ? 'initial_set_recombination' : 'elite_mutation_crossover',
      generation,
      population_size: population.length,
    },
    overwrite: args.overwrite,
  });

  const summary = await evaluateCandidatePool({
    candidatePool,
    metagamePool,
    candidates: candidatePool.teams,
    metagameTeams,
    args: {
      ...args,
      seed: `${args.seed}:${id}`,
      logDir: path.join(args.logDir, id),
      cachePath: args.cachePath,
    },
  });
  summary.generation = generation;
  summary.candidate_pool_path = relativePath(poolPath);
  const generationSummaryPath = path.join(args.outDir, `${id}.summary.json`);
  fs.writeFileSync(generationSummaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  return {
    generation,
    candidate_pool_path: relativePath(poolPath),
    summary_path: relativePath(generationSummaryPath),
    summary,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const summaryPath = path.join(args.outDir, 'summary.json');
  if (!args.overwrite && fs.existsSync(summaryPath)) {
    throw new Error(`Summary exists at ${relativePath(summaryPath)}; pass --overwrite to replace it`);
  }
  fs.mkdirSync(args.outDir, {recursive: true});

  const sourcePool = loadPool(args.sourcePool);
  const metagamePool = loadPool(args.sourcePool);
  const metagameTeams = selectMetagameTeams(metagamePool, args.metagameTeamIds);
  const catalog = loadSetCatalog({pool: sourcePool});
  const rng = makeRng(args.seed);
  const generatorOptions = {
    minMegas: args.minMegas,
    maxMegas: args.maxMegas,
    uniqueItems: args.uniqueItems,
    maxAttempts: args.maxAttempts,
  };

  let population = generateRandomCandidates({
    catalog,
    rng,
    count: args.populationSize,
    idPrefix: 'phase7-g001-cand',
    options: generatorOptions,
  }).candidates;

  const generations = [];
  for (let generation = 1; generation <= args.generations; generation++) {
    const result = await evaluateGeneration({
      population,
      sourcePool,
      metagamePool,
      metagameTeams,
      generation,
      args,
    });
    generations.push(result);
    printCandidateTable(result.summary.candidate_table);
    if (generation < args.generations) {
      const ranked = rankPopulation(population, result.summary.candidate_table);
      population = makeNextGeneration({ranked, catalog, rng, generation: generation + 1, args});
    }
  }

  const bestPerGeneration = generations.map(result => ({
    generation: result.generation,
    candidate_pool_path: result.candidate_pool_path,
    summary_path: result.summary_path,
    best: result.summary.candidate_table[0],
  }));
  const finalSummary = {
    created_at: new Date().toISOString(),
    seed: args.seed,
    source_pool: relativePath(args.sourcePool),
    metagame_team_ids: metagameTeams.map(team => team.id),
    population_size: args.populationSize,
    generations: args.generations,
    elite_count: args.eliteCount,
    games_per_matchup: args.gamesPerMatchup,
    include_side_swaps: args.includeSideSwaps,
    agent: args.agent,
    opponent_agent: args.opponentAgent,
    cache_path: relativePath(args.cachePath),
    generator_options: generatorOptions,
    best_per_generation: bestPerGeneration,
    best_overall: bestPerGeneration
      .map(entry => entry.best)
      .sort((a, b) => b.score - a.score || b.worst_matchup_win_rate - a.worst_matchup_win_rate)[0],
  };
  fs.writeFileSync(summaryPath, `${JSON.stringify(finalSummary, null, 2)}\n`, 'utf8');
  console.log(`Wrote optimization summary: ${relativePath(summaryPath)}`);
  console.log(`Best overall: ${finalSummary.best_overall.candidate_id} score=${finalSummary.best_overall.score.toFixed(3)}`);
}

main()
  .finally(() => closeTorchPolicyScorers())
  .catch(error => {
    console.error(`FAIL ${error.stack || error.message}`);
    process.exit(1);
  });
