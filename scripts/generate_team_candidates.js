const path = require('node:path');
const {makeRng} = require('../src/battle/run_battle');
const {
  generateRandomCandidates,
  loadPool,
  loadSetCatalog,
  relativePath,
  resolveRepoPath,
  writeCandidatePool,
} = require('../src/team_building/team_candidates');

const repoRoot = path.join(__dirname, '..');

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    sourcePool: path.join(repoRoot, 'data', 'teams', 'team_pool.json'),
    outDir: path.join(repoRoot, 'data', 'team_building', 'phase7_candidates'),
    count: 8,
    seed: 'phase7_team_candidates',
    idPrefix: 'phase7-cand',
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
    } else if (arg === '--count') {
      args.count = parseInteger(argv[++i], '--count');
    } else if (arg === '--seed') {
      args.seed = argv[++i];
    } else if (arg === '--id-prefix') {
      args.idPrefix = argv[++i];
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

  if (args.count <= 0) throw new Error('--count must be > 0');
  if (args.minMegas < 0) throw new Error('--min-megas must be >= 0');
  if (args.maxMegas < args.minMegas) throw new Error('--max-megas must be >= --min-megas');
  return args;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const sourcePool = loadPool(args.sourcePool);
  const catalog = loadSetCatalog({pool: sourcePool});
  const rng = makeRng(args.seed);
  const generatorOptions = {
    minMegas: args.minMegas,
    maxMegas: args.maxMegas,
    uniqueItems: args.uniqueItems,
    maxAttempts: args.maxAttempts,
  };
  const {candidates, attempts} = generateRandomCandidates({
    catalog,
    rng,
    count: args.count,
    idPrefix: args.idPrefix,
    options: generatorOptions,
  });
  const {poolPath} = writeCandidatePool({
    sourcePool,
    sourcePoolPath: args.sourcePool,
    candidates,
    outDir: args.outDir,
    seed: args.seed,
    generator: {
      strategy: 'set_recombination',
      requested_count: args.count,
      attempts,
      catalog_sets: catalog.set_count,
      options: generatorOptions,
    },
    overwrite: args.overwrite,
  });

  console.log(`Wrote ${candidates.length} candidate team(s): ${relativePath(poolPath)}`);
  for (const candidate of candidates) {
    console.log(`${candidate.id},hash=${candidate.hash},species=${candidate.species.join('|')}`);
  }
}

main();
