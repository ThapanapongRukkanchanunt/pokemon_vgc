const path = require('node:path');
const {
  loadPool,
  relativePath,
  resolveRepoPath,
  validatePoolTeam,
} = require('../src/team_building/team_candidates');

const repoRoot = path.join(__dirname, '..');

function parseArgs(argv) {
  const args = {
    candidates: path.join(repoRoot, 'data', 'team_building', 'phase7_candidates', 'candidates.json'),
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--candidates') {
      args.candidates = resolveRepoPath(argv[++i]);
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }
  return args;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const pool = loadPool(args.candidates);
  const seenHashes = new Set();
  let failed = false;

  for (const team of pool.teams || []) {
    const problems = validatePoolTeam({pool, team});
    if (team.hash) {
      if (seenHashes.has(team.hash)) problems.push(`duplicate candidate hash ${team.hash}`);
      seenHashes.add(team.hash);
    }
    if (problems.length) {
      failed = true;
      console.log(`FAIL ${team.id} ${team.name}`);
      for (const problem of problems) console.log(`  ${problem}`);
    } else {
      console.log(`PASS ${team.id} ${team.name}`);
    }
  }

  if (failed) process.exit(1);
  console.log(`Validated ${pool.teams?.length || 0} candidate team(s): ${relativePath(args.candidates)}`);
}

main();
