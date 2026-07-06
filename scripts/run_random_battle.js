const {closeTorchPolicyScorers, createAgent} = require('../src/agents');
const {
  findLeadMode,
  findTeam,
  loadTeamPool,
  makeRng,
  runBattle,
} = require('../src/battle/run_battle');

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('--')) throw new Error(`Unexpected argument: ${arg}`);
    const key = arg.slice(2);
    const value = argv[i + 1];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for --${key}`);
    args[key] = value;
    i++;
  }
  return args;
}

async function main() {
  try {
    const args = parseArgs(process.argv.slice(2));
    const seed = args.seed || String(Date.now());
    const rng = makeRng(seed);
    const pool = loadTeamPool();

    const p1Team = findTeam(pool, args['p1-team'], rng);
    const p2Team = findTeam(pool, args['p2-team'], rng);
    const p1Lead = findLeadMode(p1Team, args['p1-lead'], rng);
    const p2Lead = findLeadMode(p2Team, args['p2-lead'], rng);
    const p1Agent = createAgent(args['p1-agent'] || 'random', {
      formatId: pool.format_id,
      modelPath: args['p1-model'],
      valueModelPath: args['p1-value-model'],
      riskMode: args['p1-risk-mode'],
      pythonPath: args['p1-python'],
      torchDevice: args['p1-torch-device'],
    });
    const p2Agent = createAgent(args['p2-agent'] || 'random', {
      formatId: pool.format_id,
      modelPath: args['p2-model'],
      valueModelPath: args['p2-value-model'],
      riskMode: args['p2-risk-mode'],
      pythonPath: args['p2-python'],
      torchDevice: args['p2-torch-device'],
    });

    const result = await runBattle({
      pool,
      seed,
      p1Team,
      p2Team,
      p1Lead,
      p2Lead,
      p1Agent,
      p2Agent,
      logDir: args['log-dir'],
      rng,
    });

    console.log(JSON.stringify(result, null, 2));
  } finally {
    await closeTorchPolicyScorers();
  }
}

main().catch(error => {
  console.error(error.stack || error.message || error);
  process.exit(1);
});
