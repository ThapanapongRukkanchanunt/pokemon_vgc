const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createRandomAgent} = require('../src/agents/random_agent');

const repoRoot = path.join(__dirname, '..');
const rolloutModulePath = require.resolve('../src/battle/rollout_search');
const runBattleModulePath = require.resolve('../src/battle/run_battle');
const rolloutModule = require(rolloutModulePath);
const originalCreateRolloutSearch = rolloutModule.createRolloutSearch;
const snapshots = [];

rolloutModule.createRolloutSearch = options => {
  snapshots.push(options.battleSnapshot);
  return null;
};
delete require.cache[runBattleModulePath];

const {
  findLeadMode,
  loadTeamPool,
  makeRng,
  runBattle,
} = require(runBattleModulePath);

async function check() {
  const pool = loadTeamPool();
  const team = pool.teams[0];
  const rng = makeRng('check-simultaneous-decisions');
  const logDir = path.join(os.tmpdir(), 'pokemon-vgc-check-simultaneous-decisions');
  fs.rmSync(logDir, {recursive: true, force: true});

  try {
    await runBattle({
      pool,
      seed: 'check-simultaneous-decisions',
      p1Team: team,
      p2Team: pool.teams[1],
      p1Lead: findLeadMode(team, null, rng),
      p2Lead: findLeadMode(pool.teams[1], null, rng),
      p1Agent: createRandomAgent({formatId: pool.format_id}),
      p2Agent: createRandomAgent({formatId: pool.format_id}),
      logDir,
      rng,
    });

    assert(snapshots.length >= 2, 'expected rollout snapshots for both preview decisions');
    assert.strictEqual(snapshots[0], snapshots[1], 'both sides must receive the same snapshot object');
    for (const side of snapshots[0].sides) {
      assert.equal(side.choice.actions.length, 0, 'snapshot must not contain a submitted choice');
    }
    console.log('PASS both agents choose from one pre-action battle snapshot');
  } finally {
    fs.rmSync(logDir, {recursive: true, force: true});
    rolloutModule.createRolloutSearch = originalCreateRolloutSearch;
    delete require.cache[runBattleModulePath];
  }
}

check().catch(error => {
  console.error(`FAIL ${error.stack || error.message}`);
  process.exit(1);
});
