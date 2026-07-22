const assert = require('node:assert/strict');
const {
  payoffKey,
  pfspWeight,
  selectHistoricalOpponent,
  smoothedWinRate,
  weightedPick,
} = require('../src/pfsp');

assert.equal(smoothedWinRate(null), 0.5);
assert.equal(smoothedWinRate({games: 2, wins: 0}), 0.25);
assert.ok(
  pfspWeight({games: 10, wins: 1}) > pfspWeight({games: 10, wins: 9}),
  'PFSP must prioritize historical opponents that the current agent struggles against'
);
assert.equal(weightedPick(['easy', 'hard'], [0, 1], {next: () => 0.1}), 'hard');

const snapshots = [{id: 's1'}, {id: 's2'}];
const teams = [{id: 'mb-001'}, {id: 'mb-002'}];
const candidates = snapshots.flatMap(snapshot => teams.map(team => ({snapshot, team})));
const payoffs = new Map([
  [payoffKey('mb-001', 's1', 'mb-002'), {games: 8, wins: 8}],
  [payoffKey('mb-001', 's2', 'mb-002'), {games: 8, wins: 0}],
]);
const selected = selectHistoricalOpponent({
  currentTeamId: 'mb-001',
  candidates,
  payoffs,
  rng: {next: () => 0.8},
  exponent: 2,
  priorGames: 2,
});
assert.equal(selected.team.id, 'mb-002', 'PFSP must exclude same-team historical mirrors');
assert.equal(selected.snapshot.id, 's2', 'PFSP must favor the harder snapshot under the fixed draw');

console.log('PASS PFSP smoothing, weighting, and historical opponent selection');
