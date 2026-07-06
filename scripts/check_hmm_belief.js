const assert = require('node:assert/strict');
const {
  HIDDEN_STATES,
  HMMBeliefState,
  emissionLikelihoods,
} = require('../src/belief/hmm_belief');

function probabilitySum(snapshot) {
  return HIDDEN_STATES.reduce((sum, state) => sum + (snapshot.probabilities[state] || 0), 0);
}

function assertNormalized(snapshot, label) {
  assert.ok(Math.abs(probabilitySum(snapshot) - 1) < 0.0015, `${label} posterior should sum to 1`);
}

function updateRepeated(belief, observation, count) {
  let snapshot = null;
  for (let i = 0; i < count; i++) snapshot = belief.updateFromObservation(observation);
  return snapshot;
}

const opening = {
  turn: 0,
  requestType: 'team_preview',
  ownAlive: 4,
  ownActiveCount: 0,
  ownActiveLowHp: 0,
  ownActiveCriticalHp: 0,
  foeVisibleCount: 0,
  foeVisibleLowHp: 0,
  foeVisibleCriticalHp: 0,
  forceSwitch: false,
  switchChoices: 0,
  speedControlMoves: 0,
  pivotMoves: 0,
  protectMoves: 0,
  damagingMoves: 0,
};

const speedControl = {
  ...opening,
  turn: 1,
  requestType: 'move',
  ownActiveCount: 2,
  switchChoices: 2,
  speedControlMoves: 2,
  protectMoves: 1,
  damagingMoves: 2,
};

const pressure = {
  ...speedControl,
  turn: 3,
  ownAlive: 3,
  ownActiveLowHp: 2,
  ownActiveCriticalHp: 1,
  forceSwitch: true,
  speedControlMoves: 0,
  pivotMoves: 0,
};

const endgame = {
  ...pressure,
  turn: 8,
  ownAlive: 2,
  ownActiveLowHp: 1,
  ownActiveCriticalHp: 0,
  forceSwitch: false,
  foeVisibleLowHp: 1,
  foeVisibleCriticalHp: 1,
  damagingMoves: 3,
};

const belief = new HMMBeliefState();
const openingSnapshot = belief.updateFromObservation(opening);
assertNormalized(openingSnapshot, 'opening');
assert.equal(openingSnapshot.top_state, 'neutral');

const speedSnapshot = updateRepeated(belief, speedControl, 2);
assertNormalized(speedSnapshot, 'speed_control');
assert.equal(speedSnapshot.top_state, 'speed_control');

const pressureSnapshot = updateRepeated(belief, pressure, 2);
assertNormalized(pressureSnapshot, 'pressure');
assert.ok(
  ['pressure', 'defensive', 'pivot'].includes(pressureSnapshot.top_state),
  `pressure observation should move to a pressure-like state, got ${pressureSnapshot.top_state}`
);

const endgameSnapshot = updateRepeated(belief, endgame, 3);
assertNormalized(endgameSnapshot, 'endgame');
assert.equal(endgameSnapshot.top_state, 'endgame');

const emissions = emissionLikelihoods(speedControl);
assert.ok(emissions.speed_control > emissions.neutral);
assert.ok(emissions.speed_control > emissions.pressure);

console.log('PASS hmm belief posterior normalization and state movement');
