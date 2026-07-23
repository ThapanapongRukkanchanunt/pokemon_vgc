const assert = require('node:assert/strict');
const {
  behaviorPolicyProbability,
  bestMegaActionIndex,
  shouldForceSoleUsableMega,
} = require('../src/agents/ppo_policy_agent');
const {dexForFormat} = require('../src/battle/showdown_protocol');

const probabilities = [0.7, 0.2, 0.1];
const epsilon = 0.2;

function probability(actionIndex, options = {}) {
  return behaviorPolicyProbability({
    actionIndex,
    probabilities,
    exploitationIndex: 0,
    epsilon,
    sampled: false,
    ...options,
  });
}

assert(Math.abs(probability(0) - (0.8 + 0.2 / 3)) < 1e-12);
assert(Math.abs(probability(1) - (0.2 / 3)) < 1e-12);
assert(Math.abs([0, 1, 2].reduce((sum, index) => sum + probability(index), 0) - 1) < 1e-12);

const sampledTotal = [0, 1, 2].reduce((sum, actionIndex) => sum + probability(actionIndex, {
  sampled: true,
}), 0);
assert(Math.abs(sampledTotal - 1) < 1e-12);
assert(Math.abs(probability(1, {sampled: true}) - (0.8 * 0.2 + 0.2 / 3)) < 1e-12);

assert(Math.abs(probability(2, {uniform: true}) - 1 / 3) < 1e-12);

const dex = dexForFormat('gen9championsvgc2026regmb');
const soleMegaRequest = {
  active: [{moves: []}, {moves: [], canMegaEvo: true}],
  side: {
    pokemon: [
      {details: 'Pelipper, L50', item: 'focussash', condition: '137/137'},
      {details: 'Swampert, L50', item: 'swampertite', condition: '177/177'},
      {details: 'Archaludon, L50', item: 'leftovers', condition: '197/197'},
      {details: 'Basculegion, L50', item: 'lifeorb', condition: '199/199'},
    ],
  },
};
assert.equal(shouldForceSoleUsableMega({request: soleMegaRequest, dex}), true);
soleMegaRequest.side.pokemon.push({
  details: 'Lopunny, L50',
  item: 'lopunnite',
  condition: '141/141',
});
assert.equal(shouldForceSoleUsableMega({request: soleMegaRequest, dex}), false);
soleMegaRequest.side.pokemon.at(-1).condition = '0 fnt';
assert.equal(shouldForceSoleUsableMega({request: soleMegaRequest, dex}), true);
soleMegaRequest.active[1].canMegaEvo = false;
assert.equal(shouldForceSoleUsableMega({request: soleMegaRequest, dex}), false);

const legalActions = [
  {choice: 'move 1, move 3', decisions: [{mega: false}, {mega: false}]},
  {choice: 'move 1, move 3 mega', decisions: [{mega: false}, {mega: true}]},
  {choice: 'move 2, move 1 mega', decisions: [{mega: false}, {mega: true}]},
];
assert.equal(bestMegaActionIndex(legalActions, [2, 1, 1.5]), 2);
console.log('PASS PPO behavior probabilities match greedy, sampled, and uniform selection');
console.log('PASS sole-usable-Mega guard detects eligibility and selects the best Mega action');
