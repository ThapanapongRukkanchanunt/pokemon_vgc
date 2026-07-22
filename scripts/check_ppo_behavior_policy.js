const assert = require('node:assert/strict');
const {behaviorPolicyProbability} = require('../src/agents/ppo_policy_agent');

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
console.log('PASS PPO behavior probabilities match greedy, sampled, and uniform selection');
