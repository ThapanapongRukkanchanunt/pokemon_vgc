const {createMaxDamageAgent} = require('./max_damage_agent');
const {createBcPolicyAgent} = require('./bc_policy_agent');
const {closePpoPolicyScorers, createPpoPolicyAgent} = require('./ppo_policy_agent');
const {createRandomAgent} = require('./random_agent');
const {closeTorchPolicyScorers: closeTorchOnlyPolicyScorers, createTorchPolicyAgent} = require('./torch_policy_agent');
const {
  createHeuristicSelectorAgent,
  createHmmBeliefSelectorAgent,
  createHybridSelectorAgent,
  createPolicySelectorAgent,
  createPolicyValueRiskSelectorAgent,
  createShallowSearchSelectorAgent,
  createValueSelectorAgent,
} = require('./selector_agent');

function createAgent(name, options = {}) {
  const id = (name || 'random').toLowerCase();
  if (['random', 'random_agent'].includes(id)) return createRandomAgent(options);
  if (['maxdamage', 'max_damage', 'max_damage_agent'].includes(id)) {
    return createMaxDamageAgent(options);
  }
  if (['bc', 'bc_policy', 'bc_policy_agent'].includes(id)) {
    return createBcPolicyAgent(options);
  }
  if (['torch', 'torch_policy', 'torch_policy_agent', 'pytorch', 'pytorch_policy'].includes(id)) {
    return createTorchPolicyAgent(options);
  }
  if (['ppo_policy', 'rl_policy', 'final_rl', 'final_rl_agent'].includes(id)) {
    return createPpoPolicyAgent(options);
  }
  if (['heuristic', 'heuristic_selector', 'heuristic_selector_agent', 'selector_heuristic'].includes(id)) {
    return createHeuristicSelectorAgent(options);
  }
  if (['policy_selector', 'policy_selector_agent', 'model_only_selector'].includes(id)) {
    return createPolicySelectorAgent(options);
  }
  if ([
    'hybrid',
    'hybrid_selector',
    'hybrid_selector_agent',
    'model_plus_heuristic',
    'policy_heuristic',
  ].includes(id)) {
    return createHybridSelectorAgent(options);
  }
  if (['value', 'value_selector', 'value_selector_agent', 'q_selector'].includes(id)) {
    return createValueSelectorAgent(options);
  }
  if ([
    'policy_value',
    'policy_value_risk',
    'policy_value_risk_selector',
    'risk_selector',
  ].includes(id)) {
    return createPolicyValueRiskSelectorAgent({...options, riskMode: options.riskMode || 'balanced'});
  }
  if (['risk_balanced', 'policy_value_risk_balanced'].includes(id)) {
    return createPolicyValueRiskSelectorAgent({...options, riskMode: 'balanced'});
  }
  if (['risk_stable', 'policy_value_risk_stable'].includes(id)) {
    return createPolicyValueRiskSelectorAgent({...options, riskMode: 'stable'});
  }
  if (['risk_comeback', 'policy_value_risk_comeback'].includes(id)) {
    return createPolicyValueRiskSelectorAgent({...options, riskMode: 'comeback'});
  }
  if (['risk_closing', 'policy_value_risk_closing'].includes(id)) {
    return createPolicyValueRiskSelectorAgent({...options, riskMode: 'closing'});
  }
  if ([
    'shallow_search',
    'search',
    'search_selector',
    'search_balanced',
    'shallow_search_balanced',
  ].includes(id)) {
    return createShallowSearchSelectorAgent({...options, riskMode: 'balanced'});
  }
  if (['search_stable', 'shallow_search_stable'].includes(id)) {
    return createShallowSearchSelectorAgent({...options, riskMode: 'stable'});
  }
  if (['search_comeback', 'shallow_search_comeback'].includes(id)) {
    return createShallowSearchSelectorAgent({...options, riskMode: 'comeback'});
  }
  if (['search_closing', 'shallow_search_closing'].includes(id)) {
    return createShallowSearchSelectorAgent({...options, riskMode: 'closing'});
  }
  if ([
    'hmm_belief',
    'hmm_search',
    'belief_search',
    'hmm_belief_agent',
  ].includes(id)) {
    return createHmmBeliefSelectorAgent({...options, riskMode: options.riskMode || 'balanced'});
  }
  throw new Error(`Unknown agent: ${name}`);
}

async function closeTorchPolicyScorers() {
  await Promise.allSettled([
    closeTorchOnlyPolicyScorers(),
    closePpoPolicyScorers(),
  ]);
}

module.exports = {
  closeTorchPolicyScorers,
  createAgent,
  createBcPolicyAgent,
  createHeuristicSelectorAgent,
  createHmmBeliefSelectorAgent,
  createHybridSelectorAgent,
  createMaxDamageAgent,
  createPolicySelectorAgent,
  createPolicyValueRiskSelectorAgent,
  createPpoPolicyAgent,
  createRandomAgent,
  createShallowSearchSelectorAgent,
  createTorchPolicyAgent,
  createValueSelectorAgent,
};
