const fs = require('node:fs');
const path = require('node:path');
const {dexForFormat} = require('../battle/showdown_protocol');
const {contextFromBattle} = require('../bc/feature_encoder');
const {loadModel: loadPolicyModel, scoreChoices: scorePolicyChoices} = require('../bc/linear_policy');
const {loadModel: loadValueModel, scoreChoices: scoreValueChoices} = require('../value/linear_value');
const {
  HeuristicSelector,
  HMMBeliefSelector,
  HybridPolicyHeuristicSelector,
  PolicyValueRiskSelector,
  PolicySelector,
  ShallowSearchSelector,
  ValueSelector,
} = require('../selectors');
const {enumerateLegalActions} = require('./legal_actions');

const repoRoot = path.join(__dirname, '..', '..');

function opponentSide(side) {
  return side === 'p1' ? 'p2' : 'p1';
}

function resolveModelPath(modelPath, modelKind = 'model') {
  if (!modelPath) throw new Error(`Selector ${modelKind} agent requires a modelPath option`);
  return path.isAbsolute(modelPath) ? modelPath : path.resolve(repoRoot, modelPath);
}

function loadSelectorPolicyModel(modelPath) {
  const resolvedModelPath = resolveModelPath(modelPath, 'policy');
  if (!fs.existsSync(resolvedModelPath)) {
    throw new Error(`Selector policy model not found: ${resolvedModelPath}`);
  }
  return {
    model: loadPolicyModel(resolvedModelPath),
    modelPath: path.relative(repoRoot, resolvedModelPath).replace(/\\/g, '/'),
  };
}

function loadSelectorValueModel(modelPath) {
  const resolvedModelPath = resolveModelPath(modelPath, 'value');
  if (!fs.existsSync(resolvedModelPath)) {
    throw new Error(`Selector value model not found: ${resolvedModelPath}`);
  }
  return {
    model: loadValueModel(resolvedModelPath),
    modelPath: path.relative(repoRoot, resolvedModelPath).replace(/\\/g, '/'),
  };
}

function battleStateForSide(battleState, side) {
  if (!battleState) return battleState;
  return {
    ...battleState,
    team: battleState.teams?.[side] || battleState.team,
    leadMode: battleState.leadModes?.[side] || battleState.leadMode,
  };
}

function chooseWithSelector({
  side,
  request,
  battleState,
  rng,
  formatId,
  selector,
  policyModel = null,
  valueModel = null,
}) {
  if (request.wait) return null;
  const dex = dexForFormat(formatId);
  const sideBattleState = battleStateForSide(battleState, side);
  const legalActions = enumerateLegalActions({side, request, battleState: sideBattleState, dex});
  if (!legalActions.length) return null;

  const state = contextFromBattle({side, request, battleState: sideBattleState});
  const modelScores = policyModel ? scorePolicyChoices(policyModel, state, legalActions) : null;
  const valueScores = valueModel ? scoreValueChoices(valueModel, state, legalActions) : null;
  const foeSide = opponentSide(side);
  const opponentRequest = battleState?.requests?.[foeSide] || null;
  const opponentBattleState = battleStateForSide(battleState, foeSide);
  const opponentLegalActions = opponentRequest ?
    enumerateLegalActions({side: foeSide, request: opponentRequest, battleState: opponentBattleState, dex}) :
    null;
  const opponentState = opponentRequest ?
    contextFromBattle({side: foeSide, request: opponentRequest, battleState: opponentBattleState}) :
    null;
  const opponentModelScores = policyModel && opponentState && opponentLegalActions?.length ?
    scorePolicyChoices(policyModel, opponentState, opponentLegalActions) :
    null;
  const opponentValueScores = valueModel && opponentState && opponentLegalActions?.length ?
    scoreValueChoices(valueModel, opponentState, opponentLegalActions) :
    null;
  const selected = selector.choose({
    state,
    side,
    request,
    legalActions,
    modelScores,
    valueScores,
    opponentRequest,
    opponentLegalActions,
    opponentModelScores,
    opponentValueScores,
    battleState,
    rng,
    dex,
  });
  return selected ? selected.choice : rng.pick(legalActions).choice;
}

function createHeuristicSelectorAgent({formatId = 'vgc'} = {}) {
  const selector = new HeuristicSelector({formatId, mode: 'tactical'});
  return {
    name: 'heuristic_selector_agent',
    displayName: 'HeuristicSelector',
    selector: selector.name,
    chooseAction({side, request, battleState, rng}) {
      return chooseWithSelector({side, request, battleState, rng, formatId, selector});
    },
  };
}

function createPolicySelectorAgent({modelPath, formatId = 'vgc'} = {}) {
  const loaded = loadSelectorPolicyModel(modelPath);
  const selector = new PolicySelector();
  return {
    name: 'policy_selector_agent',
    displayName: 'PolicySelector',
    selector: selector.name,
    modelPath: loaded.modelPath,
    chooseAction({side, request, battleState, rng}) {
      return chooseWithSelector({
        side,
        request,
        battleState,
        rng,
        formatId,
        selector,
        policyModel: loaded.model,
      });
    },
  };
}

function createHybridSelectorAgent({modelPath, formatId = 'vgc'} = {}) {
  const loaded = loadSelectorPolicyModel(modelPath);
  const selector = new HybridPolicyHeuristicSelector({formatId});
  return {
    name: 'hybrid_selector_agent',
    displayName: 'PolicyHeuristic',
    selector: selector.name,
    modelPath: loaded.modelPath,
    chooseAction({side, request, battleState, rng}) {
      return chooseWithSelector({
        side,
        request,
        battleState,
        rng,
        formatId,
        selector,
        policyModel: loaded.model,
      });
    },
  };
}

function createValueSelectorAgent({valueModelPath, modelPath, formatId = 'vgc'} = {}) {
  const loaded = loadSelectorValueModel(valueModelPath || modelPath);
  const selector = new ValueSelector();
  return {
    name: 'value_selector_agent',
    displayName: 'ValueSelector',
    selector: selector.name,
    valueModelPath: loaded.modelPath,
    chooseAction({side, request, battleState, rng}) {
      return chooseWithSelector({
        side,
        request,
        battleState,
        rng,
        formatId,
        selector,
        valueModel: loaded.model,
      });
    },
  };
}

function createPolicyValueRiskSelectorAgent({
  modelPath,
  policyModelPath,
  valueModelPath,
  formatId = 'vgc',
  riskMode = 'balanced',
  weights = null,
} = {}) {
  const loadedPolicy = loadSelectorPolicyModel(policyModelPath || modelPath);
  const loadedValue = loadSelectorValueModel(valueModelPath);
  const selector = new PolicyValueRiskSelector({formatId, riskMode, weights});
  const displayMode = riskMode.slice(0, 1).toUpperCase() + riskMode.slice(1);
  return {
    name: `policy_value_risk_${riskMode}_agent`,
    displayName: `Risk${displayMode}`,
    selector: selector.name,
    riskMode,
    modelPath: loadedPolicy.modelPath,
    valueModelPath: loadedValue.modelPath,
    chooseAction({side, request, battleState, rng}) {
      return chooseWithSelector({
        side,
        request,
        battleState,
        rng,
        formatId,
        selector,
        policyModel: loadedPolicy.model,
        valueModel: loadedValue.model,
      });
    },
  };
}

function createShallowSearchSelectorAgent({
  modelPath,
  policyModelPath,
  valueModelPath,
  formatId = 'vgc',
  riskMode = 'balanced',
  maxOpponentActions = 4,
} = {}) {
  const loadedPolicy = loadSelectorPolicyModel(policyModelPath || modelPath);
  const loadedValue = loadSelectorValueModel(valueModelPath);
  const selector = new ShallowSearchSelector({formatId, riskMode, maxOpponentActions});
  const displayMode = riskMode.slice(0, 1).toUpperCase() + riskMode.slice(1);
  return {
    name: `shallow_search_${riskMode}_agent`,
    displayName: `Search${displayMode}`,
    selector: selector.name,
    riskMode,
    modelPath: loadedPolicy.modelPath,
    valueModelPath: loadedValue.modelPath,
    chooseAction({side, request, battleState, rng}) {
      return chooseWithSelector({
        side,
        request,
        battleState,
        rng,
        formatId,
        selector,
        policyModel: loadedPolicy.model,
        valueModel: loadedValue.model,
      });
    },
  };
}

function createHmmBeliefSelectorAgent({
  modelPath,
  policyModelPath,
  valueModelPath,
  formatId = 'vgc',
  riskMode = 'balanced',
  maxOpponentActions = 4,
  beliefWeight = 0.35,
} = {}) {
  const loadedPolicy = loadSelectorPolicyModel(policyModelPath || modelPath);
  const loadedValue = loadSelectorValueModel(valueModelPath);
  const selector = new HMMBeliefSelector({formatId, riskMode, maxOpponentActions, beliefWeight});
  return {
    name: 'hmm_belief_agent',
    displayName: 'HMMBelief',
    selector: selector.name,
    riskMode,
    modelPath: loadedPolicy.modelPath,
    valueModelPath: loadedValue.modelPath,
    chooseAction({side, request, battleState, rng}) {
      return chooseWithSelector({
        side,
        request,
        battleState,
        rng,
        formatId,
        selector,
        policyModel: loadedPolicy.model,
        valueModel: loadedValue.model,
      });
    },
    getDiagnostics() {
      return {
        hmm_belief: selector.diagnostics(),
      };
    },
  };
}

module.exports = {
  createHeuristicSelectorAgent,
  createHmmBeliefSelectorAgent,
  createHybridSelectorAgent,
  createPolicySelectorAgent,
  createPolicyValueRiskSelectorAgent,
  createShallowSearchSelectorAgent,
  createValueSelectorAgent,
};
