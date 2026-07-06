const fs = require('node:fs');
const path = require('node:path');
const {dexForFormat} = require('../battle/showdown_protocol');
const {enumerateLegalActions} = require('./legal_actions');
const {contextFromBattle} = require('../bc/feature_encoder');
const {loadModel, scoreChoices} = require('../bc/linear_policy');
const {PolicySelector} = require('../selectors');

const repoRoot = path.join(__dirname, '..', '..');

function resolveModelPath(modelPath) {
  if (!modelPath) throw new Error('BC policy agent requires a modelPath option');
  return path.isAbsolute(modelPath) ? modelPath : path.resolve(repoRoot, modelPath);
}

function createBcPolicyAgent({modelPath, formatId = 'vgc'} = {}) {
  const resolvedModelPath = resolveModelPath(modelPath);
  if (!fs.existsSync(resolvedModelPath)) {
    throw new Error(`BC policy model not found: ${resolvedModelPath}`);
  }
  const model = loadModel(resolvedModelPath);
  const selector = new PolicySelector();

  return {
    name: 'bc_policy_agent',
    displayName: 'BCPolicy',
    modelPath: path.relative(repoRoot, resolvedModelPath).replace(/\\/g, '/'),
    selector: selector.name,
    chooseAction({side, request, battleState, rng}) {
      if (request.wait) return null;
      const dex = dexForFormat(formatId);
      const legalActions = enumerateLegalActions({side, request, battleState, dex});
      if (!legalActions.length) return null;
      const context = contextFromBattle({side, request, battleState});
      const modelScores = scoreChoices(model, context, legalActions);
      const selected = selector.choose({
        state: context,
        side,
        request,
        legalActions,
        modelScores,
        battleState,
        rng,
        dex,
      });
      return selected ? selected.choice : rng.pick(legalActions).choice;
    },
  };
}

module.exports = {
  createBcPolicyAgent,
};
