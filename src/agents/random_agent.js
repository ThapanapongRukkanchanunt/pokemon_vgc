const {
  chooseSwitch,
  isTeamPreviewRequest,
  switchableSlots,
  teamPreviewChoice,
} = require('./action_utils');
const {dexForFormat} = require('../battle/showdown_protocol');
const {enumerateLegalActions} = require('./legal_actions');

function chooseRandomAction({side, request, battleState, rng, formatId = 'vgc'}) {
  if (request.wait) return null;
  if (isTeamPreviewRequest(request)) return teamPreviewChoice(battleState);

  const dex = dexForFormat(formatId);
  const actions = enumerateLegalActions({side, request, battleState, dex});
  return actions.length ? rng.pick(actions).choice : null;
}

function createRandomAgent({formatId = 'vgc'} = {}) {
  return {
    name: 'random_agent',
    displayName: 'Random',
    // Current limitations: uniform full-choice sampling is intentionally naive and
    // does not weight moves, switches, targets, or Mega timing by board quality.
    chooseAction({side, request, battleState, rng}) {
      return chooseRandomAction({side, request, battleState, rng, formatId});
    },
  };
}

module.exports = {
  chooseSwitch,
  createRandomAgent,
  chooseRandomAction,
  switchableSlots,
};
