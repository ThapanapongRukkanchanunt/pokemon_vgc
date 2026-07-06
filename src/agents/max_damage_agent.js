const {dexForFormat} = require('../battle/showdown_protocol');
const {enumerateLegalActions} = require('./legal_actions');
const {HeuristicSelector} = require('../selectors');

function chooseMaxDamageAction({side, request, battleState, rng, formatId = 'vgc'}) {
  if (request.wait) return null;

  const dex = dexForFormat(formatId);
  const legalActions = enumerateLegalActions({side, request, battleState, dex});
  if (!legalActions.length) return null;

  const selector = new HeuristicSelector({formatId, mode: 'max_damage'});
  const selected = selector.choose({side, request, legalActions, battleState, rng, dex});
  return selected ? selected.choice : null;
}

function createMaxDamageAgent({formatId = 'vgc'} = {}) {
  const selector = new HeuristicSelector({formatId, mode: 'max_damage'});

  return {
    name: 'max_damage_agent',
    displayName: 'MaxDamage',
    selector: selector.name,
    // Current limitations: this is a fast heuristic, not Showdown's full damage engine.
    // It scores base power with type effectiveness, STAB, and spread penalty, but ignores
    // stats, items, abilities, weather, boosts, Protect reads, and most move callbacks.
    // If no positive-damage move exists, it switches when possible instead of auto-choosing move 1.
    chooseAction({side, request, battleState, rng}) {
      if (request.wait) return null;
      const dex = dexForFormat(formatId);
      const legalActions = enumerateLegalActions({side, request, battleState, dex});
      if (!legalActions.length) return null;
      const selected = selector.choose({side, request, legalActions, battleState, rng, dex});
      return selected ? selected.choice : null;
    },
  };
}

module.exports = {
  createMaxDamageAgent,
  chooseMaxDamageAction,
};
