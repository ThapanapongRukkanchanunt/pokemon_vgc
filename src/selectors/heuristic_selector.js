const {
  basePowerForMove,
  foeActiveTargets,
  isTeamPreviewRequest,
  moveDamageScore,
  ownActiveSpeciesName,
  speciesNameFromDetails,
  teamPreviewChoice,
  typeEffectivenessMultiplier,
} = require('../agents/action_utils');
const {dexForFormat} = require('../battle/showdown_protocol');

const PROTECT_MOVE_IDS = new Set([
  'banefulbunker',
  'burningbulwark',
  'detect',
  'kingsshield',
  'protect',
  'silktrap',
  'spikyshield',
]);

const SPEED_CONTROL_MOVE_IDS = new Set([
  'afteryou',
  'bulldoze',
  'electroweb',
  'glare',
  'icywind',
  'lowerspeed',
  'quash',
  'rocktomb',
  'scaryface',
  'stringshot',
  'tailwind',
  'thunderwave',
  'trickroom',
]);

const SUPPORT_MOVE_IDS = new Set([
  'fakeout',
  'helpinghand',
  'partingshot',
  'snarl',
  'taunt',
  'willowisp',
]);

function choiceText(action) {
  return typeof action === 'string' ? action : action.choice;
}

function normalizeLegalActions(legalActions) {
  return (legalActions || []).map(action => {
    if (typeof action === 'string') return {choice: action, decisions: []};
    return action;
  });
}

function pickBestByComparator(scored, compare, rng) {
  let bestRows = [];
  for (const row of scored) {
    if (!bestRows.length) {
      bestRows = [row];
      continue;
    }
    const comparison = compare(row, bestRows[0]);
    if (comparison > 0) {
      bestRows = [row];
    } else if (comparison === 0) {
      bestRows.push(row);
    }
  }
  return rng && bestRows.length > 1 ? rng.pick(bestRows) : bestRows[0];
}

function isSpreadMove(move) {
  return ['allAdjacentFoes', 'allAdjacent', 'all'].includes(move.target);
}

function ownActivePokemon(request) {
  return (request.side?.pokemon || []).filter(pokemon => pokemon.active);
}

function ownActivePokemonAt(request, activeIndex) {
  return ownActivePokemon(request)[activeIndex] || null;
}

function speciesForPokemon(dex, pokemon) {
  const speciesName = speciesNameFromDetails(pokemon?.details) || pokemon?.species || pokemon?.ident;
  if (!speciesName) return null;
  const species = dex.species.get(speciesName);
  return species.exists ? species : null;
}

function hpFractionFromCondition(condition) {
  if (!condition) return null;
  const text = String(condition);
  if (/\bfnt\b/.test(text)) return 0;

  const fraction = text.match(/(\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)/);
  if (fraction) {
    const current = Number(fraction[1]);
    const max = Number(fraction[2]);
    return max > 0 ? Math.max(0, Math.min(1, current / max)) : null;
  }

  const percent = text.match(/(\d+(?:\.\d+)?)%/);
  if (percent) return Math.max(0, Math.min(1, Number(percent[1]) / 100));
  return null;
}

function hpPressureBonus(hpFraction) {
  if (hpFraction == null) return 0;
  if (hpFraction <= 0.25) return 18;
  if (hpFraction <= 0.5) return 8;
  return 0;
}

function incomingTypePressure(dex, ownSpecies, foeActives) {
  if (!ownSpecies?.exists) return 1;
  let pressure = 1;
  for (const foe of foeActives) {
    for (const type of foe.species.types || []) {
      pressure = Math.max(pressure, typeEffectivenessMultiplier(dex, {type}, ownSpecies));
    }
  }
  return pressure;
}

function targetObjectsForDecision({decision, move, side, battleState, dex}) {
  const foeActives = foeActiveTargets({side, battleState, dex});
  if (isSpreadMove(move)) return foeActives;

  const targetSlot = Number(decision.target);
  if (!Number.isInteger(targetSlot) || targetSlot < 1) return [];
  return foeActives.filter(active => active.slot === targetSlot - 1);
}

function scoreMaxDamageMoveDecision({decision, side, request, battleState, dex}) {
  const move = dex.moves.get(decision.moveId);
  if (!move.exists) return 0;

  const userSpeciesName = ownActiveSpeciesName(request, decision.activeIndex);
  const userSpecies = userSpeciesName ? dex.species.get(userSpeciesName) : null;
  const targets = targetObjectsForDecision({decision, move, side, battleState, dex});
  const spread = isSpreadMove(move);

  if (!targets.length) return 0;
  return targets.reduce((sum, target) => {
    return sum + moveDamageScore({dex, move, userSpecies, target, spread});
  }, 0);
}

function scoreMaxDamageAction({action, side, request, battleState, dex}) {
  const score = {
    totalDamage: 0,
    damagingMoveCount: 0,
    defensiveMoveCount: 0,
    switchCount: 0,
    statusMoveCount: 0,
  };

  for (const decision of action.decisions || []) {
    if (decision.kind === 'switch') {
      score.switchCount += 1;
      continue;
    }
    if (decision.kind !== 'move') continue;

    const move = dex.moves.get(decision.moveId);
    if (!move.exists) continue;
    const basePower = basePowerForMove(move);
    if (basePower > 0) score.damagingMoveCount += 1;
    if (PROTECT_MOVE_IDS.has(move.id)) score.defensiveMoveCount += 1;
    if (basePower === 0) score.statusMoveCount += 1;
    score.totalDamage += scoreMaxDamageMoveDecision({decision, side, request, battleState, dex});
  }

  return score;
}

function compareMaxDamageScores(a, b) {
  return a.totalDamage - b.totalDamage ||
    a.damagingMoveCount - b.damagingMoveCount ||
    (a.totalDamage <= 0 ? a.switchCount - b.switchCount : 0) ||
    a.defensiveMoveCount - b.defensiveMoveCount ||
    b.statusMoveCount - a.statusMoveCount;
}

function emptyTacticalScore() {
  return {
    total: 0,
    damage: 0,
    typeEffectiveness: 0,
    koPotential: 0,
    priority: 0,
    speedControl: 0,
    immunityAvoidance: 0,
    protectSafety: 0,
    switchSafety: 0,
    statusUtility: 0,
    passPenalty: 0,
  };
}

function addTerm(score, key, value) {
  if (!Number.isFinite(value) || value === 0) return;
  score[key] += value;
  score.total += value;
}

function typeEffectivenessTerm(effectiveness) {
  if (effectiveness === 0) return -80;
  if (effectiveness >= 4) return 36;
  if (effectiveness >= 2) return 18;
  if (effectiveness <= 0.25) return -18;
  if (effectiveness <= 0.5) return -9;
  return 0;
}

function koPotentialTerm({damage, target, effectiveness}) {
  if (damage <= 0) return 0;
  const hpFraction = hpFractionFromCondition(target.condition);
  if (hpFraction != null) {
    const roughDamageFraction = Math.min(1.25, damage / 180);
    if (roughDamageFraction >= hpFraction + 0.05) return 36;
    if (hpFraction <= 0.25) return 18;
    if (hpFraction <= 0.5 && damage >= 100) return 14;
    return 0;
  }
  if (damage >= 180) return 10;
  if (effectiveness >= 2 && damage >= 90) return 8;
  return 0;
}

function scoreProtectMove({score, request, decision}) {
  const active = ownActivePokemonAt(request, decision.activeIndex);
  const hpFraction = hpFractionFromCondition(active?.condition);
  addTerm(score, 'protectSafety', 5 + hpPressureBonus(hpFraction));
}

function scoreSpeedControlMove({score, move}) {
  const base = move.id === 'tailwind' || move.id === 'trickroom' ? 18 : 12;
  addTerm(score, 'speedControl', base);
}

function scoreStatusMove({score, move, request, decision}) {
  let appliedUtility = false;
  if (PROTECT_MOVE_IDS.has(move.id)) {
    scoreProtectMove({score, request, decision});
    appliedUtility = true;
  }
  if (SPEED_CONTROL_MOVE_IDS.has(move.id)) {
    scoreSpeedControlMove({score, move});
    appliedUtility = true;
  }
  if (SUPPORT_MOVE_IDS.has(move.id)) {
    addTerm(score, 'statusUtility', 8);
    appliedUtility = true;
  }
  if (move.priority > 0) {
    addTerm(score, 'priority', 4 * move.priority);
    appliedUtility = true;
  }
  if (!appliedUtility) addTerm(score, 'statusUtility', -4);
}

function activeHasPositiveDamage({activeData, activeIndex, side, request, battleState, dex}) {
  if (!activeData?.moves?.length) return false;
  const userSpeciesName = ownActiveSpeciesName(request, activeIndex);
  const userSpecies = userSpeciesName ? dex.species.get(userSpeciesName) : null;
  const foeActives = foeActiveTargets({side, battleState, dex});
  for (const moveSlot of activeData.moves) {
    if (moveSlot.disabled) continue;
    const move = dex.moves.get(moveSlot.id || moveSlot.move);
    if (!move.exists || basePowerForMove(move) <= 0) continue;
    const spread = isSpreadMove(move);
    for (const target of foeActives) {
      if (moveDamageScore({dex, move, userSpecies, target, spread}) > 0) return true;
    }
  }
  return false;
}

function scoreMoveDecision({score, decision, side, request, battleState, dex}) {
  const move = dex.moves.get(decision.moveId);
  if (!move.exists) {
    addTerm(score, 'statusUtility', -3);
    return;
  }

  const basePower = basePowerForMove(move);
  if (basePower <= 0) {
    scoreStatusMove({score, move, request, decision});
    return;
  }

  const userSpeciesName = ownActiveSpeciesName(request, decision.activeIndex);
  const userSpecies = userSpeciesName ? dex.species.get(userSpeciesName) : null;
  const targets = targetObjectsForDecision({decision, move, side, battleState, dex});
  const spread = isSpreadMove(move);

  if (!targets.length) {
    addTerm(score, 'immunityAvoidance', -20);
    return;
  }

  for (const target of targets) {
    const effectiveness = typeEffectivenessMultiplier(dex, move, target.species);
    const damage = moveDamageScore({dex, move, userSpecies, target, spread});
    addTerm(score, 'damage', damage);
    addTerm(score, 'typeEffectiveness', typeEffectivenessTerm(effectiveness));
    if (effectiveness === 0) addTerm(score, 'immunityAvoidance', -30);
    addTerm(score, 'koPotential', koPotentialTerm({damage, target, effectiveness}));
  }

  if (move.priority > 0) addTerm(score, 'priority', 8 * move.priority);
  if (move.priority < 0) addTerm(score, 'priority', 3 * move.priority);
  if (SPEED_CONTROL_MOVE_IDS.has(move.id)) scoreSpeedControlMove({score, move});
  if (SUPPORT_MOVE_IDS.has(move.id)) addTerm(score, 'statusUtility', 6);
}

function scoreSwitchDecision({score, decision, side, request, battleState, dex}) {
  const active = ownActivePokemonAt(request, decision.activeIndex);
  const switchPokemon = request.side?.pokemon?.[decision.slot - 1] || null;
  const activeHp = hpFractionFromCondition(active?.condition);
  const switchHp = hpFractionFromCondition(switchPokemon?.condition);
  const foeActives = foeActiveTargets({side, battleState, dex});
  const activeSpecies = speciesForPokemon(dex, active);
  const switchSpecies = speciesForPokemon(dex, switchPokemon);
  const activePressure = incomingTypePressure(dex, activeSpecies, foeActives);
  const switchPressure = incomingTypePressure(dex, switchSpecies, foeActives);

  let value = -8 + hpPressureBonus(activeHp);
  if (switchHp != null && switchHp <= 0.25) value -= 10;
  if (activePressure >= 2) value += 8 * Math.log2(activePressure);
  if (activePressure >= 2 && switchPressure < activePressure) value += 10;
  if (switchPressure > activePressure) value -= 8;

  const activeData = request.active?.[decision.activeIndex];
  if (!activeHasPositiveDamage({activeData, activeIndex: decision.activeIndex, side, request, battleState, dex})) {
    value += 10;
  }

  addTerm(score, 'switchSafety', value);
}

function scoreTacticalAction({action, side, request, battleState, dex}) {
  const score = emptyTacticalScore();
  const decisions = action.decisions || [];
  if (!decisions.length) addTerm(score, 'passPenalty', -10);

  for (const decision of decisions) {
    if (decision.kind === 'move') {
      scoreMoveDecision({score, decision, side, request, battleState, dex});
    } else if (decision.kind === 'switch') {
      scoreSwitchDecision({score, decision, side, request, battleState, dex});
    } else if (decision.kind === 'pass') {
      addTerm(score, 'passPenalty', -12);
    }
  }

  return score;
}

class HeuristicSelector {
  constructor({formatId = 'vgc', mode = 'tactical'} = {}) {
    this.formatId = formatId;
    this.mode = mode;
    this.name = mode === 'max_damage' ? 'max_damage_selector' : 'heuristic_selector';
  }

  scoreActions({side, request, legalActions, battleState, dex = null}) {
    const resolvedDex = dex || dexForFormat(this.formatId);
    const actions = normalizeLegalActions(legalActions);
    return actions.map(action => {
      const score = this.mode === 'max_damage'
        ? scoreMaxDamageAction({action, side, request, battleState, dex: resolvedDex})
        : scoreTacticalAction({action, side, request, battleState, dex: resolvedDex});
      return {
        action,
        choice: choiceText(action),
        score,
      };
    });
  }

  choose({side, request, legalActions, battleState, rng, dex = null}) {
    if (request.wait) return null;
    const actions = normalizeLegalActions(legalActions);
    if (!actions.length) return null;

    if (isTeamPreviewRequest(request)) {
      const preferred = teamPreviewChoice(battleState);
      const action = actions.find(candidate => choiceText(candidate) === preferred) || actions[0];
      return {action, choice: choiceText(action), score: 0, scores: []};
    }

    const scored = this.scoreActions({side, request, legalActions: actions, battleState, dex});
    const best = this.mode === 'max_damage'
      ? pickBestByComparator(scored, (a, b) => compareMaxDamageScores(a.score, b.score), rng)
      : pickBestByComparator(scored, (a, b) => a.score.total - b.score.total, rng);

    return {
      action: best.action,
      choice: best.choice,
      score: this.mode === 'max_damage' ? best.score.totalDamage : best.score.total,
      scoreBreakdown: best.score,
      scores: scored,
    };
  }
}

module.exports = {
  HeuristicSelector,
  compareMaxDamageScores,
  hpFractionFromCondition,
  normalizeLegalActions,
  scoreMaxDamageAction,
  scoreTacticalAction,
};
