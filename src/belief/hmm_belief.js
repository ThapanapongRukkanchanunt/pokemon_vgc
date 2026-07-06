const {
  isTeamPreviewRequest,
  sideOpponent,
  switchableSlots,
} = require('../agents/action_utils');
const {hpFractionFromCondition} = require('../selectors/heuristic_selector');

const HIDDEN_STATES = [
  'neutral',
  'pressure',
  'defensive',
  'speed_control',
  'pivot',
  'endgame',
];

const SPEED_CONTROL_MOVE_IDS = new Set([
  'afteryou',
  'bulldoze',
  'electroweb',
  'glare',
  'icywind',
  'quash',
  'rocktomb',
  'scaryface',
  'stringshot',
  'tailwind',
  'thunderwave',
  'trickroom',
]);

const PIVOT_MOVE_IDS = new Set([
  'batonpass',
  'flipturn',
  'partingshot',
  'shedtail',
  'uturn',
  'voltswitch',
]);

const PROTECT_MOVE_IDS = new Set([
  'banefulbunker',
  'burningbulwark',
  'detect',
  'kingsshield',
  'protect',
  'silktrap',
  'spikyshield',
]);

const TRANSITIONS = {
  neutral: {
    neutral: 0.56,
    pressure: 0.12,
    defensive: 0.08,
    speed_control: 0.10,
    pivot: 0.09,
    endgame: 0.05,
  },
  pressure: {
    neutral: 0.12,
    pressure: 0.54,
    defensive: 0.14,
    speed_control: 0.05,
    pivot: 0.08,
    endgame: 0.07,
  },
  defensive: {
    neutral: 0.15,
    pressure: 0.15,
    defensive: 0.45,
    speed_control: 0.05,
    pivot: 0.12,
    endgame: 0.08,
  },
  speed_control: {
    neutral: 0.12,
    pressure: 0.10,
    defensive: 0.05,
    speed_control: 0.55,
    pivot: 0.08,
    endgame: 0.10,
  },
  pivot: {
    neutral: 0.15,
    pressure: 0.12,
    defensive: 0.12,
    speed_control: 0.06,
    pivot: 0.45,
    endgame: 0.10,
  },
  endgame: {
    neutral: 0.05,
    pressure: 0.15,
    defensive: 0.12,
    speed_control: 0.04,
    pivot: 0.06,
    endgame: 0.58,
  },
};

function toId(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function requestTypeFromRequest(request) {
  if (!request || typeof request !== 'object') return 'other';
  if (isTeamPreviewRequest(request)) return 'team_preview';
  if (Array.isArray(request.forceSwitch) && request.forceSwitch.some(Boolean)) return 'force_switch';
  if (Array.isArray(request.active) && request.active.length) return 'move';
  if (request.wait) return 'wait';
  return 'other';
}

function distributionFromEntries(entries) {
  const distribution = {};
  for (const state of HIDDEN_STATES) distribution[state] = entries?.[state] || 0;
  return normalizeDistribution(distribution);
}

function uniformDistribution() {
  const value = 1 / HIDDEN_STATES.length;
  return Object.fromEntries(HIDDEN_STATES.map(state => [state, value]));
}

function normalizeDistribution(distribution) {
  const clipped = {};
  let total = 0;
  for (const state of HIDDEN_STATES) {
    const value = Math.max(0, Number(distribution?.[state]) || 0);
    clipped[state] = value;
    total += value;
  }
  if (!total) return uniformDistribution();
  return Object.fromEntries(HIDDEN_STATES.map(state => [state, clipped[state] / total]));
}

function topState(probabilities) {
  let bestState = HIDDEN_STATES[0];
  let bestValue = -Infinity;
  for (const state of HIDDEN_STATES) {
    const value = probabilities[state] || 0;
    if (value > bestValue) {
      bestState = state;
      bestValue = value;
    }
  }
  return bestState;
}

function roundDistribution(probabilities, digits = 4) {
  const factor = Math.pow(10, digits);
  return Object.fromEntries(HIDDEN_STATES.map(state => [
    state,
    Math.round((probabilities[state] || 0) * factor) / factor,
  ]));
}

function activePokemon(request) {
  return (request.side?.pokemon || []).filter(pokemon => pokemon.active);
}

function knownOwnAliveCount(request) {
  return (request.side?.pokemon || []).filter(pokemon => {
    return !/\bfnt\b/.test(pokemon?.condition || '');
  }).length;
}

function countLowHp(entries, threshold) {
  return entries.filter(entry => {
    const hp = hpFractionFromCondition(entry?.condition);
    return hp != null && hp > 0 && hp <= threshold;
  }).length;
}

function countFainted(entries) {
  return entries.filter(entry => !!entry?.fainted || /\bfnt\b/.test(entry?.condition || '')).length;
}

function moveObservation(request) {
  let speedControlMoves = 0;
  let pivotMoves = 0;
  let protectMoves = 0;
  let damagingMoves = 0;

  for (const activeData of request.active || []) {
    for (const moveSlot of activeData?.moves || []) {
      if (moveSlot.disabled) continue;
      const moveId = toId(moveSlot.id || moveSlot.move);
      if (!moveId) continue;
      if (SPEED_CONTROL_MOVE_IDS.has(moveId)) speedControlMoves += 1;
      if (PIVOT_MOVE_IDS.has(moveId)) pivotMoves += 1;
      if (PROTECT_MOVE_IDS.has(moveId)) protectMoves += 1;
      if (moveSlot.target !== 'self' && !PROTECT_MOVE_IDS.has(moveId)) damagingMoves += 1;
    }
  }

  return {
    speedControlMoves,
    pivotMoves,
    protectMoves,
    damagingMoves,
  };
}

function observationFromBattle({side, request, battleState}) {
  const requestType = requestTypeFromRequest(request);
  const ownActive = activePokemon(request);
  const foeSide = sideOpponent(side);
  const foePublicActive = battleState?.publicState?.active?.[foeSide] || [];
  const ownPublicActive = battleState?.publicState?.active?.[side] || [];
  const moves = moveObservation(request || {});
  const ownAlive = knownOwnAliveCount(request || {});
  const forceSwitch = Array.isArray(request?.forceSwitch) && request.forceSwitch.some(Boolean);
  const switchChoices = switchableSlots(request || {}).length;

  return {
    turn: Number(battleState?.turns) || 0,
    requestType,
    ownAlive,
    ownActiveCount: ownActive.length,
    ownActiveLowHp: countLowHp(ownActive, 0.5),
    ownActiveCriticalHp: countLowHp(ownActive, 0.25),
    ownPublicFainted: countFainted(ownPublicActive),
    foeVisibleCount: foePublicActive.filter(active => active && !active.fainted).length,
    foeVisibleLowHp: countLowHp(foePublicActive, 0.5),
    foeVisibleCriticalHp: countLowHp(foePublicActive, 0.25),
    foeVisibleFainted: countFainted(foePublicActive),
    forceSwitch,
    switchChoices,
    ...moves,
  };
}

function bool(value) {
  return value ? 1 : 0;
}

function bounded(value, min = 0.05) {
  return Math.max(min, value);
}

function emissionLikelihood(state, observation) {
  const turn = Number(observation.turn) || 0;
  const lowHp = Number(observation.ownActiveLowHp) || 0;
  const criticalHp = Number(observation.ownActiveCriticalHp) || 0;
  const forceSwitch = bool(observation.forceSwitch);
  const speedMoves = Number(observation.speedControlMoves) || 0;
  const pivotMoves = Number(observation.pivotMoves) || 0;
  const protectMoves = Number(observation.protectMoves) || 0;
  const switchChoices = Number(observation.switchChoices) || 0;
  const damagingMoves = Number(observation.damagingMoves) || 0;
  const ownAlive = Number(observation.ownAlive) || 0;
  const foeLowHp = Number(observation.foeVisibleLowHp) || 0;
  const foeCriticalHp = Number(observation.foeVisibleCriticalHp) || 0;
  const lateGame = bool(turn >= 6 || (ownAlive > 0 && ownAlive <= 2));
  const earlyGame = bool(turn <= 2);
  const hasSpeed = bool(speedMoves > 0);
  const hasPivot = bool(pivotMoves > 0 || switchChoices > 0 || forceSwitch);

  if (observation.requestType === 'team_preview') {
    return state === 'neutral' ? 2.4 : (state === 'speed_control' ? 1.1 : 0.75);
  }

  switch (state) {
    case 'neutral':
      return bounded(1.35 + 0.25 * earlyGame - 0.16 * lowHp - 0.12 * lateGame);
    case 'pressure':
      return bounded(0.75 + 0.70 * lowHp + 0.90 * criticalHp + 1.20 * forceSwitch +
        0.20 * damagingMoves + 0.20 * lateGame);
    case 'defensive':
      return bounded(0.70 + 0.55 * lowHp + 0.75 * criticalHp + 0.95 * forceSwitch +
        0.28 * protectMoves + 0.12 * switchChoices);
    case 'speed_control':
      return bounded(0.60 + 2.20 * hasSpeed + 0.35 * speedMoves + 0.35 * earlyGame);
    case 'pivot':
      return bounded(0.65 + 1.15 * hasPivot + 0.35 * pivotMoves + 0.10 * switchChoices +
        0.35 * forceSwitch);
    case 'endgame':
      return bounded(0.50 + 1.65 * lateGame + 0.12 * turn + 0.35 * foeLowHp +
        0.45 * foeCriticalHp);
    default:
      return 1;
  }
}

function emissionLikelihoods(observation) {
  return Object.fromEntries(HIDDEN_STATES.map(state => [state, emissionLikelihood(state, observation)]));
}

function predictDistribution(prior, transitions = TRANSITIONS) {
  const predicted = {};
  for (const state of HIDDEN_STATES) {
    predicted[state] = 0;
    for (const previous of HIDDEN_STATES) {
      predicted[state] += (prior[previous] || 0) * (transitions[previous]?.[state] || 0);
    }
  }
  return predicted;
}

function forwardUpdate(prior, observation, transitions = TRANSITIONS) {
  const normalizedPrior = distributionFromEntries(prior);
  const predicted = predictDistribution(normalizedPrior, transitions);
  const emissions = emissionLikelihoods(observation);
  const posterior = {};
  for (const state of HIDDEN_STATES) {
    posterior[state] = predicted[state] * emissions[state];
  }
  return normalizeDistribution(posterior);
}

class HMMBeliefState {
  constructor({initial = null, transitions = TRANSITIONS} = {}) {
    this.transitions = transitions;
    this.probabilities = initial ? distributionFromEntries(initial) : uniformDistribution();
    this.lastObservation = null;
    this.turn = 0;
  }

  updateFromObservation(observation) {
    this.lastObservation = {...observation};
    this.turn = Number(observation.turn) || this.turn;
    this.probabilities = forwardUpdate(this.probabilities, this.lastObservation, this.transitions);
    return this.snapshot();
  }

  update({side, request, battleState}) {
    return this.updateFromObservation(observationFromBattle({side, request, battleState}));
  }

  snapshot() {
    return {
      turn: this.turn,
      top_state: topState(this.probabilities),
      probabilities: roundDistribution(this.probabilities),
      last_observation: this.lastObservation ? {...this.lastObservation} : null,
    };
  }
}

module.exports = {
  HIDDEN_STATES,
  HMMBeliefState,
  PIVOT_MOVE_IDS,
  PROTECT_MOVE_IDS,
  SPEED_CONTROL_MOVE_IDS,
  TRANSITIONS,
  emissionLikelihoods,
  forwardUpdate,
  normalizeDistribution,
  observationFromBattle,
  topState,
  uniformDistribution,
};
