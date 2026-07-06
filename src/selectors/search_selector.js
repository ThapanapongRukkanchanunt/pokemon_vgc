const {
  isTeamPreviewRequest,
  sideOpponent,
  teamPreviewChoice,
} = require('../agents/action_utils');
const {normalizeLegalActions} = require('./heuristic_selector');
const {PolicyValueRiskSelector} = require('./risk_aware_selector');

const PROTECT_MOVE_IDS = new Set([
  'banefulbunker',
  'burningbulwark',
  'detect',
  'kingsshield',
  'protect',
  'silktrap',
  'spikyshield',
]);

function choiceText(action) {
  return typeof action === 'string' ? action : action.choice;
}

function numericScore(value) {
  if (Number.isFinite(value)) return value;
  if (Number.isFinite(value?.total)) return value.total;
  if (Number.isFinite(value?.totalDamage)) return value.totalDamage;
  return 0;
}

function pickBest(scored, rng) {
  let bestScore = -Infinity;
  let bestRows = [];
  for (const row of scored) {
    if (row.score > bestScore) {
      bestScore = row.score;
      bestRows = [row];
    } else if (row.score === bestScore) {
      bestRows.push(row);
    }
  }
  if (!bestRows.length || bestScore === -Infinity) return null;
  return rng && bestRows.length > 1 ? rng.pick(bestRows) : bestRows[0];
}

function scoreMap(rows, keys) {
  const scores = new Map();
  for (const row of rows || []) {
    if (!row) continue;
    for (const key of keys) {
      if (Number.isFinite(row[key])) {
        scores.set(row.choice, row[key]);
        break;
      }
    }
  }
  return scores;
}

function normalizeValues(rows, valueFn) {
  const values = rows.map(valueFn).filter(Number.isFinite);
  if (!values.length) return new Map(rows.map(row => [row.choice, 0]));
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / values.length;
  const stddev = Math.sqrt(variance) || 1;
  return new Map(rows.map(row => {
    const value = valueFn(row);
    return [row.choice, Number.isFinite(value) ? (value - mean) / stddev : 0];
  }));
}

function topRows(rows, limit) {
  return rows
    .slice()
    .sort((a, b) => b.responsePressure - a.responsePressure || a.choice.localeCompare(b.choice))
    .slice(0, Math.max(1, limit));
}

function actionDecisions(action) {
  return action?.decisions || [];
}

function hasDecisionKind(action, kind) {
  return actionDecisions(action).some(decision => decision.kind === kind);
}

function protectSlots(action) {
  const slots = new Set();
  for (const decision of actionDecisions(action)) {
    if (decision.kind === 'move' && PROTECT_MOVE_IDS.has(decision.moveId)) {
      slots.add(decision.activeIndex);
    }
  }
  return slots;
}

function targetedFoeSlots(action) {
  const slots = new Set();
  let includesUntargetedMove = false;
  for (const decision of actionDecisions(action)) {
    if (decision.kind !== 'move') continue;
    const target = Number(decision.target);
    if (Number.isInteger(target) && target > 0) {
      slots.add(target - 1);
    } else {
      includesUntargetedMove = true;
    }
  }
  return {slots, includesUntargetedMove};
}

function damagingMoveCount(action) {
  return actionDecisions(action).filter(decision => {
    return decision.kind === 'move' && !PROTECT_MOVE_IDS.has(decision.moveId);
  }).length;
}

function responseMitigation(ownAction) {
  let factor = 1;
  if (hasDecisionKind(ownAction, 'switch')) factor -= 0.25;
  if (protectSlots(ownAction).size) factor -= 0.45;
  if (!damagingMoveCount(ownAction) && !hasDecisionKind(ownAction, 'switch')) factor += 0.1;
  return Math.max(0.25, factor);
}

function pairInteraction(ownAction, opponentAction) {
  let interaction = 0;
  const ownTargets = targetedFoeSlots(ownAction);
  const opponentProtects = protectSlots(opponentAction);
  const opponentSwitches = hasDecisionKind(opponentAction, 'switch');
  const ownProtects = protectSlots(ownAction).size > 0;
  const opponentDamagingMoves = damagingMoveCount(opponentAction);

  if (ownProtects && opponentDamagingMoves > 0) interaction += 0.45;
  if (hasDecisionKind(ownAction, 'switch') && opponentDamagingMoves > 0) interaction += 0.18;
  if (opponentSwitches && damagingMoveCount(ownAction) > 0) interaction -= 0.16;
  for (const slot of ownTargets.slots) {
    if (opponentProtects.has(slot)) interaction -= 0.55;
  }
  if (ownTargets.includesUntargetedMove && opponentProtects.size) interaction -= 0.12;
  return interaction;
}

function battleStateForSide(battleState, side) {
  if (!battleState) return battleState;
  const team = battleState.teams?.[side] || battleState.team;
  const leadMode = battleState.leadModes?.[side] || battleState.leadMode;
  return {
    ...battleState,
    team,
    leadMode,
  };
}

class ShallowSearchSelector {
  constructor({
    formatId = 'vgc',
    riskMode = 'balanced',
    maxOpponentActions = 4,
    baseSelector = null,
    opponentSelector = null,
    responsePenaltyWeight = 0.55,
    valueSwingWeight = 0.35,
    searchWeight = 1,
  } = {}) {
    this.formatId = formatId;
    this.riskMode = riskMode;
    this.maxOpponentActions = maxOpponentActions;
    this.responsePenaltyWeight = responsePenaltyWeight;
    this.valueSwingWeight = valueSwingWeight;
    this.searchWeight = searchWeight;
    this.baseSelector = baseSelector || new PolicyValueRiskSelector({formatId, riskMode});
    this.opponentSelector = opponentSelector || new PolicyValueRiskSelector({formatId, riskMode: 'stable'});
    this.name = `shallow_search_${riskMode}_selector`;
  }

  scoreOpponentResponses({
    side,
    opponentRequest,
    opponentLegalActions,
    opponentModelScores,
    opponentValueScores,
    battleState,
    rng,
    dex,
  }) {
    const opponentSide = sideOpponent(side);
    const actions = normalizeLegalActions(opponentLegalActions);
    if (!opponentRequest || !actions.length) return [];

    const selection = this.opponentSelector.choose({
      side: opponentSide,
      request: opponentRequest,
      legalActions: actions,
      modelScores: opponentModelScores,
      valueScores: opponentValueScores,
      battleState: battleStateForSide(battleState, opponentSide),
      rng,
      dex,
    });
    const rows = (selection?.scores || []).map(row => ({
      action: row.action,
      choice: row.choice,
      responsePressure: numericScore(row.score),
      rawScore: row.score,
    }));
    const valueByChoice = scoreMap(opponentValueScores, ['value', 'score']);
    const normalizedPressure = normalizeValues(rows, row => row.responsePressure);
    const normalizedValue = normalizeValues(rows, row => {
      const value = valueByChoice.get(row.choice);
      return Number.isFinite(value) ? value : 0.5;
    });
    return rows.map(row => ({
      ...row,
      normalizedPressure: normalizedPressure.get(row.choice) || 0,
      normalizedValue: normalizedValue.get(row.choice) || 0,
      value: valueByChoice.has(row.choice) ? valueByChoice.get(row.choice) : 0.5,
    }));
  }

  choose({
    side,
    request,
    legalActions,
    modelScores,
    valueScores,
    opponentRequest = null,
    opponentLegalActions = null,
    opponentModelScores = null,
    opponentValueScores = null,
    battleState,
    rng,
    dex = null,
  }) {
    if (request.wait) return null;
    const actions = normalizeLegalActions(legalActions);
    if (!actions.length) return null;

    if (isTeamPreviewRequest(request)) {
      const preferred = teamPreviewChoice(battleState);
      const action = actions.find(candidate => choiceText(candidate) === preferred) || actions[0];
      return {
        action,
        choice: choiceText(action),
        score: 0,
        searchApplied: false,
        scores: [],
      };
    }

    const baseSelection = this.baseSelector.choose({
      side,
      request,
      legalActions: actions,
      modelScores,
      valueScores,
      battleState: battleStateForSide(battleState, side),
      rng,
      dex,
    });
    const baseRows = (baseSelection?.scores || []).map(row => ({
      action: row.action,
      choice: row.choice,
      baseScore: numericScore(row.score),
      baseRawScore: row.score,
      baseBreakdown: row,
    }));
    if (!baseRows.length) return baseSelection;

    const valueByChoice = scoreMap(valueScores, ['value', 'score']);
    const ownValueNorm = normalizeValues(baseRows, row => {
      const value = valueByChoice.get(row.choice);
      return Number.isFinite(value) ? value : 0.5;
    });
    const opponentRows = topRows(this.scoreOpponentResponses({
      side,
      opponentRequest,
      opponentLegalActions,
      opponentModelScores,
      opponentValueScores,
      battleState,
      rng,
      dex,
    }), this.maxOpponentActions);

    if (!opponentRows.length) {
      const fallback = pickBest(baseRows.map(row => ({
        action: row.action,
        choice: row.choice,
        score: row.baseScore,
        baseScore: row.baseScore,
        searchApplied: false,
      })), rng);
      return {
        action: fallback.action,
        choice: fallback.choice,
        score: fallback.score,
        searchApplied: false,
        scoreBreakdown: {
          base: fallback.baseScore,
          reason: 'no_opponent_request',
        },
        scores: baseRows.map(row => ({
          action: row.action,
          choice: row.choice,
          score: row.baseScore,
          baseScore: row.baseScore,
          searchApplied: false,
        })),
      };
    }

    const scored = baseRows.map(row => {
      const ownValue = valueByChoice.has(row.choice) ? valueByChoice.get(row.choice) : 0.5;
      const ownNorm = ownValueNorm.get(row.choice) || 0;
      let worstPair = null;
      for (const opponent of opponentRows) {
        const mitigation = responseMitigation(row.action);
        const interaction = pairInteraction(row.action, opponent.action);
        const valueSwing = ownNorm - opponent.normalizedValue;
        const responsePenalty = this.responsePenaltyWeight * opponent.normalizedPressure * mitigation;
        const pairScore = row.baseScore +
          this.valueSwingWeight * valueSwing +
          interaction -
          responsePenalty;
        const pair = {
          opponent_choice: opponent.choice,
          score: pairScore,
          opponent_pressure: opponent.normalizedPressure,
          opponent_value: opponent.value,
          mitigation,
          interaction,
          value_swing: valueSwing,
          response_penalty: responsePenalty,
        };
        if (!worstPair || pair.score < worstPair.score) worstPair = pair;
      }
      const searchScore = worstPair ? worstPair.score : row.baseScore;
      const score = (1 - this.searchWeight) * row.baseScore + this.searchWeight * searchScore;
      return {
        action: row.action,
        choice: row.choice,
        score,
        baseScore: row.baseScore,
        searchScore,
        ownValue,
        ownValueNormalized: ownNorm,
        worstPair,
        sampledOpponentActions: opponentRows.length,
        searchApplied: true,
        baseBreakdown: row.baseBreakdown,
      };
    });

    const best = pickBest(scored, rng) || scored[0];
    return {
      action: best.action,
      choice: best.choice,
      score: best.score,
      searchApplied: true,
      riskMode: this.riskMode,
      scoreBreakdown: {
        base: best.baseScore,
        search: best.searchScore,
        own_value: best.ownValue,
        own_value_normalized: best.ownValueNormalized,
        worst_pair: best.worstPair,
        sampled_opponent_actions: best.sampledOpponentActions,
      },
      scores: scored,
      opponentScores: opponentRows,
    };
  }
}

module.exports = {
  ShallowSearchSelector,
  pairInteraction,
  responseMitigation,
};
