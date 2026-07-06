const {isTeamPreviewRequest, teamPreviewChoice} = require('../agents/action_utils');
const {
  HMMBeliefState,
  PIVOT_MOVE_IDS,
  PROTECT_MOVE_IDS,
  SPEED_CONTROL_MOVE_IDS,
} = require('../belief/hmm_belief');
const {normalizeLegalActions} = require('./heuristic_selector');
const {ShallowSearchSelector} = require('./search_selector');

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

function hasDecisionKind(action, kind) {
  return (action?.decisions || []).some(decision => decision.kind === kind);
}

function countMoveIds(action, moveIds) {
  return (action?.decisions || []).filter(decision => {
    return decision.kind === 'move' && moveIds.has(decision.moveId);
  }).length;
}

function damagingMoveCount(action, dex) {
  return (action?.decisions || []).filter(decision => {
    if (decision.kind !== 'move') return false;
    if (PROTECT_MOVE_IDS.has(decision.moveId)) return false;
    const move = dex?.moves?.get(decision.moveId);
    if (!move?.exists) return true;
    return move.category !== 'Status' && Number(move.basePower || move.damage || 0) !== 0;
  }).length;
}

function priorityMoveCount(action, dex) {
  return (action?.decisions || []).filter(decision => {
    if (decision.kind !== 'move') return false;
    const move = dex?.moves?.get(decision.moveId);
    return move?.exists && move.priority > 0;
  }).length;
}

function beliefAdjustment({action, belief, dex}) {
  const probabilities = belief?.probabilities || {};
  const protectCount = countMoveIds(action, PROTECT_MOVE_IDS);
  const speedCount = countMoveIds(action, SPEED_CONTROL_MOVE_IDS);
  const pivotCount = countMoveIds(action, PIVOT_MOVE_IDS);
  const switchCount = hasDecisionKind(action, 'switch') ? 1 : 0;
  const damageCount = damagingMoveCount(action, dex);
  const priorityCount = priorityMoveCount(action, dex);
  const allOutDamage = damageCount > 0 && !protectCount && !switchCount ? 1 : 0;
  const passive = damageCount === 0 && speedCount === 0 && pivotCount === 0 && !protectCount && !switchCount ? 1 : 0;

  return (
    (probabilities.pressure || 0) *
      (0.42 * protectCount + 0.18 * switchCount - 0.10 * allOutDamage) +
    (probabilities.defensive || 0) *
      (0.35 * protectCount + 0.25 * switchCount - 0.08 * passive) +
    (probabilities.speed_control || 0) *
      (0.48 * speedCount + 0.12 * priorityCount) +
    (probabilities.pivot || 0) *
      (0.40 * (switchCount + pivotCount) - 0.06 * allOutDamage) +
    (probabilities.endgame || 0) *
      (0.24 * damageCount + 0.20 * priorityCount - 0.14 * switchCount - 0.12 * protectCount)
  );
}

class HMMBeliefSelector {
  constructor({
    formatId = 'vgc',
    riskMode = 'balanced',
    maxOpponentActions = 4,
    beliefWeight = 0.35,
    baseSelector = null,
    belief = null,
  } = {}) {
    this.formatId = formatId;
    this.riskMode = riskMode;
    this.beliefWeight = beliefWeight;
    this.belief = belief || new HMMBeliefState();
    this.baseSelector = baseSelector || new ShallowSearchSelector({formatId, riskMode, maxOpponentActions});
    this.name = `hmm_belief_${riskMode}_selector`;
    this.lastDiagnostics = this.belief.snapshot();
  }

  diagnostics() {
    return this.lastDiagnostics;
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

    const belief = this.belief.update({side, request, battleState});
    this.lastDiagnostics = belief;

    if (isTeamPreviewRequest(request)) {
      const preferred = teamPreviewChoice(battleState);
      const action = actions.find(candidate => choiceText(candidate) === preferred) || actions[0];
      return {
        action,
        choice: choiceText(action),
        score: 0,
        beliefApplied: false,
        scoreBreakdown: {belief},
        scores: [],
      };
    }

    const baseSelection = this.baseSelector.choose({
      side,
      request,
      legalActions: actions,
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
    const baseRows = (baseSelection?.scores || []).map(row => ({
      ...row,
      action: row.action,
      choice: row.choice,
      baseScore: numericScore(row.score),
    }));
    if (!baseRows.length) return baseSelection;

    const scored = baseRows.map(row => {
      const adjustment = beliefAdjustment({action: row.action, belief, dex});
      return {
        ...row,
        score: row.baseScore + this.beliefWeight * adjustment,
        baseScore: row.baseScore,
        beliefAdjustment: adjustment,
        beliefWeight: this.beliefWeight,
        beliefTopState: belief.top_state,
      };
    });
    const best = pickBest(scored, rng) || scored[0];
    return {
      action: best.action,
      choice: best.choice,
      score: best.score,
      beliefApplied: true,
      riskMode: this.riskMode,
      scoreBreakdown: {
        base_choice: baseSelection?.choice || null,
        base_score: best.baseScore,
        belief_adjustment: best.beliefAdjustment,
        belief_weight: this.beliefWeight,
        belief,
      },
      scores: scored,
      baseSelection,
    };
  }
}

module.exports = {
  HMMBeliefSelector,
  beliefAdjustment,
};
