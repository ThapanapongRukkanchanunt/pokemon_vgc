const {isTeamPreviewRequest, teamPreviewChoice} = require('../agents/action_utils');
const {HeuristicSelector, normalizeLegalActions} = require('./heuristic_selector');

const RISK_MODE_WEIGHTS = {
  stable: {
    policy: 0.85,
    value: 0.35,
    tactic: 0.75,
    safety: 0.90,
  },
  balanced: {
    policy: 1.00,
    value: 0.20,
    tactic: 1.00,
    safety: 0.25,
  },
  comeback: {
    policy: 0.75,
    value: 0.05,
    tactic: 1.35,
    safety: 0.10,
  },
  closing: {
    policy: 0.90,
    value: 0.50,
    tactic: 0.85,
    safety: 0.75,
  },
};

function choiceText(action) {
  return typeof action === 'string' ? action : action.choice;
}

function finiteOrNull(value) {
  return Number.isFinite(value) ? value : null;
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
    const choice = row.choice;
    for (const key of keys) {
      if (Number.isFinite(row[key])) {
        scores.set(choice, row[key]);
        break;
      }
    }
  }
  return scores;
}

function logPolicyMap(actions, modelScores) {
  const rawScores = scoreMap(modelScores, ['score', 'logit']);
  const scored = actions.map(action => {
    const choice = choiceText(action);
    return {
      choice,
      score: finiteOrNull(rawScores.get(choice)),
    };
  });
  const finiteScores = scored.map(row => row.score).filter(Number.isFinite);
  if (!finiteScores.length) return new Map(actions.map(action => [choiceText(action), 0]));

  const fallback = Math.min(...finiteScores) - 8;
  const completed = scored.map(row => Number.isFinite(row.score) ? row.score : fallback);
  const maxScore = Math.max(...completed);
  const logDenominator = maxScore +
    Math.log(completed.reduce((sum, score) => sum + Math.exp(score - maxScore), 0));
  const result = new Map();
  scored.forEach((row, index) => {
    result.set(row.choice, completed[index] - logDenominator);
  });
  return result;
}

function valueMap(actions, valueScores) {
  const rawValues = scoreMap(valueScores, ['value', 'score']);
  return new Map(actions.map(action => {
    const choice = choiceText(action);
    const value = rawValues.get(choice);
    return [choice, Number.isFinite(value) ? value : 0.5];
  }));
}

function splitHeuristicScore(score) {
  const total = Number.isFinite(score?.total) ? score.total :
    (Number.isFinite(score?.totalDamage) ? score.totalDamage : 0);
  const safety =
    (Number(score?.protectSafety) || 0) +
    (Number(score?.switchSafety) || 0) +
    (Number(score?.immunityAvoidance) || 0) +
    (Number(score?.passPenalty) || 0);
  return {
    total,
    tactic: total,
    safety,
  };
}

function normalizeComponent(actions, rawByChoice) {
  const rawRows = actions.map(action => {
    const choice = choiceText(action);
    const value = rawByChoice.get(choice);
    return {
      choice,
      value: Number.isFinite(value) ? value : 0,
    };
  });
  const mean = rawRows.reduce((sum, row) => sum + row.value, 0) / (rawRows.length || 1);
  const variance = rawRows.reduce((sum, row) => sum + Math.pow(row.value - mean, 2), 0) /
    (rawRows.length || 1);
  const stddev = Math.sqrt(variance) || 1;
  return new Map(rawRows.map(row => [row.choice, (row.value - mean) / stddev]));
}

function mergeWeights(modeWeights, overrideWeights) {
  return {
    ...modeWeights,
    ...(overrideWeights || {}),
  };
}

class PolicyValueRiskSelector {
  constructor({
    formatId = 'vgc',
    riskMode = 'balanced',
    weights = null,
    heuristicSelector = null,
  } = {}) {
    if (!RISK_MODE_WEIGHTS[riskMode]) {
      throw new Error(`Unknown risk mode: ${riskMode}`);
    }
    this.formatId = formatId;
    this.riskMode = riskMode;
    this.weights = mergeWeights(RISK_MODE_WEIGHTS[riskMode], weights);
    this.heuristicSelector = heuristicSelector || new HeuristicSelector({formatId, mode: 'tactical'});
    this.name = `policy_value_risk_${riskMode}_selector`;
  }

  choose({side, request, legalActions, modelScores, valueScores, battleState, rng, dex = null}) {
    if (request.wait) return null;
    const actions = normalizeLegalActions(legalActions);
    if (!actions.length) return null;

    if (isTeamPreviewRequest(request)) {
      const preferred = teamPreviewChoice(battleState);
      const action = actions.find(candidate => choiceText(candidate) === preferred) || actions[0];
      return {action, choice: choiceText(action), score: 0, scores: []};
    }

    const hasPolicyScores = (modelScores || []).some(row => row && Number.isFinite(row.score));
    const hasValueScores = (valueScores || []).some(row => row &&
      (Number.isFinite(row.value) || Number.isFinite(row.score)));
    if (!hasPolicyScores && !hasValueScores) {
      return this.heuristicSelector.choose({side, request, legalActions: actions, battleState, rng, dex});
    }

    const heuristicRows = this.heuristicSelector.scoreActions({
      side,
      request,
      legalActions: actions,
      battleState,
      dex,
    });
    const heuristicByChoice = new Map(heuristicRows.map(row => [row.choice, {
      raw: row.score,
      split: splitHeuristicScore(row.score),
    }]));

    const rawPolicy = logPolicyMap(actions, modelScores);
    const rawValue = valueMap(actions, valueScores);
    const rawTactic = new Map();
    const rawSafety = new Map();
    for (const action of actions) {
      const choice = choiceText(action);
      const split = heuristicByChoice.get(choice)?.split || {tactic: 0, safety: 0};
      rawTactic.set(choice, split.tactic);
      rawSafety.set(choice, split.safety);
    }

    const normalized = {
      policy: normalizeComponent(actions, rawPolicy),
      value: normalizeComponent(actions, rawValue),
      tactic: normalizeComponent(actions, rawTactic),
      safety: normalizeComponent(actions, rawSafety),
    };

    const scored = actions.map(action => {
      const choice = choiceText(action);
      const components = {
        policy: normalized.policy.get(choice) || 0,
        value: normalized.value.get(choice) || 0,
        tactic: normalized.tactic.get(choice) || 0,
        safety: normalized.safety.get(choice) || 0,
      };
      const weighted = {
        policy: this.weights.policy * components.policy,
        value: this.weights.value * components.value,
        tactic: this.weights.tactic * components.tactic,
        safety: this.weights.safety * components.safety,
      };
      const score = weighted.policy + weighted.value + weighted.tactic + weighted.safety;
      return {
        action,
        choice,
        score,
        components,
        weighted,
        rawComponents: {
          policyLogProb: rawPolicy.get(choice),
          value: rawValue.get(choice),
          tactic: rawTactic.get(choice),
          safety: rawSafety.get(choice),
          heuristic: heuristicByChoice.get(choice)?.raw,
        },
      };
    });

    const best = pickBest(scored, rng) || scored[0];
    return {
      action: best.action,
      choice: best.choice,
      score: best.score,
      riskMode: this.riskMode,
      weights: this.weights,
      scoreBreakdown: {
        components: best.components,
        weighted: best.weighted,
        raw: best.rawComponents,
      },
      scores: scored,
    };
  }
}

module.exports = {
  PolicyValueRiskSelector,
  RISK_MODE_WEIGHTS,
};
