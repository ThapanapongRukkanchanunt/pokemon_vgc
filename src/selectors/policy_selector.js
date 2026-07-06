const {HeuristicSelector, normalizeLegalActions} = require('./heuristic_selector');

function choiceText(action) {
  return typeof action === 'string' ? action : action.choice;
}

function scoreByChoice(modelScores) {
  const scores = new Map();
  for (const row of modelScores || []) {
    if (row && Number.isFinite(row.score)) scores.set(row.choice, row.score);
  }
  return scores;
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

function normalizedScoreMap(rows, valueFn) {
  const values = rows.map(valueFn).filter(Number.isFinite);
  if (!values.length) return new Map();
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / values.length;
  const stddev = Math.sqrt(variance) || 1;
  const normalized = new Map();
  for (const row of rows) {
    const value = valueFn(row);
    if (Number.isFinite(value)) normalized.set(row.choice, (value - mean) / stddev);
  }
  return normalized;
}

class PolicySelector {
  constructor() {
    this.name = 'policy_selector';
  }

  choose({request, legalActions, modelScores, rng}) {
    if (request.wait) return null;
    const actions = normalizeLegalActions(legalActions);
    if (!actions.length) return null;

    const priors = scoreByChoice(modelScores);
    const scored = actions.map(action => ({
      action,
      choice: choiceText(action),
      score: priors.has(choiceText(action)) ? priors.get(choiceText(action)) : -Infinity,
    }));
    let best = pickBest(scored, rng);
    if (!best) {
      const fallbackAction = rng ? rng.pick(actions) : actions[0];
      best = {
        action: fallbackAction,
        choice: choiceText(fallbackAction),
        score: 0,
      };
    }
    return {
      action: best.action,
      choice: best.choice,
      score: best.score,
      scores: scored,
    };
  }
}

class HybridPolicyHeuristicSelector {
  constructor({
    formatId = 'vgc',
    policyWeight = 1,
    heuristicWeight = 1,
    heuristicSelector = null,
  } = {}) {
    this.name = 'hybrid_policy_heuristic_selector';
    this.policyWeight = policyWeight;
    this.heuristicWeight = heuristicWeight;
    this.heuristicSelector = heuristicSelector || new HeuristicSelector({formatId, mode: 'tactical'});
  }

  choose({side, request, legalActions, modelScores, battleState, rng, dex = null}) {
    if (request.wait) return null;
    const actions = normalizeLegalActions(legalActions);
    if (!actions.length) return null;

    const heuristicRows = this.heuristicSelector.scoreActions({
      side,
      request,
      legalActions: actions,
      battleState,
      dex,
    });
    const policyRows = (modelScores || []).filter(row => Number.isFinite(row.score));
    const policyScores = normalizedScoreMap(policyRows, row => row.score);
    const heuristicScores = normalizedScoreMap(heuristicRows, row => row.score.total);

    if (!policyScores.size) {
      return this.heuristicSelector.choose({side, request, legalActions: actions, battleState, rng, dex});
    }

    const heuristicByChoice = new Map(heuristicRows.map(row => [row.choice, row.score]));
    const scored = actions.map(action => {
      const choice = choiceText(action);
      const policyScore = policyScores.has(choice) ? policyScores.get(choice) : -4;
      const heuristicScore = heuristicScores.has(choice) ? heuristicScores.get(choice) : 0;
      return {
        action,
        choice,
        policyScore,
        heuristicScore,
        heuristicBreakdown: heuristicByChoice.get(choice),
        score: this.policyWeight * policyScore + this.heuristicWeight * heuristicScore,
      };
    });
    const best = pickBest(scored, rng);
    return {
      action: best.action,
      choice: best.choice,
      score: best.score,
      scoreBreakdown: {
        policy: best.policyScore,
        heuristic: best.heuristicScore,
        heuristicTerms: best.heuristicBreakdown,
      },
      scores: scored,
    };
  }
}

module.exports = {
  HybridPolicyHeuristicSelector,
  PolicySelector,
};
