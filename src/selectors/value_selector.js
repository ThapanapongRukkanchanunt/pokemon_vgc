const {normalizeLegalActions} = require('./heuristic_selector');

function choiceText(action) {
  return typeof action === 'string' ? action : action.choice;
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

function valueByChoice(valueScores) {
  const scores = new Map();
  for (const row of valueScores || []) {
    if (row && Number.isFinite(row.value)) scores.set(row.choice, row.value);
    else if (row && Number.isFinite(row.score)) scores.set(row.choice, row.score);
  }
  return scores;
}

class ValueSelector {
  constructor() {
    this.name = 'value_selector';
  }

  choose({request, legalActions, valueScores, rng}) {
    if (request.wait) return null;
    const actions = normalizeLegalActions(legalActions);
    if (!actions.length) return null;

    const values = valueByChoice(valueScores);
    const scored = actions.map(action => {
      const choice = choiceText(action);
      const value = values.has(choice) ? values.get(choice) : -Infinity;
      return {
        action,
        choice,
        value,
        score: value,
      };
    });

    let best = pickBest(scored, rng);
    if (!best) {
      const fallbackAction = rng ? rng.pick(actions) : actions[0];
      best = {
        action: fallbackAction,
        choice: choiceText(fallbackAction),
        value: 0.5,
        score: 0.5,
      };
    }
    return {
      action: best.action,
      choice: best.choice,
      score: best.score,
      scoreBreakdown: {value: best.value},
      scores: scored,
    };
  }
}

module.exports = {
  ValueSelector,
};
