const fs = require('node:fs');
const path = require('node:path');
const {
  ENCODER_VERSION,
  contextFromExample,
  featurizeChoice,
  legalChoiceText,
  stateFeaturesFromContext,
} = require('./feature_encoder');

const MODEL_TYPE = 'hashed_linear_action_ranker';
const MODEL_VERSION = 1;

function createEmptyModel({
  featureDim = 8192,
  encoderOptions = {includeStateSignature: true},
  metadata = {},
} = {}) {
  return {
    modelType: MODEL_TYPE,
    modelVersion: MODEL_VERSION,
    encoderVersion: ENCODER_VERSION,
    featureDim,
    encoderOptions,
    weights: new Float64Array(featureDim),
    metadata,
  };
}

function scoreEntries(weights, entries) {
  let score = 0;
  for (const {index, value} of entries) score += weights[index] * value;
  return score;
}

function softmax(scores) {
  const maxScore = Math.max(...scores);
  const exps = scores.map(score => Math.exp(score - maxScore));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map(value => value / total);
}

function candidateEntriesForChoices(model, context, legalActions) {
  const stateFeatures = stateFeaturesFromContext(context, model.encoderOptions);
  return legalActions.map(action => {
    const choice = legalChoiceText(action);
    return {
      choice,
      entries: featurizeChoice({
        context,
        choice,
        featureDim: model.featureDim,
        encoderOptions: model.encoderOptions,
        stateFeatures,
      }),
    };
  });
}

function scoreChoices(model, context, legalActions) {
  return candidateEntriesForChoices(model, context, legalActions).map(candidate => ({
    choice: candidate.choice,
    score: scoreEntries(model.weights, candidate.entries),
  }));
}

function predictChoice(model, context, legalActions, rng = null) {
  if (!legalActions.length) return null;
  const rows = scoreChoices(model, context, legalActions);
  let bestScore = -Infinity;
  let bestRows = [];
  for (const row of rows) {
    if (row.score > bestScore) {
      bestScore = row.score;
      bestRows = [row];
    } else if (row.score === bestScore) {
      bestRows.push(row);
    }
  }
  const picked = rng && bestRows.length > 1 ? rng.pick(bestRows) : bestRows[0];
  return {
    choice: picked.choice,
    score: picked.score,
    scores: rows,
  };
}

function labelIndexForExample(example) {
  if (Number.isInteger(example.label_action_index) &&
      example.legal_actions[example.label_action_index] === example.label_action) {
    return example.label_action_index;
  }
  return example.legal_actions.indexOf(example.label_action);
}

function prepareExample(model, example) {
  const labelIndex = labelIndexForExample(example);
  if (labelIndex < 0) throw new Error(`${example.example_id || 'example'} label is not in legal_actions`);

  const context = contextFromExample(example);
  return {
    example_id: example.example_id,
    request_type: example.request_type || 'unknown',
    labelIndex,
    candidates: candidateEntriesForChoices(model, context, example.legal_actions),
  };
}

function trainOnPreparedExample(model, prepared, {learningRate = 0.1, l2 = 0} = {}) {
  const candidates = prepared.candidates;
  const scores = candidates.map(candidate => scoreEntries(model.weights, candidate.entries));
  const probabilities = softmax(scores);
  const loss = -Math.log(Math.max(probabilities[prepared.labelIndex], 1e-12));
  let bestIndex = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[bestIndex]) bestIndex = i;
  }

  if (l2 > 0) {
    const touched = new Set();
    for (const candidate of candidates) {
      for (const {index} of candidate.entries) touched.add(index);
    }
    for (const index of touched) model.weights[index] *= Math.max(0, 1 - learningRate * l2);
  }

  for (let i = 0; i < candidates.length; i++) {
    const coefficient = probabilities[i] - (i === prepared.labelIndex ? 1 : 0);
    if (coefficient === 0) continue;
    for (const {index, value} of candidates[i].entries) {
      model.weights[index] -= learningRate * coefficient * value;
    }
  }

  return {
    loss,
    correct: bestIndex === prepared.labelIndex,
    choices: candidates.length,
  };
}

function trainOnExample(model, example, options = {}) {
  return trainOnPreparedExample(model, prepareExample(model, example), options);
}

function evaluatePreparedExamples(model, preparedExamples) {
  const totals = {
    examples: 0,
    loss: 0,
    correct: 0,
    by_request_type: {},
  };

  for (const prepared of preparedExamples) {
    const candidates = prepared.candidates;
    const scores = candidates.map(candidate => scoreEntries(model.weights, candidate.entries));
    const probabilities = softmax(scores);
    let bestIndex = 0;
    for (let i = 1; i < scores.length; i++) {
      if (scores[i] > scores[bestIndex]) bestIndex = i;
    }
    const loss = -Math.log(Math.max(probabilities[prepared.labelIndex], 1e-12));
    const type = prepared.request_type || 'unknown';
    if (!totals.by_request_type[type]) {
      totals.by_request_type[type] = {examples: 0, loss: 0, correct: 0, accuracy: 0};
    }
    const bucket = totals.by_request_type[type];
    bucket.examples += 1;
    bucket.loss += loss;
    bucket.correct += bestIndex === prepared.labelIndex ? 1 : 0;
    totals.examples += 1;
    totals.loss += loss;
    totals.correct += bestIndex === prepared.labelIndex ? 1 : 0;
  }

  totals.avg_loss = totals.examples ? totals.loss / totals.examples : null;
  totals.accuracy = totals.examples ? totals.correct / totals.examples : null;
  for (const bucket of Object.values(totals.by_request_type)) {
    bucket.avg_loss = bucket.examples ? bucket.loss / bucket.examples : null;
    bucket.accuracy = bucket.examples ? bucket.correct / bucket.examples : null;
  }
  return totals;
}

function evaluateExamples(model, examples) {
  return evaluatePreparedExamples(model, examples.map(example => prepareExample(model, example)));
}

function sparseWeightsFromDense(weights) {
  const sparse = {};
  for (let i = 0; i < weights.length; i++) {
    if (weights[i] !== 0) sparse[i] = Number(weights[i].toFixed(12));
  }
  return sparse;
}

function denseWeightsFromSparse(sparse, featureDim) {
  const weights = new Float64Array(featureDim);
  for (const [indexText, value] of Object.entries(sparse || {})) {
    const index = Number(indexText);
    if (Number.isInteger(index) && index >= 0 && index < featureDim) {
      weights[index] = Number(value);
    }
  }
  return weights;
}

function serializeModel(model, extra = {}) {
  return {
    model_type: model.modelType || MODEL_TYPE,
    model_version: model.modelVersion || MODEL_VERSION,
    encoder_version: model.encoderVersion || ENCODER_VERSION,
    feature_dim: model.featureDim,
    encoder_options: model.encoderOptions,
    metadata: model.metadata || {},
    weights: sparseWeightsFromDense(model.weights),
    ...extra,
  };
}

function deserializeModel(raw) {
  if (!raw || raw.model_type !== MODEL_TYPE) {
    throw new Error(`Unsupported model_type: ${raw && raw.model_type}`);
  }
  const featureDim = raw.feature_dim;
  const model = createEmptyModel({
    featureDim,
    encoderOptions: raw.encoder_options || {includeStateSignature: true},
    metadata: raw.metadata || {},
  });
  model.modelVersion = raw.model_version || MODEL_VERSION;
  model.encoderVersion = raw.encoder_version || ENCODER_VERSION;
  model.weights = denseWeightsFromSparse(raw.weights, featureDim);
  return model;
}

function saveModel(model, modelPath, extra = {}) {
  fs.mkdirSync(path.dirname(modelPath), {recursive: true});
  fs.writeFileSync(modelPath, `${JSON.stringify(serializeModel(model, extra), null, 2)}\n`, 'utf8');
}

function loadModel(modelPath) {
  return deserializeModel(JSON.parse(fs.readFileSync(modelPath, 'utf8')));
}

module.exports = {
  MODEL_TYPE,
  MODEL_VERSION,
  createEmptyModel,
  deserializeModel,
  evaluateExamples,
  evaluatePreparedExamples,
  loadModel,
  prepareExample,
  predictChoice,
  saveModel,
  scoreChoices,
  scoreEntries,
  serializeModel,
  softmax,
  trainOnExample,
  trainOnPreparedExample,
};
