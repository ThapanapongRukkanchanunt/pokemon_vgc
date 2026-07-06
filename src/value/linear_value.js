const fs = require('node:fs');
const path = require('node:path');
const {
  ENCODER_VERSION,
  contextFromExample,
  featurizeChoice,
  legalChoiceText,
  stateFeaturesFromContext,
} = require('../bc/feature_encoder');
const {scoreEntries} = require('../bc/linear_policy');

const MODEL_TYPE = 'hashed_linear_action_value';
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
    bias: 0,
    metadata,
  };
}

function sigmoid(value) {
  if (value >= 0) {
    const z = Math.exp(-value);
    return 1 / (1 + z);
  }
  const z = Math.exp(value);
  return z / (1 + z);
}

function binaryLogLoss(prediction, target) {
  const clipped = Math.max(1e-12, Math.min(1 - 1e-12, prediction));
  return -(target * Math.log(clipped) + (1 - target) * Math.log(1 - clipped));
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

function predictEntries(model, entries) {
  const logit = model.bias + scoreEntries(model.weights, entries);
  return {
    logit,
    value: sigmoid(logit),
  };
}

function scoreChoices(model, context, legalActions) {
  return candidateEntriesForChoices(model, context, legalActions).map(candidate => {
    const prediction = predictEntries(model, candidate.entries);
    return {
      choice: candidate.choice,
      value: prediction.value,
      score: prediction.value,
      logit: prediction.logit,
    };
  });
}

function actionForExample(example) {
  return example.action || example.chosen_action || example.label_action;
}

function targetForExample(example) {
  const target = example.target ?? example.win_target;
  if (target === 0 || target === 1) return target;
  throw new Error(`${example.example_id || 'example'} target must be 0 or 1`);
}

function prepareExample(model, example) {
  const action = actionForExample(example);
  if (typeof action !== 'string' || !action) {
    throw new Error(`${example.example_id || 'example'} missing action`);
  }
  if (!Array.isArray(example.legal_actions) || !example.legal_actions.includes(action)) {
    throw new Error(`${example.example_id || 'example'} action is not in legal_actions`);
  }

  const context = contextFromExample(example);
  return {
    example_id: example.example_id,
    battle_id: example.battle_id,
    request_type: example.request_type || 'unknown',
    agent: example.agent || 'unknown',
    target: targetForExample(example),
    action,
    entries: featurizeChoice({
      context,
      choice: action,
      featureDim: model.featureDim,
      encoderOptions: model.encoderOptions,
    }),
  };
}

function trainOnPreparedExample(model, prepared, {learningRate = 0.1, l2 = 0} = {}) {
  const prediction = predictEntries(model, prepared.entries);
  const loss = binaryLogLoss(prediction.value, prepared.target);
  const gradient = prediction.value - prepared.target;

  if (l2 > 0) {
    const touched = new Set(prepared.entries.map(entry => entry.index));
    for (const index of touched) model.weights[index] *= Math.max(0, 1 - learningRate * l2);
  }

  for (const {index, value} of prepared.entries) {
    model.weights[index] -= learningRate * gradient * value;
  }
  model.bias -= learningRate * gradient;

  return {
    loss,
    prediction: prediction.value,
    correct: (prediction.value >= 0.5 ? 1 : 0) === prepared.target,
  };
}

function trainOnExample(model, example, options = {}) {
  return trainOnPreparedExample(model, prepareExample(model, example), options);
}

function emptyBucket() {
  return {
    examples: 0,
    loss: 0,
    brier: 0,
    correct: 0,
    positives: 0,
    prediction_sum: 0,
    avg_loss: 0,
    avg_brier: 0,
    accuracy: 0,
    positive_rate: 0,
    mean_prediction: 0,
  };
}

function updateBucket(bucket, prediction, target) {
  const loss = binaryLogLoss(prediction, target);
  bucket.examples += 1;
  bucket.loss += loss;
  bucket.brier += Math.pow(prediction - target, 2);
  bucket.correct += (prediction >= 0.5 ? 1 : 0) === target ? 1 : 0;
  bucket.positives += target;
  bucket.prediction_sum += prediction;
}

function finalizeBucket(bucket) {
  if (!bucket.examples) return bucket;
  bucket.avg_loss = bucket.loss / bucket.examples;
  bucket.avg_brier = bucket.brier / bucket.examples;
  bucket.accuracy = bucket.correct / bucket.examples;
  bucket.positive_rate = bucket.positives / bucket.examples;
  bucket.mean_prediction = bucket.prediction_sum / bucket.examples;
  return bucket;
}

function calibrationKey(prediction) {
  const lower = Math.min(9, Math.max(0, Math.floor(prediction * 10))) / 10;
  const upper = lower + 0.1;
  return `${lower.toFixed(1)}-${upper.toFixed(1)}`;
}

function evaluatePreparedExamples(model, preparedExamples) {
  const totals = emptyBucket();
  const byRequestType = {};
  const byAgent = {};
  const calibration = {};

  for (const prepared of preparedExamples) {
    const prediction = predictEntries(model, prepared.entries).value;
    updateBucket(totals, prediction, prepared.target);

    const requestType = prepared.request_type || 'unknown';
    if (!byRequestType[requestType]) byRequestType[requestType] = emptyBucket();
    updateBucket(byRequestType[requestType], prediction, prepared.target);

    const agent = prepared.agent || 'unknown';
    if (!byAgent[agent]) byAgent[agent] = emptyBucket();
    updateBucket(byAgent[agent], prediction, prepared.target);

    const bucketKey = calibrationKey(prediction);
    if (!calibration[bucketKey]) calibration[bucketKey] = emptyBucket();
    updateBucket(calibration[bucketKey], prediction, prepared.target);
  }

  finalizeBucket(totals);
  for (const bucket of Object.values(byRequestType)) finalizeBucket(bucket);
  for (const bucket of Object.values(byAgent)) finalizeBucket(bucket);
  for (const bucket of Object.values(calibration)) finalizeBucket(bucket);

  totals.chance_accuracy = 0.5;
  totals.majority_accuracy = totals.examples ?
    Math.max(totals.positive_rate, 1 - totals.positive_rate) :
    0;
  totals.beats_chance = totals.examples ? totals.accuracy > totals.chance_accuracy : false;
  return {
    ...totals,
    by_request_type: byRequestType,
    by_agent: byAgent,
    calibration,
  };
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
    bias: Number((model.bias || 0).toFixed(12)),
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
  model.bias = Number(raw.bias || 0);
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
  binaryLogLoss,
  candidateEntriesForChoices,
  createEmptyModel,
  deserializeModel,
  evaluateExamples,
  evaluatePreparedExamples,
  loadModel,
  prepareExample,
  predictEntries,
  saveModel,
  scoreChoices,
  serializeModel,
  sigmoid,
  trainOnExample,
  trainOnPreparedExample,
};
