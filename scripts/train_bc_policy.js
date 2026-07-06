const fs = require('node:fs');
const path = require('node:path');
const {
  createEmptyModel,
  evaluatePreparedExamples,
  loadModel,
  prepareExample,
  saveModel,
  scoreEntries,
  softmax,
  trainOnPreparedExample,
} = require('../src/bc/linear_policy');

const repoRoot = path.join(__dirname, '..');

function parseNumber(value, name) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`${name} must be numeric`);
  return parsed;
}

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    dataset: path.join(repoRoot, 'data', 'datasets', 'bc', 'trace_test_maxdamage.jsonl'),
    outDir: path.join(repoRoot, 'models', 'bc_policy', 'trace_test_maxdamage'),
    initModel: null,
    epochs: 80,
    learningRate: 0.25,
    l2: 0.00001,
    featureDim: 8192,
    seed: 'phase2_bc_policy',
    validationSplit: 0.2,
    evalEvery: 10,
    limit: null,
    overwrite: false,
    includeStateSignature: true,
    cacheFeatures: true,
    compactExamples: false,
    trainProgressEvery: 0,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--dataset') {
      args.dataset = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--out-dir') {
      args.outDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--init-model') {
      args.initModel = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--epochs') {
      args.epochs = parseInteger(argv[++i], '--epochs');
    } else if (arg === '--learning-rate') {
      args.learningRate = parseNumber(argv[++i], '--learning-rate');
    } else if (arg === '--l2') {
      args.l2 = parseNumber(argv[++i], '--l2');
    } else if (arg === '--feature-dim') {
      args.featureDim = parseInteger(argv[++i], '--feature-dim');
    } else if (arg === '--seed') {
      args.seed = argv[++i];
    } else if (arg === '--validation-split') {
      args.validationSplit = parseNumber(argv[++i], '--validation-split');
    } else if (arg === '--eval-every') {
      args.evalEvery = parseInteger(argv[++i], '--eval-every');
    } else if (arg === '--limit') {
      args.limit = parseInteger(argv[++i], '--limit');
    } else if (arg === '--no-state-signature') {
      args.includeStateSignature = false;
    } else if (arg === '--no-feature-cache') {
      args.cacheFeatures = false;
    } else if (arg === '--compact-examples') {
      args.compactExamples = true;
    } else if (arg === '--train-progress-every') {
      args.trainProgressEvery = parseInteger(argv[++i], '--train-progress-every');
    } else if (arg === '--overwrite') {
      args.overwrite = true;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (args.epochs < 0) throw new Error('--epochs must be >= 0');
  if (args.learningRate <= 0) throw new Error('--learning-rate must be > 0');
  if (args.l2 < 0) throw new Error('--l2 must be >= 0');
  if (args.featureDim <= 0) throw new Error('--feature-dim must be > 0');
  if (args.validationSplit < 0 || args.validationSplit >= 1) {
    throw new Error('--validation-split must be >= 0 and < 1');
  }
  if (args.evalEvery <= 0) throw new Error('--eval-every must be > 0');
  if (args.trainProgressEvery < 0) throw new Error('--train-progress-every must be >= 0');
  return args;
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function hashSeed(value) {
  const text = String(value);
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function makeRng(seedValue) {
  let state = hashSeed(seedValue) || 0x9e3779b9;
  return {
    next() {
      state += 0x6d2b79f5;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    },
  };
}

function shuffle(items, rng) {
  const copy = items.slice();
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function compactExample(example) {
  return {
    example_id: example.example_id,
    request_type: example.request_type || 'unknown',
    side: example.side,
    turn: example.turn,
    team: example.team,
    lead: example.lead,
    state: {
      request: example.state?.request,
      public_state: example.state?.public_state,
    },
    legal_actions: example.legal_actions,
    label_action: example.label_action,
    label_action_index: example.label_action_index,
  };
}

function loadExamples(datasetPath, limit = null, {compact = false} = {}) {
  if (!fs.existsSync(datasetPath)) throw new Error(`Dataset not found: ${datasetPath}`);
  const examples = fs.readFileSync(datasetPath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`${datasetPath}:${index + 1} invalid JSON: ${error.message}`);
      }
    })
    .map(example => compact ? compactExample(example) : example);
  return limit == null ? examples : examples.slice(0, limit);
}

function splitExamples(examples, validationSplit, seed) {
  const shuffled = shuffle(examples, makeRng(`${seed}:split`));
  const validationCount = Math.floor(shuffled.length * validationSplit);
  return {
    validationExamples: shuffled.slice(0, validationCount),
    trainExamples: shuffled.slice(validationCount),
  };
}

function compactMetrics(metrics) {
  if (!metrics) return null;
  return {
    examples: metrics.examples,
    avg_loss: metrics.avg_loss,
    accuracy: metrics.accuracy,
    correct: metrics.correct,
  };
}

function emptyMetrics() {
  return {
    examples: 0,
    loss: 0,
    correct: 0,
    by_request_type: {},
  };
}

function updateMetricsBucket(bucket, loss, correct) {
  bucket.examples += 1;
  bucket.loss += loss;
  bucket.correct += correct ? 1 : 0;
}

function finalizeMetrics(metrics) {
  metrics.avg_loss = metrics.examples ? metrics.loss / metrics.examples : null;
  metrics.accuracy = metrics.examples ? metrics.correct / metrics.examples : null;
  for (const bucket of Object.values(metrics.by_request_type)) {
    bucket.avg_loss = bucket.examples ? bucket.loss / bucket.examples : null;
    bucket.accuracy = bucket.examples ? bucket.correct / bucket.examples : null;
  }
  return metrics;
}

function evaluateOnePrepared(model, prepared) {
  const scores = prepared.candidates.map(candidate => scoreEntries(model.weights, candidate.entries));
  const probabilities = softmax(scores);
  let bestIndex = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[bestIndex]) bestIndex = i;
  }
  return {
    loss: -Math.log(Math.max(probabilities[prepared.labelIndex], 1e-12)),
    correct: bestIndex === prepared.labelIndex,
  };
}

function evaluateExamplesUncached(model, examples) {
  const totals = emptyMetrics();
  for (const example of examples) {
    const prepared = prepareExample(model, example);
    const result = evaluateOnePrepared(model, prepared);
    const type = prepared.request_type || 'unknown';
    if (!totals.by_request_type[type]) {
      totals.by_request_type[type] = {examples: 0, loss: 0, correct: 0, accuracy: 0};
    }
    updateMetricsBucket(totals, result.loss, result.correct);
    updateMetricsBucket(totals.by_request_type[type], result.loss, result.correct);
  }
  return finalizeMetrics(totals);
}

function assertWritableOutput(args) {
  const modelPath = path.join(args.outDir, 'model.json');
  const metricsPath = path.join(args.outDir, 'metrics.json');
  if (!args.overwrite && (fs.existsSync(modelPath) || fs.existsSync(metricsPath))) {
    throw new Error(`Output exists in ${relativePath(args.outDir)}; pass --overwrite to replace model/metrics`);
  }
  fs.mkdirSync(args.outDir, {recursive: true});
}

function train(args) {
  const examples = loadExamples(args.dataset, args.limit, {compact: args.compactExamples});
  if (!examples.length) throw new Error('No examples loaded');
  assertWritableOutput(args);
  if (args.initModel && !fs.existsSync(args.initModel)) {
    throw new Error(`Initial model not found: ${args.initModel}`);
  }

  const {trainExamples, validationExamples} = splitExamples(examples, args.validationSplit, args.seed);
  if (!trainExamples.length) throw new Error('No training examples after validation split');

  const model = args.initModel ?
    loadModel(args.initModel) :
    createEmptyModel({
      featureDim: args.featureDim,
      encoderOptions: {includeStateSignature: args.includeStateSignature},
      metadata: {},
    });
  model.metadata = {
    ...(model.metadata || {}),
    dataset_path: relativePath(args.dataset),
    seed: args.seed,
    initialized_from: args.initModel ? relativePath(args.initModel) : null,
  };
  const preparedTrainExamples = args.cacheFeatures ?
    trainExamples.map(example => prepareExample(model, example)) :
    null;
  const preparedValidationExamples = args.cacheFeatures ?
    validationExamples.map(example => prepareExample(model, example)) :
    null;
  console.log(args.cacheFeatures ? 'Preparing feature caches...' : 'Feature cache disabled; training examples will be featurized on demand.');

  const history = [];
  const recordMetrics = epoch => {
    const trainMetrics = args.cacheFeatures ?
      evaluatePreparedExamples(model, preparedTrainExamples) :
      evaluateExamplesUncached(model, trainExamples);
    const validationMetrics = validationExamples.length ?
      (args.cacheFeatures ?
        evaluatePreparedExamples(model, preparedValidationExamples) :
        evaluateExamplesUncached(model, validationExamples)) :
      null;
    const row = {
      epoch,
      train: compactMetrics(trainMetrics),
      validation: compactMetrics(validationMetrics),
    };
    history.push(row);
    const validationText = validationMetrics ?
      ` val_loss=${validationMetrics.avg_loss.toFixed(4)} val_acc=${validationMetrics.accuracy.toFixed(3)}` :
      '';
    console.log(
      `epoch=${epoch} train_loss=${trainMetrics.avg_loss.toFixed(4)} ` +
      `train_acc=${trainMetrics.accuracy.toFixed(3)}${validationText}`
    );
  };

  console.log(
    `Loaded ${examples.length} example(s); train=${trainExamples.length} ` +
    `validation=${validationExamples.length}`
  );
  recordMetrics(0);

  for (let epoch = 1; epoch <= args.epochs; epoch++) {
    const epochRng = makeRng(`${args.seed}:epoch:${epoch}`);
    const epochExamples = shuffle(args.cacheFeatures ? preparedTrainExamples : trainExamples, epochRng);
    for (let index = 0; index < epochExamples.length; index++) {
      const prepared = args.cacheFeatures ? epochExamples[index] : prepareExample(model, epochExamples[index]);
      trainOnPreparedExample(model, prepared, {
        learningRate: args.learningRate,
        l2: args.l2,
      });
      if (!args.cacheFeatures &&
          args.trainProgressEvery > 0 &&
          (index + 1) % args.trainProgressEvery === 0) {
        console.log(`epoch=${epoch} trained=${index + 1}/${epochExamples.length}`);
      }
    }
    if (epoch === 1 || epoch === args.epochs || epoch % args.evalEvery === 0) {
      recordMetrics(epoch);
    }
  }

  const metricsPath = path.join(args.outDir, 'metrics.json');
  const modelPath = path.join(args.outDir, 'model.json');
  const finalTrain = args.cacheFeatures ?
    evaluatePreparedExamples(model, preparedTrainExamples) :
    evaluateExamplesUncached(model, trainExamples);
  const finalValidation = validationExamples.length ?
    (args.cacheFeatures ?
      evaluatePreparedExamples(model, preparedValidationExamples) :
      evaluateExamplesUncached(model, validationExamples)) :
    null;
  const metrics = {
    created_at: new Date().toISOString(),
    dataset_path: relativePath(args.dataset),
    out_dir: relativePath(args.outDir),
    config: {
      epochs: args.epochs,
      learning_rate: args.learningRate,
      l2: args.l2,
      feature_dim: model.featureDim,
      seed: args.seed,
      validation_split: args.validationSplit,
      limit: args.limit,
      include_state_signature: args.includeStateSignature,
      cache_features: args.cacheFeatures,
      compact_examples: args.compactExamples,
      train_progress_every: args.trainProgressEvery,
      init_model_path: args.initModel ? relativePath(args.initModel) : null,
    },
    examples: examples.length,
    train_examples: trainExamples.length,
    validation_examples: validationExamples.length,
    history,
    final_train: compactMetrics(finalTrain),
    final_validation: compactMetrics(finalValidation),
  };

  fs.writeFileSync(metricsPath, `${JSON.stringify(metrics, null, 2)}\n`, 'utf8');
  saveModel(model, modelPath, {
    trained_at: metrics.created_at,
    training: {
      dataset_path: metrics.dataset_path,
      metrics_path: relativePath(metricsPath),
      examples: examples.length,
      train_examples: trainExamples.length,
      validation_examples: validationExamples.length,
      initialized_from: args.initModel ? relativePath(args.initModel) : null,
      final_train: metrics.final_train,
      final_validation: metrics.final_validation,
    },
  });

  console.log(`Wrote model: ${relativePath(modelPath)}`);
  console.log(`Wrote metrics: ${relativePath(metricsPath)}`);
  return {modelPath, metricsPath, metrics};
}

function main() {
  try {
    train(parseArgs(process.argv.slice(2)));
  } catch (error) {
    console.error(`FAIL ${error.stack || error.message}`);
    process.exit(1);
  }
}

main();
