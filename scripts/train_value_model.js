const fs = require('node:fs');
const path = require('node:path');
const {
  createEmptyModel,
  evaluatePreparedExamples,
  loadModel,
  prepareExample,
  saveModel,
  trainOnPreparedExample,
} = require('../src/value/linear_value');

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
    dataset: path.join(repoRoot, 'data', 'datasets', 'value', 'phase4_mixed_q.jsonl'),
    outDir: path.join(repoRoot, 'models', 'value_model', 'phase4_mixed_q'),
    initModel: null,
    epochs: 120,
    learningRate: 0.15,
    l2: 0.00001,
    featureDim: 8192,
    seed: 'phase4_value_model',
    validationSplit: 0.25,
    evalEvery: 10,
    limit: null,
    overwrite: false,
    includeStateSignature: true,
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

function loadExamples(datasetPath, limit = null) {
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
    });
  return limit == null ? examples : examples.slice(0, limit);
}

function splitExamplesByBattle(examples, validationSplit, seed) {
  if (validationSplit === 0) {
    return {trainExamples: examples, validationExamples: [], validationBattleIds: []};
  }
  const byBattle = new Map();
  for (const example of examples) {
    const battleId = example.battle_id || example.example_id || 'unknown';
    if (!byBattle.has(battleId)) byBattle.set(battleId, []);
    byBattle.get(battleId).push(example);
  }
  const battleIds = shuffle([...byBattle.keys()], makeRng(`${seed}:battle-split`));
  const validationBattleCount = Math.max(1, Math.floor(battleIds.length * validationSplit));
  if (validationBattleCount >= battleIds.length) {
    throw new Error('Validation split leaves no training battles');
  }
  const validationBattleIds = new Set(battleIds.slice(0, validationBattleCount));
  const trainExamples = [];
  const validationExamples = [];
  for (const [battleId, rows] of byBattle) {
    if (validationBattleIds.has(battleId)) validationExamples.push(...rows);
    else trainExamples.push(...rows);
  }
  return {
    trainExamples,
    validationExamples,
    validationBattleIds: [...validationBattleIds].sort(),
  };
}

function compactMetrics(metrics) {
  if (!metrics) return null;
  return {
    examples: metrics.examples,
    avg_loss: metrics.avg_loss,
    avg_brier: metrics.avg_brier,
    accuracy: metrics.accuracy,
    correct: metrics.correct,
    positive_rate: metrics.positive_rate,
    mean_prediction: metrics.mean_prediction,
    chance_accuracy: metrics.chance_accuracy,
    majority_accuracy: metrics.majority_accuracy,
    beats_chance: metrics.beats_chance,
  };
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
  const examples = loadExamples(args.dataset, args.limit);
  if (!examples.length) throw new Error('No examples loaded');
  assertWritableOutput(args);
  if (args.initModel && !fs.existsSync(args.initModel)) {
    throw new Error(`Initial model not found: ${args.initModel}`);
  }

  const {trainExamples, validationExamples, validationBattleIds} = splitExamplesByBattle(
    examples,
    args.validationSplit,
    args.seed
  );
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
    target: 'P(acting side eventually wins | state, action)',
  };

  console.log('Preparing feature caches...');
  const preparedTrainExamples = trainExamples.map(example => prepareExample(model, example));
  const preparedValidationExamples = validationExamples.map(example => prepareExample(model, example));

  const history = [];
  const recordMetrics = epoch => {
    const trainMetrics = evaluatePreparedExamples(model, preparedTrainExamples);
    const validationMetrics = preparedValidationExamples.length ?
      evaluatePreparedExamples(model, preparedValidationExamples) :
      null;
    const row = {
      epoch,
      train: compactMetrics(trainMetrics),
      validation: compactMetrics(validationMetrics),
    };
    history.push(row);
    const validationText = validationMetrics ?
      ` val_loss=${validationMetrics.avg_loss.toFixed(4)} val_acc=${validationMetrics.accuracy.toFixed(3)}` +
      ` val_brier=${validationMetrics.avg_brier.toFixed(4)}` :
      '';
    console.log(
      `epoch=${epoch} train_loss=${trainMetrics.avg_loss.toFixed(4)} ` +
      `train_acc=${trainMetrics.accuracy.toFixed(3)} train_brier=${trainMetrics.avg_brier.toFixed(4)}` +
      validationText
    );
  };

  console.log(
    `Loaded ${examples.length} example(s); train=${trainExamples.length} ` +
    `validation=${validationExamples.length} validation_battles=${validationBattleIds.length}`
  );
  recordMetrics(0);

  for (let epoch = 1; epoch <= args.epochs; epoch++) {
    const epochRng = makeRng(`${args.seed}:epoch:${epoch}`);
    const epochExamples = shuffle(preparedTrainExamples, epochRng);
    for (const example of epochExamples) {
      trainOnPreparedExample(model, example, {
        learningRate: args.learningRate,
        l2: args.l2,
      });
    }
    if (epoch === 1 || epoch === args.epochs || epoch % args.evalEvery === 0) {
      recordMetrics(epoch);
    }
  }

  const metricsPath = path.join(args.outDir, 'metrics.json');
  const modelPath = path.join(args.outDir, 'model.json');
  const finalTrain = evaluatePreparedExamples(model, preparedTrainExamples);
  const finalValidation = preparedValidationExamples.length ?
    evaluatePreparedExamples(model, preparedValidationExamples) :
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
      split_unit: 'battle_id',
      init_model_path: args.initModel ? relativePath(args.initModel) : null,
    },
    examples: examples.length,
    train_examples: trainExamples.length,
    validation_examples: validationExamples.length,
    validation_battle_ids: validationBattleIds,
    history,
    final_train: compactMetrics(finalTrain),
    final_validation: compactMetrics(finalValidation),
    final_validation_by_request_type: finalValidation ? finalValidation.by_request_type : null,
    final_validation_calibration: finalValidation ? finalValidation.calibration : null,
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
