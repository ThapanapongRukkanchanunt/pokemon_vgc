const fs = require('node:fs');
const path = require('node:path');
const {contextFromExample} = require('../src/bc/feature_encoder');
const {loadModel, scoreChoices} = require('../src/value/linear_value');

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
    model: path.join(repoRoot, 'models', 'value_model', 'phase4_mixed_q', 'model.json'),
    dataset: path.join(repoRoot, 'data', 'datasets', 'value', 'phase4_mixed_q.jsonl'),
    maxRemainingTurns: 1,
    minMargin: 0.01,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--model') {
      args.model = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--dataset') {
      args.dataset = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--max-remaining-turns') {
      args.maxRemainingTurns = parseInteger(argv[++i], '--max-remaining-turns');
    } else if (arg === '--min-margin') {
      args.minMargin = parseNumber(argv[++i], '--min-margin');
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (args.maxRemainingTurns < 0) throw new Error('--max-remaining-turns must be >= 0');
  return args;
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function loadExamples(datasetPath) {
  if (!fs.existsSync(datasetPath)) throw new Error(`Dataset not found: ${datasetPath}`);
  return fs.readFileSync(datasetPath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`${datasetPath}:${index + 1} invalid JSON: ${error.message}`);
      }
    });
}

function chosenValue(model, example) {
  const context = contextFromExample(example);
  const scores = scoreChoices(model, context, example.legal_actions || []);
  const row = scores.find(candidate => candidate.choice === example.action);
  if (!row) throw new Error(`${example.example_id} action not found in value scores`);
  return row.value;
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function check(args) {
  if (!fs.existsSync(args.model)) throw new Error(`Model not found: ${args.model}`);
  const model = loadModel(args.model);
  const examples = loadExamples(args.dataset);
  let candidates = examples.filter(example => {
    return Number.isFinite(example.remaining_trace_turns) &&
      example.remaining_trace_turns <= args.maxRemainingTurns &&
      (example.target === 0 || example.target === 1);
  });
  if (!candidates.length) {
    candidates = examples.filter(example => example.target === 0 || example.target === 1);
  }

  const winning = [];
  const losing = [];
  for (const example of candidates) {
    const value = chosenValue(model, example);
    if (example.target === 1) winning.push(value);
    else losing.push(value);
  }

  if (!winning.length || !losing.length) {
    throw new Error('Need at least one winning and one losing example for sanity check');
  }

  const winningMean = mean(winning);
  const losingMean = mean(losing);
  const margin = winningMean - losingMean;
  console.log(`Model: ${relativePath(args.model)}`);
  console.log(`Dataset: ${relativePath(args.dataset)}`);
  console.log(`Examples checked: ${candidates.length}`);
  console.log(`Late winning mean: ${winningMean.toFixed(4)} (${winning.length} examples)`);
  console.log(`Late losing mean: ${losingMean.toFixed(4)} (${losing.length} examples)`);
  console.log(`Margin: ${margin.toFixed(4)}`);
  if (margin < args.minMargin) {
    throw new Error(`Sanity margin ${margin.toFixed(4)} is below ${args.minMargin}`);
  }
  console.log('PASS late winning states score above late losing states');
}

function main() {
  try {
    check(parseArgs(process.argv.slice(2)));
  } catch (error) {
    console.error(`FAIL ${error.stack || error.message}`);
    process.exit(1);
  }
}

main();
