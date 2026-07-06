const fs = require('node:fs');
const path = require('node:path');
const {contextFromExample} = require('../src/bc/feature_encoder');
const {loadModel, scoreChoices} = require('../src/value/linear_value');

const repoRoot = path.join(__dirname, '..');

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    model: path.join(repoRoot, 'models', 'value_model', 'phase4_mixed_q', 'model.json'),
    dataset: null,
    trace: null,
    limit: 5,
    topK: 5,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--model') {
      args.model = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--dataset') {
      args.dataset = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--trace') {
      args.trace = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--limit') {
      args.limit = parseInteger(argv[++i], '--limit');
    } else if (arg === '--top-k') {
      args.topK = parseInteger(argv[++i], '--top-k');
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (!args.dataset && !args.trace) {
    args.dataset = path.join(repoRoot, 'data', 'datasets', 'value', 'phase4_mixed_q.jsonl');
  }
  if (args.dataset && args.trace) throw new Error('Pass only one of --dataset or --trace');
  if (args.limit <= 0) throw new Error('--limit must be > 0');
  if (args.topK <= 0) throw new Error('--top-k must be > 0');
  return args;
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function requestType(request) {
  if (!request || typeof request !== 'object') return 'other';
  if (request.teamPreview) return 'team_preview';
  if (Array.isArray(request.forceSwitch) && request.forceSwitch.some(Boolean)) return 'force_switch';
  if (Array.isArray(request.active) && request.active.length) return 'move';
  if (request.wait) return 'wait';
  return 'other';
}

function winnerSide(winner) {
  if (typeof winner !== 'string') return null;
  const match = winner.match(/\b(P[12])$/i);
  return match ? match[1].toLowerCase() : null;
}

function readJsonl(filePath) {
  if (!fs.existsSync(filePath)) throw new Error(`File not found: ${filePath}`);
  return fs.readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`${filePath}:${index + 1} invalid JSON: ${error.message}`);
      }
    });
}

function exampleFromTraceRow(row, lineNumber, tracePath) {
  const sideWon = winnerSide(row.outcome_context && row.outcome_context.winner);
  return {
    example_id: `${row.battle_id || relativePath(tracePath)}:${lineNumber}`,
    source_trace_path: relativePath(tracePath),
    battle_id: row.battle_id,
    seed: row.seed,
    format: row.format,
    turn: row.turn,
    side: row.side,
    agent: row.agent,
    team: row.team,
    lead: row.lead,
    request_type: requestType(row.request),
    state: {
      request: row.request,
      public_state: row.public_state,
    },
    legal_actions: row.legal_actions,
    action: row.chosen_action,
    winner: row.outcome_context ? row.outcome_context.winner : null,
    winner_side: sideWon,
    target: sideWon == null ? null : (sideWon === row.side ? 1 : 0),
    is_recovery: Boolean(row.error_recovery),
  };
}

function loadExamples(args) {
  if (args.dataset) return readJsonl(args.dataset);
  return readJsonl(args.trace).map((row, index) => exampleFromTraceRow(row, index + 1, args.trace));
}

function inspect(args) {
  if (!fs.existsSync(args.model)) throw new Error(`Model not found: ${args.model}`);
  const model = loadModel(args.model);
  const examples = loadExamples(args).slice(0, args.limit);
  console.log(`Model: ${relativePath(args.model)}`);
  console.log(`Examples: ${examples.length}`);

  for (const example of examples) {
    const context = contextFromExample(example);
    const scores = scoreChoices(model, context, example.legal_actions || [])
      .sort((a, b) => b.value - a.value);
    const chosen = scores.find(row => row.choice === example.action);
    const chosenText = chosen ? chosen.value.toFixed(3) : 'missing';
    const targetText = example.target == null ? 'unknown' : String(example.target);
    console.log('');
    console.log(`${example.example_id} side=${example.side} agent=${example.agent} turn=${example.turn} target=${targetText}`);
    console.log(`chosen value=${chosenText} action=${example.action}`);
    scores.slice(0, args.topK).forEach((row, index) => {
      console.log(`#${index + 1} value=${row.value.toFixed(3)} action=${row.choice}`);
    });
  }
}

function main() {
  try {
    inspect(parseArgs(process.argv.slice(2)));
  } catch (error) {
    console.error(`FAIL ${error.stack || error.message}`);
    process.exit(1);
  }
}

main();
