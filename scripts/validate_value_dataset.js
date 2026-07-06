const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.join(__dirname, '..');

function parseArgs(argv) {
  const args = {dataset: null, summary: null};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--dataset') {
      args.dataset = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--summary') {
      args.summary = path.resolve(repoRoot, argv[++i]);
    } else if (!args.dataset) {
      args.dataset = path.resolve(repoRoot, arg);
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }
  if (!args.dataset) throw new Error('Pass a dataset path or --dataset <path>');
  return args;
}

function inc(map, key) {
  const normalizedKey = key == null ? 'null' : String(key);
  map[normalizedKey] = (map[normalizedKey] || 0) + 1;
}

function* readJsonlRows(filePath) {
  const fd = fs.openSync(filePath, 'r');
  const buffer = Buffer.alloc(1024 * 1024);
  let leftover = '';
  let lineNumber = 0;
  try {
    while (true) {
      const bytesRead = fs.readSync(fd, buffer, 0, buffer.length, null);
      if (!bytesRead) break;
      const chunk = leftover + buffer.toString('utf8', 0, bytesRead);
      const lines = chunk.split('\n');
      leftover = lines.pop() || '';
      for (let line of lines) {
        lineNumber += 1;
        if (line.endsWith('\r')) line = line.slice(0, -1);
        if (line.trim()) yield {line, lineNumber};
      }
    }
    if (leftover) {
      lineNumber += 1;
      let line = leftover;
      if (line.endsWith('\r')) line = line.slice(0, -1);
      if (line.trim()) yield {line, lineNumber};
    }
  } finally {
    fs.closeSync(fd);
  }
}

function validateExample(example, fileName, lineNumber, errors) {
  const prefix = `${fileName}:${lineNumber}`;
  const requiredStrings = [
    'example_id',
    'source_trace_path',
    'battle_id',
    'format',
    'side',
    'agent',
    'team',
    'lead',
    'request_type',
    'action',
  ];

  for (const key of requiredStrings) {
    if (typeof example[key] !== 'string' || !example[key]) {
      errors.push(`${prefix} missing string ${key}`);
    }
  }
  if (typeof example.turn !== 'number') errors.push(`${prefix} missing numeric turn`);
  if (typeof example.battle_max_trace_turn !== 'number') {
    errors.push(`${prefix} missing numeric battle_max_trace_turn`);
  }
  if (typeof example.remaining_trace_turns !== 'number') {
    errors.push(`${prefix} missing numeric remaining_trace_turns`);
  }
  if (!['p1', 'p2'].includes(example.side)) errors.push(`${prefix} invalid side`);
  if (!example.state || typeof example.state !== 'object') errors.push(`${prefix} missing state`);
  if (!example.state || !example.state.request || typeof example.state.request !== 'object') {
    errors.push(`${prefix} missing state.request`);
  }
  if (!example.state || !example.state.public_state || typeof example.state.public_state !== 'object') {
    errors.push(`${prefix} missing state.public_state`);
  }
  if (!Array.isArray(example.legal_actions) || !example.legal_actions.length) {
    errors.push(`${prefix} missing non-empty legal_actions`);
  }
  if (typeof example.action_index !== 'number') {
    errors.push(`${prefix} missing numeric action_index`);
  } else if (
    Array.isArray(example.legal_actions) &&
    example.legal_actions[example.action_index] !== example.action
  ) {
    errors.push(`${prefix} action_index does not point to action`);
  }
  if (![null, 'p1', 'p2'].includes(example.winner_side)) {
    errors.push(`${prefix} invalid winner_side`);
  }
  if (![0, 1].includes(example.target)) {
    errors.push(`${prefix} invalid target`);
  }
  if (example.winner_side && example.target !== (example.winner_side === example.side ? 1 : 0)) {
    errors.push(`${prefix} target does not match winner_side and side`);
  }
  if (typeof example.is_recovery !== 'boolean') {
    errors.push(`${prefix} missing boolean is_recovery`);
  }
}

function validateDataset(datasetPath, summaryPath) {
  if (!fs.existsSync(datasetPath)) throw new Error(`Dataset not found: ${datasetPath}`);
  const fileName = path.basename(datasetPath);
  const errors = [];
  const counts = {
    examples: 0,
    battles: {},
    agents: {},
    request_types: {},
    winners: {},
    winner_sides: {},
    targets: {},
    teams: {},
    recovery_rows: {},
  };

  for (const {line, lineNumber} of readJsonlRows(datasetPath)) {
    let example;
    try {
      example = JSON.parse(line);
    } catch (error) {
      errors.push(`${fileName}:${lineNumber} invalid JSON: ${error.message}`);
      continue;
    }

    validateExample(example, fileName, lineNumber, errors);
    counts.examples += 1;
    inc(counts.battles, example.battle_id);
    inc(counts.agents, example.agent);
    inc(counts.request_types, example.request_type);
    inc(counts.winners, example.winner);
    inc(counts.winner_sides, example.winner_side);
    inc(counts.targets, example.target);
    inc(counts.teams, example.team);
    inc(counts.recovery_rows, example.is_recovery);
  }

  if (!counts.examples) errors.push(`${fileName} has no examples`);

  if (summaryPath) {
    if (!fs.existsSync(summaryPath)) {
      errors.push(`Summary not found: ${summaryPath}`);
    } else {
      const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf8'));
      if (summary.examples !== counts.examples) {
        errors.push(`Summary examples ${summary.examples} does not match dataset examples ${counts.examples}`);
      }
      const battleCount = Object.keys(counts.battles).length;
      if (summary.battles !== battleCount) {
        errors.push(`Summary battles ${summary.battles} does not match dataset battles ${battleCount}`);
      }
    }
  }

  return {counts, errors};
}

function main() {
  try {
    const args = parseArgs(process.argv.slice(2));
    const {counts, errors} = validateDataset(args.dataset, args.summary);
    if (errors.length) {
      for (const error of errors) console.error(`FAIL ${error}`);
      process.exit(1);
    }
    console.log(`PASS ${path.relative(repoRoot, args.dataset)} (${counts.examples} examples)`);
    console.log(`Battles: ${Object.keys(counts.battles).length}`);
    console.log(`Agents: ${JSON.stringify(counts.agents)}`);
    console.log(`Request types: ${JSON.stringify(counts.request_types)}`);
    console.log(`Targets: ${JSON.stringify(counts.targets)}`);
    console.log(`Recovery rows: ${JSON.stringify(counts.recovery_rows)}`);
  } catch (error) {
    console.error(`FAIL ${error.message}`);
    process.exit(1);
  }
}

main();
