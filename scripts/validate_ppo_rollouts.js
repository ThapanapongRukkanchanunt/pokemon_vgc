const fs = require('node:fs');

function parseArgs(argv) {
  const args = {dataset: null, summary: null};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--dataset') {
      args.dataset = argv[++i];
    } else if (arg === '--summary') {
      args.summary = argv[++i];
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }
  if (!args.dataset) throw new Error('--dataset is required');
  return args;
}

function requireField(row, field, errors, index) {
  if (!Object.prototype.hasOwnProperty.call(row, field)) {
    errors.push(`row ${index}: missing ${field}`);
  }
}

function validateRow(row, index, errors, counts) {
  for (const field of [
    'battle_id',
    'trajectory_id',
    'turn',
    'side',
    'opponent_id',
    'state',
    'legal_actions',
    'action',
    'action_index',
    'log_prob',
    'value_prediction',
    'reward',
    'done',
    'winner_side',
  ]) {
    requireField(row, field, errors, index);
  }
  if (!Array.isArray(row.legal_actions) || !row.legal_actions.length) {
    errors.push(`row ${index}: legal_actions must be a non-empty array`);
  }
  if (!Number.isInteger(row.action_index)) {
    errors.push(`row ${index}: action_index must be an integer`);
  } else if (Array.isArray(row.legal_actions) &&
      (row.action_index < 0 || row.action_index >= row.legal_actions.length)) {
    errors.push(`row ${index}: action_index out of range`);
  }
  if (Array.isArray(row.legal_actions) && row.legal_actions[row.action_index] !== row.action) {
    errors.push(`row ${index}: action does not match legal_actions[action_index]`);
  }
  for (const field of ['log_prob', 'value_prediction', 'reward']) {
    if (!Number.isFinite(row[field])) errors.push(`row ${index}: ${field} must be finite`);
  }
  if (typeof row.done !== 'boolean') errors.push(`row ${index}: done must be boolean`);
  counts.rows += 1;
  counts.done += row.done ? 1 : 0;
  counts.by_opponent[row.opponent_id] = (counts.by_opponent[row.opponent_id] || 0) + 1;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const lines = fs.readFileSync(args.dataset, 'utf8').split(/\r?\n/).filter(Boolean);
  const errors = [];
  const counts = {rows: 0, done: 0, by_opponent: {}};
  lines.forEach((line, index) => {
    try {
      validateRow(JSON.parse(line), index + 1, errors, counts);
    } catch (error) {
      errors.push(`row ${index + 1}: invalid JSON: ${error.message}`);
    }
  });
  const summary = {
    dataset: args.dataset,
    ...counts,
    errors,
  };
  if (args.summary) {
    fs.writeFileSync(args.summary, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  }
  if (errors.length) {
    console.error(`FAIL ${args.dataset} errors=${errors.length}`);
    for (const error of errors.slice(0, 20)) console.error(error);
    process.exit(1);
  }
  console.log(`PASS ${args.dataset} rows=${counts.rows} done=${counts.done}`);
}

main();
