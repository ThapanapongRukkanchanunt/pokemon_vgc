const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.join(__dirname, '..');

function parseArgs(argv) {
  const args = {logDir: path.join(repoRoot, 'logs', 'battles')};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--log-dir') {
      args.logDir = path.resolve(repoRoot, argv[++i]);
      continue;
    }
    throw new Error(`Unexpected argument: ${arg}`);
  }
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function validateTraceRow(row, fileName, lineNumber, errors) {
  const prefix = `${fileName}:${lineNumber}`;
  if (!row || typeof row !== 'object') errors.push(`${prefix} row is not an object`);
  if (typeof row.battle_id !== 'string' || !row.battle_id) errors.push(`${prefix} missing battle_id`);
  if (typeof row.turn !== 'number') errors.push(`${prefix} missing numeric turn`);
  if (!['p1', 'p2'].includes(row.side)) errors.push(`${prefix} invalid side`);
  if (typeof row.agent !== 'string' || !row.agent) errors.push(`${prefix} missing agent`);
  if (!row.request || typeof row.request !== 'object') errors.push(`${prefix} missing request`);
  if (!row.public_state || typeof row.public_state !== 'object') errors.push(`${prefix} missing public_state`);
  if (!Array.isArray(row.legal_actions)) errors.push(`${prefix} missing legal_actions array`);
  if (typeof row.chosen_action !== 'string' || !row.chosen_action) errors.push(`${prefix} missing chosen_action`);
  if (Array.isArray(row.legal_actions) && !row.legal_actions.includes(row.chosen_action)) {
    errors.push(`${prefix} chosen_action is not listed in legal_actions: ${row.chosen_action}`);
  }
}

function validateTraceFile(filePath) {
  const errors = [];
  const fileName = path.basename(filePath);
  const lines = fs.readFileSync(filePath, 'utf8').split(/\r?\n/).filter(Boolean);
  if (!lines.length) errors.push(`${fileName} has no trace rows`);

  lines.forEach((line, index) => {
    try {
      validateTraceRow(JSON.parse(line), fileName, index + 1, errors);
    } catch (error) {
      errors.push(`${fileName}:${index + 1} invalid JSON: ${error.message}`);
    }
  });

  return {fileName, rows: lines.length, errors};
}

function validateSummaries(logDir) {
  const warnings = [];
  const summaryFiles = fs.readdirSync(logDir).filter(file => file.endsWith('.summary.json'));
  for (const fileName of summaryFiles) {
    const summary = readJson(path.join(logDir, fileName));
    if (!summary.trace_jsonl_path) {
      warnings.push(`${fileName} has no trace_jsonl_path; likely an older battle artifact`);
      continue;
    }
    const tracePath = path.resolve(repoRoot, summary.trace_jsonl_path);
    if (!fs.existsSync(tracePath)) {
      warnings.push(`${fileName} points to missing trace file: ${summary.trace_jsonl_path}`);
    }
  }
  return warnings;
}

function main() {
  const {logDir} = parseArgs(process.argv.slice(2));
  const traceFiles = fs.readdirSync(logDir)
    .filter(file => file.endsWith('.trace.jsonl'))
    .map(file => path.join(logDir, file));

  const results = traceFiles.map(validateTraceFile);
  const errors = results.flatMap(result => result.errors);
  const warnings = validateSummaries(logDir);
  const rowCount = results.reduce((sum, result) => sum + result.rows, 0);

  for (const result of results) {
    console.log(`PASS ${result.fileName} (${result.rows} rows)`);
  }
  for (const warning of warnings) {
    console.warn(`WARN ${warning}`);
  }

  if (!traceFiles.length) errors.push(`No *.trace.jsonl files found in ${logDir}`);
  if (errors.length) {
    for (const error of errors) console.error(`FAIL ${error}`);
    process.exit(1);
  }

  console.log(`Validated ${traceFiles.length} trace file(s), ${rowCount} row(s).`);
}

main();
