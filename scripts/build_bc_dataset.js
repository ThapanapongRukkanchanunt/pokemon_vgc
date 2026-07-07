const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.join(__dirname, '..');

function parseArgs(argv) {
  const args = {
    traceDir: path.join(repoRoot, 'logs', 'battles'),
    outDir: path.join(repoRoot, 'data', 'datasets', 'bc'),
    name: 'bc_dataset',
    agents: new Set(['max_damage_agent']),
    includeRecovery: false,
    includeTeamPreview: true,
    overwrite: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--trace-dir') {
      args.traceDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--out-dir') {
      args.outDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--name') {
      args.name = argv[++i];
    } else if (arg === '--agent') {
      args.agents = new Set(argv[++i].split(',').map(value => value.trim()).filter(Boolean));
    } else if (arg === '--include-recovery') {
      args.includeRecovery = true;
    } else if (arg === '--exclude-team-preview') {
      args.includeTeamPreview = false;
    } else if (arg === '--overwrite') {
      args.overwrite = true;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (!args.name || /[\\/:*?"<>|]/.test(args.name)) {
    throw new Error('--name must be a non-empty filename-safe value');
  }
  if (!args.agents.size) throw new Error('--agent must include at least one agent name');
  return args;
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function listTraceFiles(dir) {
  const entries = fs.readdirSync(dir, {withFileTypes: true});
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...listTraceFiles(fullPath));
    } else if (entry.isFile() && entry.name.endsWith('.trace.jsonl')) {
      files.push(fullPath);
    }
  }
  return files.sort();
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

function inc(map, key, amount = 1) {
  const normalizedKey = key == null ? 'null' : String(key);
  map[normalizedKey] = (map[normalizedKey] || 0) + amount;
}

function createSummary(args, datasetPath, summaryPath) {
  return {
    dataset_id: args.name,
    created_at: new Date().toISOString(),
    trace_dir: relativePath(args.traceDir),
    output_path: relativePath(datasetPath),
    summary_path: relativePath(summaryPath),
    agents: [...args.agents].sort(),
    include_recovery: args.includeRecovery,
    include_team_preview: args.includeTeamPreview,
    examples: 0,
    trace_files: 0,
    source_trace_files: [],
    skipped: {
      agent: 0,
      recovery: 0,
      team_preview: 0,
      invalid: 0,
    },
    counts: {
      agents: {},
      request_types: {},
      winners: {},
      winner_sides: {},
      teams: {},
      leads: {},
      recovery_rows: {},
    },
  };
}

function buildExample(row, sourceTracePath, lineNumber) {
  const type = requestType(row.request);
  const sideWon = winnerSide(row.outcome_context && row.outcome_context.winner);
  const labelActionIndex = row.legal_actions.indexOf(row.chosen_action);

  return {
    example_id: `${row.battle_id}:${lineNumber}`,
    source_trace_path: relativePath(sourceTracePath),
    battle_id: row.battle_id,
    seed: row.seed,
    format: row.format,
    turn: row.turn,
    side: row.side,
    agent: row.agent,
    team: row.team,
    lead: row.lead,
    request_type: type,
    state: {
      request: row.request,
      public_state: row.public_state,
    },
    legal_actions: row.legal_actions,
    label_action: row.chosen_action,
    label_action_index: labelActionIndex,
    winner: row.outcome_context ? row.outcome_context.winner : null,
    winner_side: sideWon,
    win_target: sideWon == null ? null : (sideWon === row.side ? 1 : 0),
    is_recovery: Boolean(row.error_recovery),
  };
}

function validateCandidate(row) {
  if (!row || typeof row !== 'object') return 'row is not an object';
  if (!Array.isArray(row.legal_actions) || !row.legal_actions.length) return 'missing legal_actions';
  if (typeof row.chosen_action !== 'string' || !row.chosen_action) return 'missing chosen_action';
  if (!row.legal_actions.includes(row.chosen_action)) return 'chosen_action is not legal';
  if (!['p1', 'p2'].includes(row.side)) return 'invalid side';
  if (typeof row.agent !== 'string' || !row.agent) return 'missing agent';
  if (!row.request || typeof row.request !== 'object') return 'missing request';
  if (!row.public_state || typeof row.public_state !== 'object') return 'missing public_state';
  return null;
}

function buildDataset(args) {
  if (!fs.existsSync(args.traceDir)) throw new Error(`Trace directory does not exist: ${args.traceDir}`);
  const traceFiles = listTraceFiles(args.traceDir);
  if (!traceFiles.length) throw new Error(`No *.trace.jsonl files found in ${args.traceDir}`);

  fs.mkdirSync(args.outDir, {recursive: true});
  const datasetPath = path.join(args.outDir, `${args.name}.jsonl`);
  const summaryPath = path.join(args.outDir, `${args.name}.summary.json`);
  if (!args.overwrite && (fs.existsSync(datasetPath) || fs.existsSync(summaryPath))) {
    throw new Error(`Output already exists for ${args.name}; pass --overwrite to replace it`);
  }

  const summary = createSummary(args, datasetPath, summaryPath);
  const datasetFd = fs.openSync(datasetPath, 'w');

  try {
    for (const tracePath of traceFiles) {
      summary.trace_files += 1;
      summary.source_trace_files.push(relativePath(tracePath));
      const rawLines = fs.readFileSync(tracePath, 'utf8').split(/\r?\n/).filter(Boolean);

      rawLines.forEach((line, index) => {
        let row;
        try {
          row = JSON.parse(line);
        } catch (error) {
          summary.skipped.invalid += 1;
          return;
        }

        if (!args.agents.has(row.agent)) {
          summary.skipped.agent += 1;
          return;
        }
        if (!args.includeRecovery && row.error_recovery) {
          summary.skipped.recovery += 1;
          return;
        }
        if (!args.includeTeamPreview && requestType(row.request) === 'team_preview') {
          summary.skipped.team_preview += 1;
          return;
        }

        const invalidReason = validateCandidate(row);
        if (invalidReason) {
          summary.skipped.invalid += 1;
          return;
        }

        const example = buildExample(row, tracePath, index + 1);
        fs.writeSync(datasetFd, `${JSON.stringify(example)}\n`);

        summary.examples += 1;
        inc(summary.counts.agents, example.agent);
        inc(summary.counts.request_types, example.request_type);
        inc(summary.counts.winners, example.winner);
        inc(summary.counts.winner_sides, example.winner_side);
        inc(summary.counts.teams, example.team);
        inc(summary.counts.leads, example.lead);
        inc(summary.counts.recovery_rows, example.is_recovery);
      });
    }
  } finally {
    fs.closeSync(datasetFd);
  }

  if (!summary.examples) {
    fs.unlinkSync(datasetPath);
    throw new Error('No dataset examples were produced with the selected filters');
  }

  fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
  return {datasetPath, summaryPath, summary};
}

function main() {
  try {
    const args = parseArgs(process.argv.slice(2));
    const {datasetPath, summaryPath, summary} = buildDataset(args);
    console.log(`Wrote ${summary.examples} examples`);
    console.log(`Dataset: ${relativePath(datasetPath)}`);
    console.log(`Summary: ${relativePath(summaryPath)}`);
    console.log(`Trace files: ${summary.trace_files}`);
    console.log(`Skipped: ${JSON.stringify(summary.skipped)}`);
  } catch (error) {
    console.error(`FAIL ${error.message}`);
    process.exit(1);
  }
}

main();
