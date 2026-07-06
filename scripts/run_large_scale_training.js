const fs = require('node:fs');
const path = require('node:path');
const {spawnSync} = require('node:child_process');
const {closeTorchPolicyScorers, createAgent} = require('../src/agents');
const {
  findLeadMode,
  findTeam,
  loadTeamPool,
  makeRng,
  runBattle,
} = require('../src/battle/run_battle');

const repoRoot = path.join(__dirname, '..');

const PARTICIPANTS = {
  random: {
    label: 'Random',
    agent: 'random',
  },
  maxdamage: {
    label: 'MaxDamage',
    agent: 'maxdamage',
  },
  phase2_bc: {
    label: 'Phase2BC',
    agent: 'bc_policy',
    policy: 'baseline',
  },
  phase6_search_policy: {
    label: 'Phase6SearchPolicy',
    agent: 'bc_policy',
    policy: 'current',
  },
  torch_current_policy: {
    label: 'TorchCurrentPolicy',
    agent: 'torch_policy',
    policy: 'current',
  },
  heuristic: {
    label: 'Heuristic',
    agent: 'heuristic_selector',
  },
  risk_balanced: {
    label: 'RiskBalanced',
    agent: 'risk_balanced',
    policy: 'baseline',
    value: 'bootstrap',
  },
  shallow_search: {
    label: 'ShallowSearch',
    agent: 'search_balanced',
    policy: 'current',
    value: 'bootstrap',
  },
  candidate_policy: {
    label: 'LargeScalePolicy',
    agent: 'bc_policy',
    policy: 'candidate',
  },
  candidate_torch_policy: {
    label: 'LargeScaleTorchPolicy',
    agent: 'torch_policy',
    policy: 'candidate',
  },
  candidate_search: {
    label: 'LargeScaleSearch',
    agent: 'search_balanced',
    policy: 'candidate',
    value: 'candidate',
  },
};

const DEFAULT_TRAIN_PARTICIPANTS = [
  'random',
  'maxdamage',
  'phase2_bc',
  'phase6_search_policy',
  'heuristic',
  'risk_balanced',
  'shallow_search',
];

const DEFAULT_EVAL_PARTICIPANTS = [
  'phase2_bc',
  'phase6_search_policy',
  'heuristic',
  'risk_balanced',
  'candidate_policy',
  'candidate_search',
];

const DEFAULT_STAGES = [
  'generate',
  'value-dataset',
  'train-value',
  'search-dataset',
  'train-policy',
  'eval',
];

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseNumber(value, name) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`${name} must be numeric`);
  return parsed;
}

function parseList(value) {
  return String(value || '')
    .split(',')
    .map(item => item.trim())
    .filter(Boolean);
}

function parseArgs(argv) {
  const args = {
    runId: 'phase6_10k',
    iterations: 1,
    games: 10000,
    evalGames: 0,
    seed: null,
    outRoot: path.join(repoRoot, 'experiments', 'large_scale'),
    logDir: null,
    evalLogDir: null,
    baselineModel: path.join(repoRoot, 'models', 'bc_policy', 'trace_test_maxdamage', 'model.json'),
    currentModel: path.join(repoRoot, 'models', 'bc_policy', 'phase6_search_improved', 'model.json'),
    bootstrapValueModel: path.join(repoRoot, 'models', 'value_model', 'phase4_mixed_q', 'model.json'),
    initPolicyModel: null,
    initValueModel: null,
    trainParticipants: DEFAULT_TRAIN_PARTICIPANTS,
    evalParticipants: DEFAULT_EVAL_PARTICIPANTS,
    stages: DEFAULT_STAGES,
    valueEpochs: 20,
    valueLearningRate: 0.03,
    valueL2: 0.001,
    valueFeatureDim: 16384,
    policyEpochs: 80,
    policyLearningRate: 0.2,
    policyL2: 0.00001,
    policyFeatureDim: 16384,
    validationSplit: 0.2,
    evalEvery: 5,
    searchProgressEvery: 500,
    maxFailures: 0,
    compactLogs: false,
    compactTrainLogs: false,
    compactEvalLogs: false,
    pythonPath: process.env.POKEMON_RL_PYTHON || null,
    torchDevice: process.env.POKEMON_RL_TORCH_DEVICE || null,
    overwrite: false,
    dryRun: false,
  };

  let seedExplicit = false;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--run-id') {
      args.runId = argv[++i];
    } else if (arg === '--iterations') {
      args.iterations = parseInteger(argv[++i], '--iterations');
    } else if (arg === '--games') {
      args.games = parseInteger(argv[++i], '--games');
    } else if (arg === '--eval-games') {
      args.evalGames = parseInteger(argv[++i], '--eval-games');
    } else if (arg === '--seed') {
      args.seed = argv[++i];
      seedExplicit = true;
    } else if (arg === '--out-root') {
      args.outRoot = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--log-dir') {
      args.logDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--eval-log-dir') {
      args.evalLogDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--baseline-model') {
      args.baselineModel = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--current-model') {
      args.currentModel = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--bootstrap-value-model') {
      args.bootstrapValueModel = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--init-policy-model') {
      args.initPolicyModel = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--init-value-model') {
      args.initValueModel = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--train-participants') {
      args.trainParticipants = parseList(argv[++i]);
    } else if (arg === '--eval-participants') {
      args.evalParticipants = parseList(argv[++i]);
    } else if (arg === '--stages') {
      args.stages = parseList(argv[++i]);
    } else if (arg === '--value-epochs') {
      args.valueEpochs = parseInteger(argv[++i], '--value-epochs');
    } else if (arg === '--value-learning-rate') {
      args.valueLearningRate = parseNumber(argv[++i], '--value-learning-rate');
    } else if (arg === '--value-l2') {
      args.valueL2 = parseNumber(argv[++i], '--value-l2');
    } else if (arg === '--value-feature-dim') {
      args.valueFeatureDim = parseInteger(argv[++i], '--value-feature-dim');
    } else if (arg === '--policy-epochs') {
      args.policyEpochs = parseInteger(argv[++i], '--policy-epochs');
    } else if (arg === '--policy-learning-rate') {
      args.policyLearningRate = parseNumber(argv[++i], '--policy-learning-rate');
    } else if (arg === '--policy-l2') {
      args.policyL2 = parseNumber(argv[++i], '--policy-l2');
    } else if (arg === '--policy-feature-dim') {
      args.policyFeatureDim = parseInteger(argv[++i], '--policy-feature-dim');
    } else if (arg === '--validation-split') {
      args.validationSplit = parseNumber(argv[++i], '--validation-split');
    } else if (arg === '--eval-every') {
      args.evalEvery = parseInteger(argv[++i], '--eval-every');
    } else if (arg === '--search-progress-every') {
      args.searchProgressEvery = parseInteger(argv[++i], '--search-progress-every');
    } else if (arg === '--max-failures') {
      args.maxFailures = parseInteger(argv[++i], '--max-failures');
    } else if (arg === '--compact-logs') {
      args.compactLogs = true;
    } else if (arg === '--compact-train-logs') {
      args.compactTrainLogs = true;
    } else if (arg === '--compact-eval-logs') {
      args.compactEvalLogs = true;
    } else if (arg === '--python') {
      args.pythonPath = argv[++i];
    } else if (arg === '--torch-device') {
      args.torchDevice = argv[++i];
    } else if (arg === '--overwrite') {
      args.overwrite = true;
    } else if (arg === '--dry-run') {
      args.dryRun = true;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (!args.runId || /[\\/:*?"<>|]/.test(args.runId)) {
    throw new Error('--run-id must be a non-empty filename-safe value');
  }
  if (!seedExplicit) args.seed = args.runId;
  if (args.iterations <= 0) throw new Error('--iterations must be > 0');
  if (args.games <= 0) throw new Error('--games must be > 0');
  if (args.evalGames < 0) throw new Error('--eval-games must be >= 0');
  if (args.valueEpochs < 0) throw new Error('--value-epochs must be >= 0');
  if (args.policyEpochs < 0) throw new Error('--policy-epochs must be >= 0');
  if (args.validationSplit < 0 || args.validationSplit >= 1) {
    throw new Error('--validation-split must be >= 0 and < 1');
  }
  if (args.evalEvery <= 0) throw new Error('--eval-every must be > 0');
  if (args.searchProgressEvery < 0) throw new Error('--search-progress-every must be >= 0');
  if (args.maxFailures < 0) throw new Error('--max-failures must be >= 0');
  validateParticipants(args.trainParticipants, '--train-participants');
  validateParticipants(args.evalParticipants, '--eval-participants');
  validateStages(args.stages);
  return args;
}

function validateParticipants(participants, flag) {
  if (!participants.length) throw new Error(`${flag} must include at least one participant`);
  for (const id of participants) {
    if (!PARTICIPANTS[id]) {
      throw new Error(`Unknown participant ${id}; choose from ${Object.keys(PARTICIPANTS).join(', ')}`);
    }
  }
}

function validateStages(stages) {
  const known = new Set(DEFAULT_STAGES);
  if (!stages.length) throw new Error('--stages must include at least one stage');
  for (const stage of stages) {
    if (!known.has(stage)) {
      throw new Error(`Unknown stage ${stage}; choose from ${[...known].join(', ')}`);
    }
  }
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function iterationTag(iteration) {
  return `iter_${String(iteration).padStart(3, '0')}`;
}

function outputsForArgs(args, iteration = null) {
  const runDir = path.join(args.outRoot, args.runId);
  const tag = iteration == null ? null : iterationTag(iteration);
  const outputId = tag ? `${args.runId}_${tag}` : args.runId;
  const stageDir = tag ? path.join(runDir, tag) : runDir;
  const trainLogDir = args.logDir ?
    (tag ? path.join(args.logDir, tag) : args.logDir) :
    path.join(repoRoot, 'logs', 'battles', `${outputId}_train`);
  const evalLogDir = args.evalLogDir ?
    (tag ? path.join(args.evalLogDir, tag) : args.evalLogDir) :
    path.join(repoRoot, 'logs', 'battles', `${outputId}_eval`);
  return {
    runDir,
    stageDir,
    iteration,
    outputId,
    trainLogDir,
    evalLogDir,
    trainManifestPath: path.join(stageDir, 'train_games_manifest.jsonl'),
    trainFailuresPath: path.join(stageDir, 'train_failures.jsonl'),
    trainSummaryPath: path.join(stageDir, 'train_generation_summary.json'),
    evalManifestPath: path.join(stageDir, 'eval_games_manifest.jsonl'),
    evalFailuresPath: path.join(stageDir, 'eval_failures.jsonl'),
    evalSummaryPath: path.join(stageDir, 'eval_summary.json'),
    valueDatasetName: `${outputId}_value`,
    valueDatasetPath: path.join(repoRoot, 'data', 'datasets', 'value', `${outputId}_value.jsonl`),
    valueDatasetSummaryPath: path.join(repoRoot, 'data', 'datasets', 'value', `${outputId}_value.summary.json`),
    valueModelDir: path.join(repoRoot, 'models', 'value_model', `${outputId}_value`),
    valueModelPath: path.join(repoRoot, 'models', 'value_model', `${outputId}_value`, 'model.json'),
    searchDatasetName: `${outputId}_search`,
    searchDatasetPath: path.join(repoRoot, 'data', 'datasets', 'search', `${outputId}_search.jsonl`),
    searchDatasetSummaryPath: path.join(repoRoot, 'data', 'datasets', 'search', `${outputId}_search.summary.json`),
    policyModelDir: path.join(repoRoot, 'models', 'bc_policy', `${outputId}_search_policy`),
    policyModelPath: path.join(repoRoot, 'models', 'bc_policy', `${outputId}_search_policy`, 'model.json'),
  };
}

function assertInsideRepo(filePath, label) {
  const resolved = path.resolve(filePath);
  const relative = path.relative(repoRoot, resolved);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`${label} must stay inside the repo when --overwrite is used: ${resolved}`);
  }
}

function removeIfExists(filePath, label) {
  if (!fs.existsSync(filePath)) return;
  assertInsideRepo(filePath, label);
  fs.rmSync(filePath, {recursive: true, force: true});
}

function prepareOutputDirs(args, outputs) {
  if (args.overwrite) {
    if (args.stages.includes('generate')) {
      removeIfExists(outputs.trainLogDir, 'training log directory');
      removeIfExists(outputs.trainManifestPath, 'training manifest');
      removeIfExists(outputs.trainFailuresPath, 'training failures');
      removeIfExists(outputs.trainSummaryPath, 'training summary');
    }
    if (args.stages.includes('eval')) {
      removeIfExists(outputs.evalLogDir, 'evaluation log directory');
      removeIfExists(outputs.evalManifestPath, 'evaluation manifest');
      removeIfExists(outputs.evalFailuresPath, 'evaluation failures');
      removeIfExists(outputs.evalSummaryPath, 'evaluation summary');
    }
  }
  fs.mkdirSync(outputs.runDir, {recursive: true});
  fs.mkdirSync(outputs.stageDir, {recursive: true});
  fs.mkdirSync(outputs.trainLogDir, {recursive: true});
  fs.mkdirSync(outputs.evalLogDir, {recursive: true});
}

function modelPathForParticipant(participant, args, outputs) {
  if (participant.policy === 'baseline') return args.baselineModel;
  if (participant.policy === 'current') return args.currentModel;
  if (participant.policy === 'candidate') return outputs.policyModelPath;
  return null;
}

function valuePathForParticipant(participant, args, outputs) {
  if (participant.value === 'bootstrap') return args.bootstrapValueModel;
  if (participant.value === 'candidate') return outputs.valueModelPath;
  return null;
}

function assertParticipantInputs(participants, args, outputs) {
  const needed = new Set();
  for (const id of participants) {
    const participant = PARTICIPANTS[id];
    const modelPath = modelPathForParticipant(participant, args, outputs);
    const valuePath = valuePathForParticipant(participant, args, outputs);
    if (modelPath) needed.add(modelPath);
    if (valuePath) needed.add(valuePath);
  }
  for (const filePath of needed) {
    if (!fs.existsSync(filePath)) throw new Error(`Required model not found: ${filePath}`);
  }
}

function createParticipantAgent(id, args, outputs, formatId) {
  const participant = PARTICIPANTS[id];
  const options = {formatId};
  const modelPath = modelPathForParticipant(participant, args, outputs);
  const valuePath = valuePathForParticipant(participant, args, outputs);
  if (modelPath) options.modelPath = modelPath;
  if (valuePath) options.valueModelPath = valuePath;
  if (participant.agent === 'torch_policy') {
    options.pythonPath = args.pythonPath;
    options.torchDevice = args.torchDevice;
  }
  const agent = createAgent(participant.agent, options);
  return {
    ...agent,
    name: `${id}_agent`,
    displayName: participant.label,
    participantId: id,
  };
}

function sideFromWinner(winner) {
  if (typeof winner !== 'string') return 'unknown';
  const match = winner.match(/\b(P[12])$/i);
  return match ? match[1].toLowerCase() : 'unknown';
}

function countRecoveryRows(tracePath) {
  const recovery = {total: 0, by_side: {p1: 0, p2: 0, unknown: 0}};
  if (!fs.existsSync(tracePath)) return recovery;
  for (const line of fs.readFileSync(tracePath, 'utf8').split(/\r?\n/).filter(Boolean)) {
    try {
      const row = JSON.parse(line);
      if (!row.error_recovery) continue;
      const side = row.side === 'p1' || row.side === 'p2' ? row.side : 'unknown';
      recovery.total += 1;
      recovery.by_side[side] += 1;
    } catch (error) {
      recovery.total += 1;
      recovery.by_side.unknown += 1;
    }
  }
  return recovery;
}

function cleanupCompactArtifacts(result) {
  for (const relative of [result.protocol_log_path, result.replay_html_path]) {
    if (!relative) continue;
    const filePath = path.resolve(repoRoot, relative);
    assertInsideRepo(filePath, 'compact log artifact');
    if (fs.existsSync(filePath)) fs.rmSync(filePath, {force: true});
  }
}

function shouldCompactArtifacts(args, mode) {
  return args.compactLogs ||
    (mode === 'train' && args.compactTrainLogs) ||
    (mode === 'eval' && args.compactEvalLogs);
}

function ensureTableRow(table, id) {
  if (!table.has(id)) {
    table.set(id, {
      participant: id,
      games: 0,
      wins: 0,
      losses: 0,
      unknown: 0,
      recovery_rows: 0,
      win_rate: 0,
    });
  }
  return table.get(id);
}

function recordTableResult(table, id, side, winnerSide, recovery) {
  const row = ensureTableRow(table, id);
  row.games += 1;
  row.recovery_rows += recovery.by_side[side] || 0;
  if (winnerSide === side) row.wins += 1;
  else if (winnerSide === 'unknown') row.unknown += 1;
  else row.losses += 1;
}

function finalizeTable(table) {
  return [...table.values()]
    .map(row => ({
      ...row,
      win_rate: row.wins + row.losses ? row.wins / (row.wins + row.losses) : 0,
    }))
    .sort((a, b) => b.win_rate - a.win_rate || b.wins - a.wins || a.participant.localeCompare(b.participant));
}

function schedulePairings(participants, includeMirrors = false) {
  const pairings = [];
  for (let i = 0; i < participants.length; i++) {
    for (let j = i + 1; j < participants.length; j++) {
      pairings.push({p1: participants[i], p2: participants[j]});
      pairings.push({p1: participants[j], p2: participants[i]});
    }
  }
  if (includeMirrors) {
    for (const id of participants) pairings.push({p1: id, p2: id});
  }
  return pairings;
}

function readJsonl(filePath) {
  if (!fs.existsSync(filePath)) return [];
  return fs.readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map(line => JSON.parse(line));
}

function appendJsonl(filePath, row) {
  fs.mkdirSync(path.dirname(filePath), {recursive: true});
  fs.appendFileSync(filePath, `${JSON.stringify(row)}\n`, 'utf8');
}

function writeJson(filePath, data) {
  fs.mkdirSync(path.dirname(filePath), {recursive: true});
  fs.writeFileSync(filePath, `${JSON.stringify(data, null, 2)}\n`, 'utf8');
}

function summarizeOutputs(outputs) {
  return Object.fromEntries(Object.entries(outputs).map(([key, value]) => {
    if (typeof value === 'string' && path.isAbsolute(value)) return [key, relativePath(value)];
    return [key, value];
  }));
}

async function generateGames({
  args,
  outputs,
  mode,
  totalGames,
  participants,
  logDir,
  manifestPath,
  failuresPath,
  summaryPath,
}) {
  if (totalGames <= 0) {
    console.log(`${mode}: skipped because requested games is 0`);
    return;
  }
  assertParticipantInputs(participants, args, outputs);
  const existingRows = readJsonl(manifestPath);
  const completed = existingRows.length;
  if (completed >= totalGames) {
    console.log(`${mode}: ${completed}/${totalGames} games already complete; skipping generation`);
    return;
  }

  const pool = loadTeamPool();
  const pairings = schedulePairings(participants);
  if (!pairings.length) throw new Error(`${mode}: need at least two participants`);
  const table = new Map();
  let failures = readJsonl(failuresPath).length;

  for (const row of existingRows) {
    recordTableResult(table, row.p1_participant, 'p1', row.winner_side, row.recovery || {by_side: {p1: 0}});
    recordTableResult(table, row.p2_participant, 'p2', row.winner_side, row.recovery || {by_side: {p2: 0}});
  }

  for (let index = completed + 1; index <= totalGames; index++) {
    const pairing = pairings[(index - 1) % pairings.length];
    const battleSeed = `${args.seed}:${mode}:${index}:${pairing.p1}_vs_${pairing.p2}`;
    const rng = makeRng(battleSeed);
    try {
      const p1Team = findTeam(pool, null, rng);
      const p2Team = findTeam(pool, null, rng);
      const p1Lead = findLeadMode(p1Team, null, rng);
      const p2Lead = findLeadMode(p2Team, null, rng);
      const p1Agent = createParticipantAgent(pairing.p1, args, outputs, pool.format_id);
      const p2Agent = createParticipantAgent(pairing.p2, args, outputs, pool.format_id);
      const result = await runBattle({
        pool,
        seed: battleSeed,
        p1Team,
        p2Team,
        p1Lead,
        p2Lead,
        p1Agent,
        p2Agent,
        logDir,
        rng,
      });
      if (shouldCompactArtifacts(args, mode)) cleanupCompactArtifacts(result);
      const winnerSide = sideFromWinner(result.winner);
      const recovery = countRecoveryRows(path.resolve(repoRoot, result.trace_jsonl_path));
      const row = {
        index,
        mode,
        seed: battleSeed,
        p1_participant: pairing.p1,
        p2_participant: pairing.p2,
        winner: result.winner,
        winner_side: winnerSide,
        turns: result.turns,
        p1_team: result.p1.team_id,
        p2_team: result.p2.team_id,
        p1_lead: result.p1.lead_id,
        p2_lead: result.p2.lead_id,
        recovery,
        recovery_rows: recovery.total,
        summary_json_path: result.summary_json_path,
        trace_jsonl_path: result.trace_jsonl_path,
      };
      appendJsonl(manifestPath, row);
      recordTableResult(table, pairing.p1, 'p1', winnerSide, recovery);
      recordTableResult(table, pairing.p2, 'p2', winnerSide, recovery);
      if (index === 1 || index % 25 === 0 || index === totalGames) {
        console.log(
          `${mode}: game ${index}/${totalGames} ${pairing.p1} vs ${pairing.p2} ` +
          `winner=${result.winner} turns=${result.turns} recovery=${recovery.total}`
        );
      }
    } catch (error) {
      failures += 1;
      appendJsonl(failuresPath, {
        index,
        mode,
        seed: battleSeed,
        p1_participant: pairing.p1,
        p2_participant: pairing.p2,
        error: error.stack || error.message,
      });
      if (failures > args.maxFailures) throw error;
      console.warn(`${mode}: failure ${failures}/${args.maxFailures} at game ${index}: ${error.message}`);
    }
  }

  const finalRows = readJsonl(manifestPath);
  const summary = {
    created_at: new Date().toISOString(),
    run_id: args.runId,
    mode,
    seed: args.seed,
    requested_games: totalGames,
    completed_games: finalRows.length,
    failures,
    participants,
    participant_config: Object.fromEntries(participants.map(id => [id, PARTICIPANTS[id]])),
    log_dir: relativePath(logDir),
    manifest_path: relativePath(manifestPath),
    compact_logs: args.compactLogs,
    compact_train_logs: args.compactTrainLogs,
    compact_eval_logs: args.compactEvalLogs,
    agent_table: finalizeTable(table),
  };
  writeJson(summaryPath, summary);
  console.log(`${mode}: wrote summary ${relativePath(summaryPath)}`);
}

function runNodeScript(scriptRelativePath, args, options = {}) {
  const scriptPath = path.join(repoRoot, scriptRelativePath);
  const commandArgs = [scriptPath, ...args];
  console.log(`\n> ${relativePath(process.execPath)} ${scriptRelativePath} ${args.join(' ')}`);
  const result = spawnSync(process.execPath, commandArgs, {
    cwd: repoRoot,
    stdio: 'inherit',
  });
  if (result.status !== 0) {
    throw new Error(`${scriptRelativePath} failed with exit code ${result.status}`);
  }
  if (options.expectPath && !fs.existsSync(options.expectPath)) {
    throw new Error(`${scriptRelativePath} did not create expected output: ${options.expectPath}`);
  }
}

function maybeRunCliStage({stage, outputPath, overwrite, command}) {
  if (fs.existsSync(outputPath) && !overwrite) {
    console.log(`${stage}: output exists; skipping (${relativePath(outputPath)})`);
    return;
  }
  command();
}

function buildValueDataset(args, outputs) {
  maybeRunCliStage({
    stage: 'value-dataset',
    outputPath: outputs.valueDatasetPath,
    overwrite: args.overwrite,
    command: () => runNodeScript('scripts/build_value_dataset.js', [
      '--trace-dir', relativePath(outputs.trainLogDir),
      '--out-dir', 'data/datasets/value',
      '--name', outputs.valueDatasetName,
      '--agent', 'all',
      '--overwrite',
    ], {expectPath: outputs.valueDatasetPath}),
  });
}

function trainValueModel(args, outputs) {
  maybeRunCliStage({
    stage: 'train-value',
    outputPath: outputs.valueModelPath,
    overwrite: args.overwrite,
    command: () => {
      const commandArgs = [
        '--dataset', relativePath(outputs.valueDatasetPath),
        '--out-dir', relativePath(outputs.valueModelDir),
        '--epochs', String(args.valueEpochs),
        '--learning-rate', String(args.valueLearningRate),
        '--l2', String(args.valueL2),
        '--feature-dim', String(args.valueFeatureDim),
        '--seed', `${outputs.outputId}_value`,
        '--validation-split', String(args.validationSplit),
        '--eval-every', String(args.evalEvery),
      ];
      if (args.initValueModel) commandArgs.push('--init-model', relativePath(args.initValueModel));
      commandArgs.push('--overwrite');
      runNodeScript('scripts/train_value_model.js', commandArgs, {expectPath: outputs.valueModelPath});
    },
  });
}

function buildSearchDataset(args, outputs) {
  maybeRunCliStage({
    stage: 'search-dataset',
    outputPath: outputs.searchDatasetPath,
    overwrite: args.overwrite,
    command: () => {
      runNodeScript('scripts/build_search_improved_dataset.js', [
        '--dataset', relativePath(outputs.valueDatasetPath),
        '--policy-model', relativePath(args.currentModel),
        '--value-model', relativePath(outputs.valueModelPath),
        '--out-dir', 'data/datasets/search',
        '--name', outputs.searchDatasetName,
        '--progress-every', String(args.searchProgressEvery),
        '--overwrite',
      ], {expectPath: outputs.searchDatasetPath});
      runNodeScript('scripts/evaluate_search_labels.js', [
        '--dataset', relativePath(outputs.searchDatasetPath),
        '--summary', relativePath(outputs.searchDatasetSummaryPath),
        '--min-changed', '1',
      ]);
    },
  });
}

function trainPolicyModel(args, outputs) {
  maybeRunCliStage({
    stage: 'train-policy',
    outputPath: outputs.policyModelPath,
    overwrite: args.overwrite,
    command: () => {
      const commandArgs = [
        '--dataset', relativePath(outputs.searchDatasetPath),
        '--out-dir', relativePath(outputs.policyModelDir),
        '--epochs', String(args.policyEpochs),
        '--learning-rate', String(args.policyLearningRate),
        '--l2', String(args.policyL2),
        '--feature-dim', String(args.policyFeatureDim),
        '--seed', `${outputs.outputId}_policy`,
        '--validation-split', String(args.validationSplit),
        '--eval-every', String(args.evalEvery),
        '--no-feature-cache',
        '--compact-examples',
        '--train-progress-every', '1000',
      ];
      if (args.initPolicyModel) commandArgs.push('--init-model', relativePath(args.initPolicyModel));
      commandArgs.push('--overwrite');
      runNodeScript('scripts/train_bc_policy.js', commandArgs, {expectPath: outputs.policyModelPath});
    },
  });
}

function printPlan(args, outputs) {
  const trainPairings = schedulePairings(args.trainParticipants).length;
  const evalPairings = schedulePairings(args.evalParticipants).length;
  const plan = {
    run_id: args.runId,
    iterations: args.iterations,
    games_per_iteration: args.games,
    total_train_games: args.games * args.iterations,
    stages: args.stages,
    train_games_per_iteration: args.games,
    train_participants: args.trainParticipants,
    train_pairings: trainPairings,
    train_games_per_pairing_approx: args.games / trainPairings,
    eval_games_per_iteration: args.evalGames,
    total_eval_games: args.evalGames * args.iterations,
    eval_participants: args.evalParticipants,
    eval_pairings: evalPairings,
    eval_games_per_pairing_approx: evalPairings ? args.evalGames / evalPairings : 0,
    train_log_dir: relativePath(outputs.trainLogDir),
    eval_log_dir: relativePath(outputs.evalLogDir),
    value_dataset: relativePath(outputs.valueDatasetPath),
    value_model: relativePath(outputs.valueModelPath),
    search_dataset: relativePath(outputs.searchDatasetPath),
    policy_model: relativePath(outputs.policyModelPath),
    compact_logs: args.compactLogs,
    compact_train_logs: args.compactTrainLogs,
    compact_eval_logs: args.compactEvalLogs,
    overwrite: args.overwrite,
  };
  console.log(JSON.stringify(plan, null, 2));
}

async function runPipelineIteration(args, outputs) {
  prepareOutputDirs(args, outputs);

  if (args.stages.includes('generate')) {
    await generateGames({
      args,
      outputs,
      mode: 'train',
      totalGames: args.games,
      participants: args.trainParticipants,
      logDir: outputs.trainLogDir,
      manifestPath: outputs.trainManifestPath,
      failuresPath: outputs.trainFailuresPath,
      summaryPath: outputs.trainSummaryPath,
    });
  }
  if (args.stages.includes('value-dataset')) buildValueDataset(args, outputs);
  if (args.stages.includes('train-value')) trainValueModel(args, outputs);
  if (args.stages.includes('search-dataset')) buildSearchDataset(args, outputs);
  if (args.stages.includes('train-policy')) trainPolicyModel(args, outputs);
  if (args.stages.includes('eval') && args.evalGames > 0) {
    await generateGames({
      args,
      outputs,
      mode: 'eval',
      totalGames: args.evalGames,
      participants: args.evalParticipants,
      logDir: outputs.evalLogDir,
      manifestPath: outputs.evalManifestPath,
      failuresPath: outputs.evalFailuresPath,
      summaryPath: outputs.evalSummaryPath,
    });
  } else if (args.stages.includes('eval')) {
    console.log('eval: skipped because --eval-games is 0');
  }

  writeJson(path.join(outputs.stageDir, 'pipeline_summary.json'), {
    completed_at: new Date().toISOString(),
    run_id: args.runId,
    iteration: outputs.iteration,
    output_id: outputs.outputId,
    stages: args.stages,
    current_model_path: args.currentModel ? relativePath(args.currentModel) : null,
    bootstrap_value_model_path: args.bootstrapValueModel ? relativePath(args.bootstrapValueModel) : null,
    init_policy_model_path: args.initPolicyModel ? relativePath(args.initPolicyModel) : null,
    init_value_model_path: args.initValueModel ? relativePath(args.initValueModel) : null,
    outputs: summarizeOutputs(outputs),
  });
  console.log(`Iteration done. Summary: ${relativePath(path.join(outputs.stageDir, 'pipeline_summary.json'))}`);
}

function writeRootSummary(args, iterationSummaries) {
  const rootOutputs = outputsForArgs(args, args.iterations > 1 ? 1 : null);
  const rootSummaryPath = path.join(rootOutputs.runDir, 'pipeline_summary.json');
  writeJson(rootSummaryPath, {
    completed_at: new Date().toISOString(),
    run_id: args.runId,
    iterations: args.iterations,
    games_per_iteration: args.games,
    eval_games_per_iteration: args.evalGames,
    stages: args.stages,
    compact_logs: args.compactLogs,
    compact_train_logs: args.compactTrainLogs,
    compact_eval_logs: args.compactEvalLogs,
    iterations_completed: iterationSummaries.length,
    iteration_summaries: iterationSummaries,
  });
  console.log(`Done. Summary: ${relativePath(rootSummaryPath)}`);
}

async function runPipeline(args) {
  const firstOutputs = outputsForArgs(args, args.iterations > 1 ? 1 : null);
  printPlan(args, firstOutputs);
  if (args.dryRun) return;

  let playPolicyModel = args.currentModel;
  let playValueModel = args.bootstrapValueModel;
  let initPolicyModel = args.initPolicyModel || (args.iterations > 1 ? args.currentModel : null);
  let initValueModel = args.initValueModel || (args.iterations > 1 ? args.bootstrapValueModel : null);
  const iterationSummaries = [];

  for (let iteration = 1; iteration <= args.iterations; iteration++) {
    const outputs = outputsForArgs(args, args.iterations > 1 ? iteration : null);
    const tag = args.iterations > 1 ? iterationTag(iteration) : 'single';
    console.log(`\n=== Iteration ${iteration}/${args.iterations} (${tag}) ===`);
    const iterationArgs = {
      ...args,
      seed: args.iterations > 1 ? `${args.seed}:${tag}` : args.seed,
      currentModel: playPolicyModel,
      bootstrapValueModel: playValueModel,
      initPolicyModel,
      initValueModel,
    };

    await runPipelineIteration(iterationArgs, outputs);

    if (fs.existsSync(outputs.policyModelPath)) {
      playPolicyModel = outputs.policyModelPath;
      initPolicyModel = outputs.policyModelPath;
    }
    if (fs.existsSync(outputs.valueModelPath)) {
      playValueModel = outputs.valueModelPath;
      initValueModel = outputs.valueModelPath;
    }
    iterationSummaries.push({
      iteration,
      output_id: outputs.outputId,
      summary_path: relativePath(path.join(outputs.stageDir, 'pipeline_summary.json')),
      train_summary_path: relativePath(outputs.trainSummaryPath),
      eval_summary_path: relativePath(outputs.evalSummaryPath),
      value_model_path: fs.existsSync(outputs.valueModelPath) ? relativePath(outputs.valueModelPath) : null,
      policy_model_path: fs.existsSync(outputs.policyModelPath) ? relativePath(outputs.policyModelPath) : null,
    });
  }

  writeRootSummary(args, iterationSummaries);
}

function main() {
  runPipeline(parseArgs(process.argv.slice(2)))
    .finally(() => closeTorchPolicyScorers())
    .catch(error => {
    console.error(`FAIL ${error.stack || error.message}`);
    process.exit(1);
  });
}

main();
