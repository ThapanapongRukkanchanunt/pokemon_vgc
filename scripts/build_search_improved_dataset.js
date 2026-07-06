const fs = require('node:fs');
const path = require('node:path');
const {enumerateLegalActions} = require('../src/agents/legal_actions');
const {dexForFormat} = require('../src/battle/showdown_protocol');
const {contextFromExample} = require('../src/bc/feature_encoder');
const {loadModel: loadPolicyModel, scoreChoices: scorePolicyChoices} = require('../src/bc/linear_policy');
const {PolicySelector, ShallowSearchSelector} = require('../src/selectors');
const {loadModel: loadValueModel, scoreChoices: scoreValueChoices} = require('../src/value/linear_value');

const repoRoot = path.join(__dirname, '..');

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    dataset: path.join(repoRoot, 'data', 'datasets', 'value', 'phase4_mixed_q.jsonl'),
    policyModel: path.join(repoRoot, 'models', 'bc_policy', 'trace_test_maxdamage', 'model.json'),
    valueModel: path.join(repoRoot, 'models', 'value_model', 'phase4_mixed_q', 'model.json'),
    outDir: path.join(repoRoot, 'data', 'datasets', 'search'),
    name: 'phase6_search_improved',
    limit: null,
    requestTypes: new Set(['move', 'force_switch']),
    maxOpponentActions: 4,
    progressEvery: 500,
    riskMode: 'balanced',
    overwrite: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--dataset') {
      args.dataset = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--policy-model' || arg === '--model') {
      args.policyModel = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--value-model') {
      args.valueModel = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--out-dir') {
      args.outDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--name') {
      args.name = argv[++i];
    } else if (arg === '--limit') {
      args.limit = parseInteger(argv[++i], '--limit');
    } else if (arg === '--request-types') {
      args.requestTypes = new Set(argv[++i].split(',').map(value => value.trim()).filter(Boolean));
    } else if (arg === '--max-opponent-actions') {
      args.maxOpponentActions = parseInteger(argv[++i], '--max-opponent-actions');
    } else if (arg === '--progress-every') {
      args.progressEvery = parseInteger(argv[++i], '--progress-every');
    } else if (arg === '--risk-mode') {
      args.riskMode = argv[++i];
    } else if (arg === '--overwrite') {
      args.overwrite = true;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (!args.name || /[\\/:*?"<>|]/.test(args.name)) {
    throw new Error('--name must be a non-empty filename-safe value');
  }
  if (args.limit != null && args.limit <= 0) throw new Error('--limit must be > 0');
  if (!args.requestTypes.size) throw new Error('--request-types must include at least one type');
  if (args.maxOpponentActions <= 0) throw new Error('--max-opponent-actions must be > 0');
  if (args.progressEvery < 0) throw new Error('--progress-every must be >= 0');
  if (!['stable', 'balanced', 'comeback', 'closing'].includes(args.riskMode)) {
    throw new Error('--risk-mode must be one of stable, balanced, comeback, closing');
  }
  return args;
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function sideOpponent(side) {
  return side === 'p1' ? 'p2' : 'p1';
}

function round(value, digits = 6) {
  return Number.isFinite(value) ? Number(value.toFixed(digits)) : null;
}

function* readJsonlRows(filePath, limit = null) {
  if (!fs.existsSync(filePath)) throw new Error(`Dataset not found: ${filePath}`);
  const fd = fs.openSync(filePath, 'r');
  const buffer = Buffer.alloc(1024 * 1024);
  let leftover = '';
  let lineNumber = 0;
  let yielded = 0;
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
        if (!line.trim()) continue;
        try {
          yield JSON.parse(line);
        } catch (error) {
          throw new Error(`${relativePath(filePath)}:${lineNumber} invalid JSON: ${error.message}`);
        }
        yielded += 1;
        if (limit != null && yielded >= limit) return;
      }
    }
    if (leftover) {
      lineNumber += 1;
      let line = leftover;
      if (line.endsWith('\r')) line = line.slice(0, -1);
      if (line.trim()) {
        try {
          yield JSON.parse(line);
        } catch (error) {
          throw new Error(`${relativePath(filePath)}:${lineNumber} invalid JSON: ${error.message}`);
        }
      }
    }
  } finally {
    fs.closeSync(fd);
  }
}

function sourceGroupKey(example) {
  if (example?.battle_id) return `battle:${example.battle_id}`;
  if (example?.source_trace_path) return `trace:${example.source_trace_path}`;
  return 'unknown';
}

function processSourceExample({
  args,
  source,
  sourceIndex,
  total,
  summary,
  datasetFd,
  margin,
  opponentIndex,
  policyModel,
  valueModel,
  policySelector,
  searchSelectors,
  startedAt,
}) {
  const requestType = source.request_type || 'unknown';
  if (!args.requestTypes.has(requestType)) {
    summary.skipped.request_type += 1;
    logProgress({args, startMs: startedAt, processed: sourceIndex + 1, total, summary});
    return margin;
  }
  const request = source.state?.request;
  if (!request) {
    summary.skipped.missing_request += 1;
    logProgress({args, startMs: startedAt, processed: sourceIndex + 1, total, summary});
    return margin;
  }
  const formatId = source.format || 'vgc';
  const dex = dexForFormat(formatId);
  const opponentExample = pairedOpponent(source, opponentIndex);
  const battleState = battleStateFromExample(source, opponentExample);
  const legalActions = enumerateLegalActions({side: source.side, request, battleState, dex});
  if (legalActions.length < 2) {
    summary.skipped.legal_actions += 1;
    logProgress({args, startMs: startedAt, processed: sourceIndex + 1, total, summary});
    return margin;
  }
  const context = contextFromExample(source);
  const modelScores = scorePolicyChoices(policyModel, context, legalActions);
  const valueScores = scoreValueChoices(valueModel, context, legalActions);
  let opponentLegalActions = null;
  let opponentModelScores = null;
  let opponentValueScores = null;
  if (opponentExample?.state?.request) {
    const foeSide = sideOpponent(source.side);
    opponentLegalActions = enumerateLegalActions({
      side: foeSide,
      request: opponentExample.state.request,
      battleState,
      dex,
    });
    const opponentContext = contextFromExample(opponentExample);
    opponentModelScores = scorePolicyChoices(policyModel, opponentContext, opponentLegalActions);
    opponentValueScores = scoreValueChoices(valueModel, opponentContext, opponentLegalActions);
  }
  if (!searchSelectors.has(formatId)) {
    searchSelectors.set(formatId, new ShallowSearchSelector({
      formatId,
      riskMode: args.riskMode,
      maxOpponentActions: args.maxOpponentActions,
    }));
  }
  const searchSelector = searchSelectors.get(formatId);
  const policySelection = policySelector.choose({
    request,
    legalActions,
    modelScores,
    rng: null,
  });
  const searchSelection = searchSelector.choose({
    state: context,
    side: source.side,
    request,
    legalActions,
    modelScores,
    valueScores,
    opponentRequest: opponentExample?.state?.request || null,
    opponentLegalActions,
    opponentModelScores,
    opponentValueScores,
    battleState,
    rng: null,
    dex,
  });
  if (!searchSelection?.choice) {
    summary.skipped.search_selection += 1;
    logProgress({args, startMs: startedAt, processed: sourceIndex + 1, total, summary});
    return margin;
  }

  const labelAction = searchSelection.choice;
  const legalChoices = choiceRows(legalActions);
  if (!legalChoices.includes(labelAction)) {
    summary.skipped.invalid_label += 1;
    logProgress({args, startMs: startedAt, processed: sourceIndex + 1, total, summary});
    return margin;
  }
  const example = buildExample({
    source,
    legalActions,
    labelAction,
    policySelection,
    searchSelection,
    opponentExample,
  });
  fs.writeSync(datasetFd, `${JSON.stringify(example)}\n`);

  summary.examples += 1;
  inc(summary.counts.request_types, example.request_type);
  inc(summary.counts.original_agents, example.original_agent);
  inc(summary.counts.teams, example.team);
  inc(summary.counts.search_applied, example.search_metadata.search_applied);
  inc(summary.counts.paired_opponent_rows, Boolean(opponentExample));
  if (example.original_action !== example.label_action) summary.changed_vs_original += 1;
  if (example.policy_action && example.policy_action !== example.label_action) summary.changed_vs_policy += 1;
  const searchMargin = example.search_metadata.search_margin_over_policy;
  if (Number.isFinite(searchMargin)) {
    summary.policy_comparison.compared += 1;
    margin += searchMargin;
    if (searchMargin > 1e-9) summary.policy_comparison.search_better += 1;
    else if (searchMargin < -1e-9) summary.policy_comparison.policy_better += 1;
    else summary.policy_comparison.ties += 1;
  }
  logProgress({args, startMs: startedAt, processed: sourceIndex + 1, total, summary});
  return margin;
}

function processBattleRows({
  args,
  rows,
  sourceIndexStart,
  total,
  summary,
  datasetFd,
  marginSum,
  policyModel,
  valueModel,
  policySelector,
  searchSelectors,
  startedAt,
}) {
  const opponentIndex = buildOpponentIndex(rows);
  let margin = marginSum;
  rows.forEach((source, offset) => {
    margin = processSourceExample({
      args,
      source,
      sourceIndex: sourceIndexStart + offset,
      total,
      summary,
      datasetFd,
      margin,
      opponentIndex,
      policyModel,
      valueModel,
      policySelector,
      searchSelectors,
      startedAt,
    });
  });
  return margin;
}

function choiceText(action) {
  return typeof action === 'string' ? action : action.choice;
}

function choiceRows(actions) {
  return actions.map(action => choiceText(action));
}

function sourceAction(example) {
  return example.action || example.chosen_action || example.label_action;
}

function sourceWinnerSide(example) {
  if (example.winner_side === 'p1' || example.winner_side === 'p2') return example.winner_side;
  if (typeof example.winner !== 'string') return null;
  const match = example.winner.match(/\b(P[12])$/i);
  return match ? match[1].toLowerCase() : null;
}

function buildOpponentIndex(examples) {
  const index = new Map();
  for (const example of examples) {
    const key = `${example.battle_id}|${example.turn}|${example.side}|${example.request_type || 'unknown'}`;
    if (!index.has(key)) index.set(key, example);
  }
  return index;
}

function pairedOpponent(example, opponentIndex) {
  const key = `${example.battle_id}|${example.turn}|${sideOpponent(example.side)}|${example.request_type || 'unknown'}`;
  return opponentIndex.get(key) || null;
}

function battleStateFromExample(example, opponentExample = null) {
  const foeSide = sideOpponent(example.side);
  return {
    turns: example.turn || 0,
    team: {id: example.team || 'unknown'},
    leadMode: {id: example.lead || 'unknown', team_spec: ''},
    teams: {
      [example.side]: {id: example.team || 'unknown'},
      [foeSide]: {id: opponentExample?.team || 'unknown'},
    },
    leadModes: {
      [example.side]: {id: example.lead || 'unknown', team_spec: ''},
      [foeSide]: {id: opponentExample?.lead || 'unknown', team_spec: ''},
    },
    requests: {
      [example.side]: example.state?.request || null,
      [foeSide]: opponentExample?.state?.request || null,
    },
    publicState: example.state?.public_state || example.public_state || {},
  };
}

function ensureOutput(args) {
  fs.mkdirSync(args.outDir, {recursive: true});
  const datasetPath = path.join(args.outDir, `${args.name}.jsonl`);
  const summaryPath = path.join(args.outDir, `${args.name}.summary.json`);
  if (!args.overwrite && (fs.existsSync(datasetPath) || fs.existsSync(summaryPath))) {
    throw new Error(`Output already exists for ${args.name}; pass --overwrite to replace it`);
  }
  return {datasetPath, summaryPath};
}

function inc(map, key) {
  const normalizedKey = key == null ? 'null' : String(key);
  map[normalizedKey] = (map[normalizedKey] || 0) + 1;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return 'unknown';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  if (minutes < 60) return `${minutes}m ${rest}s`;
  const hours = Math.floor(minutes / 60);
  const minuteRest = minutes % 60;
  return `${hours}h ${minuteRest}m`;
}

function logProgress({args, startMs, processed, total, summary}) {
  if (!args.progressEvery || processed <= 0) return;
  const hasTotal = Number.isFinite(total) && total > 0;
  if (hasTotal) {
    if (processed !== total && processed % args.progressEvery !== 0) return;
  } else if (processed % args.progressEvery !== 0) {
    return;
  }
  const elapsedSeconds = (Date.now() - startMs) / 1000;
  const rate = processed / Math.max(elapsedSeconds, 1e-9);
  const etaSeconds = hasTotal ? (total - processed) / Math.max(rate, 1e-9) : null;
  const processedLabel = hasTotal ? `${processed}/${total}` : String(processed);
  console.log(
    `search-labels: processed=${processedLabel} produced=${summary.examples} ` +
    `changed_vs_policy=${summary.changed_vs_policy} rate=${rate.toFixed(1)}/s ` +
    `elapsed=${formatDuration(elapsedSeconds)} eta=${hasTotal ? formatDuration(etaSeconds) : 'unknown'}`
  );
}

function scoreForChoice(selection, choice) {
  const row = (selection?.scores || []).find(candidate => candidate.choice === choice);
  return Number.isFinite(row?.score) ? row.score : null;
}

function buildExample({
  source,
  legalActions,
  labelAction,
  policySelection,
  searchSelection,
  opponentExample,
}) {
  const legalChoices = choiceRows(legalActions);
  const winnerSide = sourceWinnerSide(source);
  const originalAction = sourceAction(source);
  const policyAction = policySelection?.choice || null;
  const searchScore = scoreForChoice(searchSelection, labelAction);
  const policySearchScore = policyAction ? scoreForChoice(searchSelection, policyAction) : null;

  return {
    example_id: `${source.example_id}:search`,
    source_trace_path: source.source_trace_path || 'unknown',
    source_example_id: source.example_id,
    battle_id: source.battle_id,
    seed: source.seed,
    format: source.format || 'vgc',
    turn: source.turn,
    side: source.side,
    agent: 'search_improved_labeler',
    original_agent: source.agent || 'unknown',
    team: source.team,
    lead: source.lead,
    request_type: source.request_type || 'unknown',
    state: {
      request: source.state?.request,
      public_state: source.state?.public_state,
    },
    legal_actions: legalChoices,
    label_action: labelAction,
    label_action_index: legalChoices.indexOf(labelAction),
    original_action: originalAction,
    policy_action: policyAction,
    winner: source.winner || null,
    winner_side: winnerSide,
    win_target: winnerSide == null ? null : (winnerSide === source.side ? 1 : 0),
    is_recovery: Boolean(source.is_recovery),
    search_metadata: {
      selector: 'shallow_search',
      risk_mode: searchSelection?.riskMode || 'balanced',
      search_applied: Boolean(searchSelection?.searchApplied),
      opponent_example_id: opponentExample?.example_id || null,
      opponent_action_count: searchSelection?.scoreBreakdown?.sampled_opponent_actions || 0,
      search_score: round(searchScore),
      policy_choice_search_score: round(policySearchScore),
      search_margin_over_policy: round(
        Number.isFinite(searchScore) && Number.isFinite(policySearchScore) ?
          searchScore - policySearchScore :
          null
      ),
      score_breakdown: searchSelection?.scoreBreakdown || null,
    },
  };
}

function buildDataset(args) {
  if (!fs.existsSync(args.policyModel)) throw new Error(`Policy model not found: ${args.policyModel}`);
  if (!fs.existsSync(args.valueModel)) throw new Error(`Value model not found: ${args.valueModel}`);
  const {datasetPath, summaryPath} = ensureOutput(args);
  const policyModel = loadPolicyModel(args.policyModel);
  const valueModel = loadValueModel(args.valueModel);
  const policySelector = new PolicySelector();
  const searchSelectors = new Map();
  const datasetFd = fs.openSync(datasetPath, 'w');
  const total = args.limit == null ? null : args.limit;
  const summary = {
    dataset_id: args.name,
    created_at: new Date().toISOString(),
    source_dataset_path: relativePath(args.dataset),
    output_path: relativePath(datasetPath),
    summary_path: relativePath(summaryPath),
    policy_model_path: relativePath(args.policyModel),
    value_model_path: relativePath(args.valueModel),
    request_types: [...args.requestTypes],
    risk_mode: args.riskMode,
    max_opponent_actions: args.maxOpponentActions,
    examples_loaded: 0,
    examples: 0,
    skipped: {
      request_type: 0,
      missing_request: 0,
      legal_actions: 0,
      search_selection: 0,
      invalid_label: 0,
    },
    counts: {
      request_types: {},
      original_agents: {},
      teams: {},
      search_applied: {},
      paired_opponent_rows: {},
    },
    changed_vs_original: 0,
    changed_vs_policy: 0,
    policy_comparison: {
      compared: 0,
      search_better: 0,
      ties: 0,
      policy_better: 0,
      mean_margin: 0,
    },
  };
  let marginSum = 0;
  const startedAt = Date.now();
  let currentGroupKey = null;
  let currentGroupRows = [];
  let currentGroupStartIndex = 0;

  try {
    for (const source of readJsonlRows(args.dataset, args.limit)) {
      const groupKey = sourceGroupKey(source);
      if (currentGroupRows.length && groupKey !== currentGroupKey) {
        marginSum = processBattleRows({
          args,
          rows: currentGroupRows,
          sourceIndexStart: currentGroupStartIndex,
          total,
          summary,
          datasetFd,
          marginSum,
          policyModel,
          valueModel,
          policySelector,
          searchSelectors,
          startedAt,
        });
        currentGroupRows = [];
      }

      if (!currentGroupRows.length) {
        currentGroupKey = groupKey;
        currentGroupStartIndex = summary.examples_loaded;
      }
      currentGroupRows.push(source);
      summary.examples_loaded += 1;
    }
    if (currentGroupRows.length) {
      marginSum = processBattleRows({
        args,
        rows: currentGroupRows,
        sourceIndexStart: currentGroupStartIndex,
        total,
        summary,
        datasetFd,
        marginSum,
        policyModel,
        valueModel,
        policySelector,
        searchSelectors,
        startedAt,
      });
    }
  } finally {
    fs.closeSync(datasetFd);
  }

  if (!summary.examples) {
    fs.rmSync(datasetPath, {force: true});
    throw new Error('No search-improved examples were produced');
  }
  summary.policy_comparison.mean_margin = summary.policy_comparison.compared ?
    marginSum / summary.policy_comparison.compared :
    0;

  fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  console.log(`Wrote ${summary.examples} search-improved examples`);
  console.log(`Changed vs policy: ${summary.changed_vs_policy}`);
  console.log(`Mean search margin over policy: ${summary.policy_comparison.mean_margin.toFixed(6)}`);
  console.log(`Dataset: ${relativePath(datasetPath)}`);
  console.log(`Summary: ${relativePath(summaryPath)}`);
  return {datasetPath, summaryPath, summary};
}

function main() {
  try {
    buildDataset(parseArgs(process.argv.slice(2)));
  } catch (error) {
    console.error(`FAIL ${error.stack || error.message}`);
    process.exit(1);
  }
}

main();
