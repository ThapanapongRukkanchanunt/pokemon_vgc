const fs = require('node:fs');
const path = require('node:path');
const {enumerateLegalActions} = require('../src/agents/legal_actions');
const {dexForFormat} = require('../src/battle/showdown_protocol');
const {contextFromExample} = require('../src/bc/feature_encoder');
const {loadModel: loadPolicyModel, scoreChoices: scorePolicyChoices} = require('../src/bc/linear_policy');
const {PolicyValueRiskSelector, RISK_MODE_WEIGHTS} = require('../src/selectors');
const {loadModel: loadValueModel, scoreChoices: scoreValueChoices} = require('../src/value/linear_value');

const repoRoot = path.join(__dirname, '..');
const MODES = Object.keys(RISK_MODE_WEIGHTS);

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
    outDir: path.join(repoRoot, 'experiments', 'risk_modes', 'phase5_samples'),
    limit: 300,
    maxCases: 8,
    requestTypes: new Set(['move', 'force_switch']),
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
    } else if (arg === '--limit') {
      args.limit = parseInteger(argv[++i], '--limit');
    } else if (arg === '--max-cases') {
      args.maxCases = parseInteger(argv[++i], '--max-cases');
    } else if (arg === '--request-types') {
      args.requestTypes = new Set(argv[++i].split(',').map(value => value.trim()).filter(Boolean));
    } else if (arg === '--overwrite') {
      args.overwrite = true;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (args.limit <= 0) throw new Error('--limit must be > 0');
  if (args.maxCases < 0) throw new Error('--max-cases must be >= 0');
  return args;
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function round(value, digits = 4) {
  return Number.isFinite(value) ? Number(value.toFixed(digits)) : null;
}

function loadJsonl(filePath, limit) {
  const rows = [];
  for (const line of fs.readFileSync(filePath, 'utf8').split(/\r?\n/)) {
    if (!line.trim()) continue;
    rows.push(JSON.parse(line));
    if (rows.length >= limit) break;
  }
  return rows;
}

function commandKind(choice) {
  const first = String(choice || '').split(',')[0].trim().split(/\s+/)[0];
  return first || 'unknown';
}

function battleStateFromExample(example) {
  return {
    turns: example.turn || 0,
    team: {id: example.team || 'unknown'},
    leadMode: {id: example.lead || 'unknown', team_spec: ''},
    publicState: example.state?.public_state || example.public_state || {},
  };
}

function summarizeSelection(selection) {
  const raw = selection.scoreBreakdown?.raw || {};
  const components = selection.scoreBreakdown?.components || {};
  const weighted = selection.scoreBreakdown?.weighted || {};
  return {
    choice: selection.choice,
    score: round(selection.score),
    kind: commandKind(selection.choice),
    raw: {
      policy_log_prob: round(raw.policyLogProb),
      value: round(raw.value),
      tactic: round(raw.tactic),
      safety: round(raw.safety),
    },
    normalized: {
      policy: round(components.policy),
      value: round(components.value),
      tactic: round(components.tactic),
      safety: round(components.safety),
    },
    weighted: {
      policy: round(weighted.policy),
      value: round(weighted.value),
      tactic: round(weighted.tactic),
      safety: round(weighted.safety),
    },
  };
}

function emptyModeStats() {
  return {
    decisions: 0,
    selected_kinds: {},
    raw_value_sum: 0,
    raw_tactic_sum: 0,
    raw_safety_sum: 0,
    mean_raw_value: 0,
    mean_raw_tactic: 0,
    mean_raw_safety: 0,
  };
}

function updateModeStats(stats, selection) {
  const summary = summarizeSelection(selection);
  stats.decisions += 1;
  stats.selected_kinds[summary.kind] = (stats.selected_kinds[summary.kind] || 0) + 1;
  stats.raw_value_sum += summary.raw.value || 0;
  stats.raw_tactic_sum += summary.raw.tactic || 0;
  stats.raw_safety_sum += summary.raw.safety || 0;
}

function finalizeModeStats(stats) {
  if (!stats.decisions) return stats;
  stats.mean_raw_value = round(stats.raw_value_sum / stats.decisions);
  stats.mean_raw_tactic = round(stats.raw_tactic_sum / stats.decisions);
  stats.mean_raw_safety = round(stats.raw_safety_sum / stats.decisions);
  delete stats.raw_value_sum;
  delete stats.raw_tactic_sum;
  delete stats.raw_safety_sum;
  return stats;
}

function disagreementDeltas(selections) {
  const stable = selections.stable ? summarizeSelection(selections.stable) : null;
  const comeback = selections.comeback ? summarizeSelection(selections.comeback) : null;
  const closing = selections.closing ? summarizeSelection(selections.closing) : null;
  if (!stable || !comeback) return {};
  return {
    stable_vs_comeback: {
      same_choice: stable.choice === comeback.choice,
      stable_value_minus_comeback: round((stable.raw.value || 0) - (comeback.raw.value || 0)),
      stable_safety_minus_comeback: round((stable.raw.safety || 0) - (comeback.raw.safety || 0)),
      comeback_tactic_minus_stable: round((comeback.raw.tactic || 0) - (stable.raw.tactic || 0)),
    },
    closing_vs_comeback: closing ? {
      same_choice: closing.choice === comeback.choice,
      closing_value_minus_comeback: round((closing.raw.value || 0) - (comeback.raw.value || 0)),
      closing_safety_minus_comeback: round((closing.raw.safety || 0) - (comeback.raw.safety || 0)),
      comeback_tactic_minus_closing: round((comeback.raw.tactic || 0) - (closing.raw.tactic || 0)),
    } : null,
  };
}

function ensureOutput(args) {
  const summaryPath = path.join(args.outDir, 'summary.json');
  if (!args.overwrite && fs.existsSync(summaryPath)) {
    throw new Error(`Summary exists at ${relativePath(summaryPath)}; pass --overwrite to replace it`);
  }
  fs.mkdirSync(args.outDir, {recursive: true});
  return summaryPath;
}

function compareRiskModes(args) {
  if (!fs.existsSync(args.dataset)) throw new Error(`Dataset not found: ${args.dataset}`);
  if (!fs.existsSync(args.policyModel)) throw new Error(`Policy model not found: ${args.policyModel}`);
  if (!fs.existsSync(args.valueModel)) throw new Error(`Value model not found: ${args.valueModel}`);

  const summaryPath = ensureOutput(args);
  const policyModel = loadPolicyModel(args.policyModel);
  const valueModel = loadValueModel(args.valueModel);
  const examples = loadJsonl(args.dataset, args.limit);
  const selectorsByFormat = new Map();
  const modeStats = Object.fromEntries(MODES.map(mode => [mode, emptyModeStats()]));
  const pairDisagreements = {
    stable_vs_comeback: 0,
    closing_vs_comeback: 0,
    any: 0,
  };
  const cases = [];
  let considered = 0;
  let scored = 0;

  for (const example of examples) {
    if (!args.requestTypes.has(example.request_type || 'unknown')) continue;
    const request = example.state?.request;
    if (!request) continue;

    const formatId = example.format || 'vgc';
    const dex = dexForFormat(formatId);
    const battleState = battleStateFromExample(example);
    const legalActions = enumerateLegalActions({side: example.side, request, battleState, dex});
    if (legalActions.length < 2) continue;

    considered += 1;
    const context = contextFromExample(example);
    const modelScores = scorePolicyChoices(policyModel, context, legalActions);
    const valueScores = scoreValueChoices(valueModel, context, legalActions);
    if (!selectorsByFormat.has(formatId)) {
      selectorsByFormat.set(formatId, Object.fromEntries(MODES.map(mode => [
        mode,
        new PolicyValueRiskSelector({formatId, riskMode: mode}),
      ])));
    }
    const selectors = selectorsByFormat.get(formatId);
    const selections = {};

    for (const mode of MODES) {
      const selection = selectors[mode].choose({
        state: context,
        side: example.side,
        request,
        legalActions,
        modelScores,
        valueScores,
        battleState,
        rng: null,
        dex,
      });
      if (!selection) continue;
      selections[mode] = selection;
      updateModeStats(modeStats[mode], selection);
    }

    if (Object.keys(selections).length !== MODES.length) continue;
    scored += 1;
    const choices = new Set(MODES.map(mode => selections[mode].choice));
    if (choices.size > 1) {
      pairDisagreements.any += 1;
      if (selections.stable.choice !== selections.comeback.choice) pairDisagreements.stable_vs_comeback += 1;
      if (selections.closing.choice !== selections.comeback.choice) pairDisagreements.closing_vs_comeback += 1;
      if (cases.length < args.maxCases) {
        cases.push({
          example_id: example.example_id,
          battle_id: example.battle_id,
          turn: example.turn,
          side: example.side,
          request_type: example.request_type,
          trace_action: example.action,
          target: example.target,
          mode_choices: Object.fromEntries(MODES.map(mode => [mode, summarizeSelection(selections[mode])])),
          deltas: disagreementDeltas(selections),
        });
      }
    }
  }

  for (const stats of Object.values(modeStats)) finalizeModeStats(stats);
  const summary = {
    created_at: new Date().toISOString(),
    dataset_path: relativePath(args.dataset),
    policy_model_path: relativePath(args.policyModel),
    value_model_path: relativePath(args.valueModel),
    request_types: [...args.requestTypes],
    limit: args.limit,
    examples_loaded: examples.length,
    examples_considered: considered,
    examples_scored: scored,
    disagreements: {
      ...pairDisagreements,
      any_rate: scored ? pairDisagreements.any / scored : 0,
      stable_vs_comeback_rate: scored ? pairDisagreements.stable_vs_comeback / scored : 0,
      closing_vs_comeback_rate: scored ? pairDisagreements.closing_vs_comeback / scored : 0,
    },
    risk_mode_weights: RISK_MODE_WEIGHTS,
    mode_stats: modeStats,
    sample_cases: cases,
  };

  fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  console.log(`scored=${scored} disagreements=${pairDisagreements.any} ` +
    `stable_vs_comeback=${pairDisagreements.stable_vs_comeback} ` +
    `closing_vs_comeback=${pairDisagreements.closing_vs_comeback}`);
  for (const item of cases.slice(0, 3)) {
    console.log([
      item.example_id,
      `stable=${item.mode_choices.stable.choice}`,
      `balanced=${item.mode_choices.balanced.choice}`,
      `comeback=${item.mode_choices.comeback.choice}`,
      `closing=${item.mode_choices.closing.choice}`,
    ].join(' | '));
  }
  console.log(`Wrote summary: ${relativePath(summaryPath)}`);
  return {summaryPath, summary};
}

function main() {
  compareRiskModes(parseArgs(process.argv.slice(2)));
}

main();
