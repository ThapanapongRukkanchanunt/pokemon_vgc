const fs = require('node:fs');
const path = require('node:path');

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
    dataset: path.join(repoRoot, 'data', 'datasets', 'search', 'phase6_search_improved.jsonl'),
    summary: path.join(repoRoot, 'data', 'datasets', 'search', 'phase6_search_improved.summary.json'),
    minMeanMargin: 0,
    minChanged: 1,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--dataset') {
      args.dataset = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--summary') {
      args.summary = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--min-mean-margin') {
      args.minMeanMargin = parseNumber(argv[++i], '--min-mean-margin');
    } else if (arg === '--min-changed') {
      args.minChanged = parseInteger(argv[++i], '--min-changed');
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }
  return args;
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function loadJsonl(filePath) {
  if (!fs.existsSync(filePath)) throw new Error(`Dataset not found: ${filePath}`);
  return fs.readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`${relativePath(filePath)}:${index + 1} invalid JSON: ${error.message}`);
      }
    });
}

function evaluate(args) {
  const examples = loadJsonl(args.dataset);
  if (!examples.length) throw new Error('Dataset has no examples');
  const errors = [];
  const stats = {
    examples: examples.length,
    compared: 0,
    search_better: 0,
    ties: 0,
    policy_better: 0,
    changed_vs_policy: 0,
    margin_sum: 0,
    mean_margin: 0,
  };

  examples.forEach((example, index) => {
    const prefix = `${relativePath(args.dataset)}:${index + 1}`;
    if (!Array.isArray(example.legal_actions) || !example.legal_actions.includes(example.label_action)) {
      errors.push(`${prefix} label_action is not legal`);
    }
    if (example.policy_action && example.policy_action !== example.label_action) {
      stats.changed_vs_policy += 1;
    }
    const margin = example.search_metadata?.search_margin_over_policy;
    if (Number.isFinite(margin)) {
      stats.compared += 1;
      stats.margin_sum += margin;
      if (margin > 1e-9) stats.search_better += 1;
      else if (margin < -1e-9) stats.policy_better += 1;
      else stats.ties += 1;
    }
  });

  stats.mean_margin = stats.compared ? stats.margin_sum / stats.compared : 0;
  delete stats.margin_sum;

  if (args.summary) {
    if (!fs.existsSync(args.summary)) {
      errors.push(`Summary not found: ${relativePath(args.summary)}`);
    } else {
      const summary = JSON.parse(fs.readFileSync(args.summary, 'utf8'));
      if (summary.examples !== examples.length) {
        errors.push(`Summary examples ${summary.examples} does not match dataset examples ${examples.length}`);
      }
    }
  }
  if (!stats.compared) errors.push('No policy comparison margins were available');
  if (stats.policy_better > 0) errors.push(`Policy-only beat search on ${stats.policy_better} compared example(s)`);
  if (stats.mean_margin < args.minMeanMargin) {
    errors.push(`Mean margin ${stats.mean_margin} is below ${args.minMeanMargin}`);
  }
  if (stats.changed_vs_policy < args.minChanged) {
    errors.push(`Changed decisions ${stats.changed_vs_policy} is below ${args.minChanged}`);
  }

  if (errors.length) {
    for (const error of errors) console.error(`FAIL ${error}`);
    process.exit(1);
  }
  console.log(
    `PASS ${relativePath(args.dataset)} examples=${stats.examples} ` +
    `changed_vs_policy=${stats.changed_vs_policy} compared=${stats.compared} ` +
    `search_better=${stats.search_better} ties=${stats.ties} ` +
    `mean_margin=${stats.mean_margin.toFixed(6)}`
  );
  return stats;
}

function main() {
  try {
    evaluate(parseArgs(process.argv.slice(2)));
  } catch (error) {
    console.error(`FAIL ${error.stack || error.message}`);
    process.exit(1);
  }
}

main();
