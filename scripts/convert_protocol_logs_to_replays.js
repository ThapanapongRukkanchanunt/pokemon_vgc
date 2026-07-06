const fs = require('node:fs');
const path = require('node:path');
const {replayHtmlFromProtocolText, replayLinesFromProtocolText} = require('../src/battle/replay_export');

const repoRoot = path.join(__dirname, '..');

function parseArgs(argv) {
  const args = {overwrite: false};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--overwrite') {
      args.overwrite = true;
      continue;
    }
    if (!arg.startsWith('--')) throw new Error(`Unexpected argument: ${arg}`);
    const key = arg.slice(2);
    const value = argv[i + 1];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for --${key}`);
    args[key] = value;
    i++;
  }
  return args;
}

function battleIdForProtocolFile(fileName) {
  if (fileName.endsWith('.protocol.txt')) return fileName.slice(0, -'.protocol.txt'.length);
  if (fileName.endsWith('.log')) return fileName.slice(0, -'.log'.length);
  return null;
}

function protocolFiles(logDir) {
  return fs.readdirSync(logDir, {withFileTypes: true})
    .filter(entry => entry.isFile())
    .map(entry => entry.name)
    .filter(name => name.endsWith('.protocol.txt') || name.endsWith('.log'))
    .sort();
}

function convertOne({logDir, fileName, overwrite}) {
  const battleId = battleIdForProtocolFile(fileName);
  const protocolPath = path.join(logDir, fileName);
  const replayPath = path.join(logDir, `${battleId}.replay.html`);

  if (!overwrite && fs.existsSync(replayPath)) {
    return {status: 'skipped', fileName, replayPath, reason: 'exists'};
  }

  const protocolText = fs.readFileSync(protocolPath, 'utf8');
  const replayLines = replayLinesFromProtocolText(protocolText);
  if (!replayLines.length) {
    return {status: 'failed', fileName, replayPath, reason: 'no spectator replay lines found'};
  }

  fs.writeFileSync(replayPath, replayHtmlFromProtocolText({
    title: battleId,
    protocolText,
  }), 'utf8');
  return {status: 'converted', fileName, replayPath, replayLines: replayLines.length};
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const logDir = path.resolve(repoRoot, args['log-dir'] || path.join('logs', 'battles'));
  const files = fs.existsSync(logDir) ? protocolFiles(logDir) : [];
  const results = files.map(fileName => convertOne({logDir, fileName, overwrite: args.overwrite}));

  const counts = results.reduce((acc, result) => {
    acc[result.status] = (acc[result.status] || 0) + 1;
    return acc;
  }, {});

  for (const result of results) {
    if (result.status === 'converted') {
      console.log(`CONVERTED ${result.fileName} -> ${path.basename(result.replayPath)} (${result.replayLines} lines)`);
    } else if (result.status === 'skipped') {
      console.log(`SKIPPED ${result.fileName} (${result.reason})`);
    } else {
      console.log(`FAILED ${result.fileName} (${result.reason})`);
    }
  }

  console.log(JSON.stringify({
    log_dir: path.relative(repoRoot, logDir).replace(/\\/g, '/'),
    total: results.length,
    converted: counts.converted || 0,
    skipped: counts.skipped || 0,
    failed: counts.failed || 0,
  }, null, 2));

  if (counts.failed) process.exit(1);
}

main();
