const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');
const {once} = require('node:events');
const {canonicalTeamPreviewChoice} = require('../src/team_preview/preview_actions');
const {teamContextFromTeams} = require('../src/team_preview/team_context');

const repoRoot = path.join(__dirname, '..');

function parseArgs(argv) {
  const args = {
    traceDir: null,
    rollouts: null,
    out: null,
    teamPool: path.join(repoRoot, 'data', 'teams', 'team_pool.json'),
    append: false,
    overwrite: false,
  };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--trace-dir') args.traceDir = path.resolve(repoRoot, argv[++index]);
    else if (arg === '--rollouts') args.rollouts = path.resolve(repoRoot, argv[++index]);
    else if (arg === '--out') args.out = path.resolve(repoRoot, argv[++index]);
    else if (arg === '--team-pool') args.teamPool = path.resolve(repoRoot, argv[++index]);
    else if (arg === '--append') args.append = true;
    else if (arg === '--overwrite') args.overwrite = true;
    else throw new Error(`Unexpected argument: ${arg}`);
  }
  if (Boolean(args.traceDir) === Boolean(args.rollouts)) {
    throw new Error('Provide exactly one of --trace-dir or --rollouts');
  }
  if (!args.out) throw new Error('--out is required');
  if (args.append && args.overwrite) throw new Error('--append and --overwrite are mutually exclusive');
  return args;
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function listTraceFiles(dir) {
  const files = [];
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...listTraceFiles(fullPath));
    else if (entry.isFile() && entry.name.endsWith('.trace.jsonl')) files.push(fullPath);
  }
  return files.sort();
}

function winnerSide(winner) {
  if (typeof winner !== 'string') return null;
  const match = winner.match(/\b(P[12])$/i);
  return match ? match[1].toLowerCase() : null;
}

function validPreviewChoice(choice) {
  const match = String(choice || '').match(/^team\s+(\d{4})$/i);
  return Boolean(match && new Set(match[1]).size === 4);
}

function makeExample({row, ownTeam, opponentTeam, sourcePath, winner}) {
  const sideWon = row.winner_side || winnerSide(winner);
  if (!['p1', 'p2'].includes(row.side) || !['p1', 'p2'].includes(sideWon)) return null;
  const action = canonicalTeamPreviewChoice(row.action || row.chosen_action);
  if (!validPreviewChoice(action)) return null;
  const state = row.state || {
    request: row.request,
    public_state: row.public_state,
  };
  if (!state.request?.teamPreview) return null;
  const foeSide = row.side === 'p1' ? 'p2' : 'p1';
  if (!state.team_context && ownTeam && opponentTeam) {
    state.team_context = teamContextFromTeams({
      ownTeam,
      opponentTeam,
      publicState: state.public_state,
      foeSide,
    });
  }
  if (!state.team_context?.own_team || !state.team_context?.opponent_team) return null;
  return {
    example_id: `${row.battle_id}:${row.side}:team_preview`,
    source_path: relativePath(sourcePath),
    battle_id: row.battle_id,
    run_id: row.run_id || null,
    request_type: 'team_preview',
    side: row.side,
    team: row.team || row.agent_team_id || ownTeam?.id || 'unknown',
    opponent_team: opponentTeam?.id || state.team_context.opponent_team.id || 'unknown',
    lead: row.lead || 'unknown',
    turn: 0,
    state,
    action,
    target: sideWon === row.side ? 1 : 0,
    winner_side: sideWon,
  };
}

async function readJsonl(filePath, onRow) {
  const input = fs.createReadStream(filePath, {encoding: 'utf8'});
  const lines = readline.createInterface({input, crlfDelay: Infinity});
  for await (const line of lines) {
    if (!line.trim()) continue;
    await onRow(JSON.parse(line));
  }
}

async function loadExistingIds(filePath) {
  const ids = new Set();
  if (!fs.existsSync(filePath)) return ids;
  await readJsonl(filePath, row => {
    if (row.example_id) ids.add(row.example_id);
  });
  return ids;
}

async function writeLine(stream, row) {
  if (!stream.write(`${JSON.stringify(row)}\n`)) await once(stream, 'drain');
}

async function build(args) {
  if (!fs.existsSync(args.teamPool)) throw new Error(`Missing team pool: ${args.teamPool}`);
  const pool = JSON.parse(fs.readFileSync(args.teamPool, 'utf8'));
  const teams = new Map((pool.teams || []).map(team => [team.id, team]));
  const sourcePath = args.traceDir || args.rollouts;
  if (!fs.existsSync(sourcePath)) throw new Error(`Missing input: ${sourcePath}`);
  fs.mkdirSync(path.dirname(args.out), {recursive: true});
  if (fs.existsSync(args.out) && !args.append && !args.overwrite) {
    throw new Error(`Output exists: ${args.out}; pass --append or --overwrite`);
  }

  const existingIds = args.append ? await loadExistingIds(args.out) : new Set();
  const output = fs.createWriteStream(args.out, {flags: args.append ? 'a' : 'w', encoding: 'utf8'});
  const summary = {
    created_at: new Date().toISOString(),
    source_kind: args.traceDir ? 'battle_traces' : 'ppo_rollouts',
    source_path: relativePath(sourcePath),
    output_path: relativePath(args.out),
    append: args.append,
    existing_examples: existingIds.size,
    examples_added: 0,
    duplicates_skipped: 0,
    invalid_skipped: 0,
    targets: {0: 0, 1: 0},
    teams: {},
  };

  async function add(example) {
    if (!example) {
      summary.invalid_skipped += 1;
      return;
    }
    if (existingIds.has(example.example_id)) {
      summary.duplicates_skipped += 1;
      return;
    }
    existingIds.add(example.example_id);
    await writeLine(output, example);
    summary.examples_added += 1;
    summary.targets[example.target] += 1;
    summary.teams[example.team] = (summary.teams[example.team] || 0) + 1;
  }

  if (args.rollouts) {
    await readJsonl(args.rollouts, async row => {
      if (row.request_type !== 'team_preview') return;
      const ownId = row.team || row.agent_team_id || (row.side === 'p1' ? row.p1_team : row.p2_team);
      const opponentId = row.opponent_id || (row.side === 'p1' ? row.p2_team : row.p1_team);
      await add(makeExample({
        row,
        ownTeam: teams.get(ownId),
        opponentTeam: teams.get(opponentId),
        sourcePath: args.rollouts,
        winner: row.winner,
      }));
    });
  } else {
    const traceFiles = listTraceFiles(args.traceDir);
    summary.trace_files = traceFiles.length;
    for (const tracePath of traceFiles) {
      const rows = [];
      await readJsonl(tracePath, row => rows.push(row));
      const teamBySide = new Map();
      for (const row of rows) {
        if (['p1', 'p2'].includes(row.side) && row.team && !teamBySide.has(row.side)) {
          teamBySide.set(row.side, row.team);
        }
      }
      for (const row of rows) {
        if (!row.request?.teamPreview || row.error_recovery) continue;
        const foeSide = row.side === 'p1' ? 'p2' : 'p1';
        await add(makeExample({
          row,
          ownTeam: teams.get(teamBySide.get(row.side)),
          opponentTeam: teams.get(teamBySide.get(foeSide)),
          sourcePath: tracePath,
          winner: row.outcome_context?.winner,
        }));
      }
    }
  }

  output.end();
  await once(output, 'finish');
  summary.total_examples = existingIds.size;
  if (!summary.total_examples) {
    fs.rmSync(args.out, {force: true});
    throw new Error('No valid team-preview examples were produced');
  }
  const summaryPath = `${args.out}.summary.json`;
  fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  console.log(`Preview examples added: ${summary.examples_added}`);
  console.log(`Preview replay examples: ${summary.total_examples}`);
  console.log(`Dataset: ${relativePath(args.out)}`);
  console.log(`Summary: ${relativePath(summaryPath)}`);
}

build(parseArgs(process.argv.slice(2))).catch(error => {
  console.error(`FAIL ${error.stack || error.message}`);
  process.exit(1);
});
