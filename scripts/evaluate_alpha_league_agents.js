const fs = require('node:fs');
const path = require('node:path');
const {closeTorchPolicyScorers, createAgent} = require('../src/agents');
const {
  findLeadMode,
  findTeam,
  loadTeamPool,
  makeRng,
  runBattle,
} = require('../src/battle/run_battle');

const repoRoot = path.join(__dirname, '..');

function parseInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) throw new Error(`${name} must be an integer`);
  return parsed;
}

function parseArgs(argv) {
  const args = {
    runId: 'mb_alpha_eval_iter_001',
    modelsDir: path.join(repoRoot, 'models', 'torch', 'mb_alpha_league', 'agents'),
    modelManifest: null,
    agentTeams: null,
    teamPreviewModel: null,
    previewMode: 'learned',
    opponentAgent: 'random',
    opponentModelsDir: null,
    opponentModelManifest: null,
    opponentTeams: null,
    opponentTeamPreviewModel: null,
    opponentPreviewMode: 'learned',
    outDir: path.join(repoRoot, 'experiments', 'mb_alpha_league_eval'),
    logDir: null,
    gamesPerPairing: 1,
    seed: null,
    pythonPath: process.env.POKEMON_RL_PYTHON || null,
    torchDevice: process.env.POKEMON_RL_TORCH_DEVICE || null,
    sideSwaps: true,
    topK: 1,
    megaPolicy: 'model',
    rolloutMaxDecisions: 120,
    compactLogs: false,
    deleteBattleLogs: false,
    overwrite: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--run-id') {
      args.runId = argv[++i];
    } else if (arg === '--models-dir') {
      args.modelsDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--model-manifest') {
      args.modelManifest = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--agent-teams') {
      args.agentTeams = argv[++i].split(',').map(value => value.trim()).filter(Boolean);
    } else if (arg === '--team-preview-model') {
      args.teamPreviewModel = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--preview-mode') {
      args.previewMode = argv[++i];
    } else if (arg === '--opponent-agent') {
      args.opponentAgent = argv[++i].toLowerCase();
    } else if (arg === '--opponent-models-dir') {
      args.opponentModelsDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--opponent-model-manifest') {
      args.opponentModelManifest = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--opponent-teams') {
      args.opponentTeams = argv[++i].split(',').map(value => value.trim()).filter(Boolean);
    } else if (arg === '--opponent-team-preview-model') {
      args.opponentTeamPreviewModel = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--opponent-preview-mode') {
      args.opponentPreviewMode = argv[++i];
    } else if (arg === '--out-dir') {
      args.outDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--log-dir') {
      args.logDir = path.resolve(repoRoot, argv[++i]);
    } else if (arg === '--games-per-pairing') {
      args.gamesPerPairing = parseInteger(argv[++i], '--games-per-pairing');
    } else if (arg === '--seed') {
      args.seed = argv[++i];
    } else if (arg === '--python') {
      args.pythonPath = argv[++i];
    } else if (arg === '--torch-device') {
      args.torchDevice = argv[++i];
    } else if (arg === '--side-swaps') {
      args.sideSwaps = true;
    } else if (arg === '--no-side-swaps') {
      args.sideSwaps = false;
    } else if (arg === '--top-k') {
      args.topK = parseInteger(argv[++i], '--top-k');
    } else if (arg === '--mega-policy') {
      args.megaPolicy = argv[++i].toLowerCase().replace(/-/g, '_');
    } else if (arg === '--rollout-max-decisions') {
      args.rolloutMaxDecisions = parseInteger(argv[++i], '--rollout-max-decisions');
    } else if (arg === '--compact-logs') {
      args.compactLogs = true;
    } else if (arg === '--delete-battle-logs') {
      args.deleteBattleLogs = true;
    } else if (arg === '--overwrite') {
      args.overwrite = true;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }

  if (!args.runId || /[\\/:*?"<>|]/.test(args.runId)) throw new Error('--run-id must be filename-safe');
  if (args.gamesPerPairing <= 0) throw new Error('--games-per-pairing must be > 0');
  if (!['learned', 'random', 'battle-model'].includes(args.previewMode)) {
    throw new Error('--preview-mode must be learned, random, or battle-model');
  }
  if (!['random', 'maxdamage', 'heuristic', 'rl'].includes(args.opponentAgent)) {
    throw new Error('--opponent-agent must be random, maxdamage, heuristic, or rl');
  }
  if (!['learned', 'random', 'battle-model'].includes(args.opponentPreviewMode)) {
    throw new Error('--opponent-preview-mode must be learned, random, or battle-model');
  }
  if (args.topK <= 0) throw new Error('--top-k must be > 0');
  if (!['model', 'sole_usable'].includes(args.megaPolicy)) {
    throw new Error('--mega-policy must be model or sole_usable');
  }
  if (args.rolloutMaxDecisions <= 0) throw new Error('--rollout-max-decisions must be > 0');
  if (!args.seed) args.seed = args.runId;
  if (!args.logDir) args.logDir = path.join(repoRoot, 'logs', 'battles', `${args.runId}_eval`);
  return args;
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function assertInsideRepo(filePath, label) {
  const resolved = path.resolve(filePath);
  const relative = path.relative(repoRoot, resolved);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`${label} must stay inside repo: ${resolved}`);
  }
}

function removeIfExists(filePath, label) {
  if (!fs.existsSync(filePath)) return;
  assertInsideRepo(filePath, label);
  fs.rmSync(filePath, {recursive: true, force: true});
}

function winnerSide(winner) {
  if (typeof winner !== 'string') return 'unknown';
  const match = winner.match(/\b(P[12])$/i);
  return match ? match[1].toLowerCase() : 'unknown';
}

function loadModelManifest(filePath, label) {
  if (!filePath) return null;
  if (!fs.existsSync(filePath)) throw new Error(`Missing ${label}: ${filePath}`);
  const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  const entries = parsed.models || parsed;
  if (!entries || typeof entries !== 'object' || Array.isArray(entries)) {
    throw new Error(`${label} must be a JSON object mapping team IDs to checkpoints`);
  }
  return entries;
}

function checkpointFromEntry(entry) {
  if (typeof entry === 'string') return entry;
  if (entry && typeof entry.checkpoint === 'string') return entry.checkpoint;
  return null;
}

function modelPathForTeam({modelsDir, manifest}, teamId) {
  if (!manifest) return path.join(modelsDir, teamId, 'checkpoint.pt');
  const checkpoint = checkpointFromEntry(manifest[teamId]);
  if (!checkpoint) throw new Error(`Model manifest has no checkpoint for ${teamId}`);
  return path.isAbsolute(checkpoint) ? checkpoint : path.resolve(repoRoot, checkpoint);
}

function selectTeams(pool, selectors, label) {
  if (!selectors) return pool.teams;
  const unique = [];
  const seen = new Set();
  for (const selector of selectors) {
    const team = findTeam(pool, selector, makeRng(`${label}:${selector}`));
    if (!seen.has(team.id)) {
      unique.push(team);
      seen.add(team.id);
    }
  }
  if (!unique.length) throw new Error(`${label} selected no teams`);
  return unique;
}

function ensureOutput(args) {
  fs.mkdirSync(args.outDir, {recursive: true});
  const summaryPath = path.join(args.outDir, `${args.runId}_summary.json`);
  if (args.overwrite) {
    removeIfExists(summaryPath, 'summary');
    removeIfExists(args.logDir, 'log directory');
  } else if (fs.existsSync(summaryPath)) {
    throw new Error(`Summary exists at ${relativePath(summaryPath)}; pass --overwrite`);
  }
  fs.mkdirSync(args.logDir, {recursive: true});
  return {summaryPath};
}

function compactResultArtifacts(result, deleteTraces) {
  const artifacts = [result.protocol_log_path, result.replay_html_path];
  if (deleteTraces) artifacts.push(result.trace_jsonl_path, result.summary_json_path);
  for (const relative of artifacts) {
    if (!relative) continue;
    const filePath = path.resolve(repoRoot, relative);
    assertInsideRepo(filePath, 'battle artifact');
    if (fs.existsSync(filePath)) fs.rmSync(filePath, {force: true});
  }
}

function megaPolicyStats(result, rlSide) {
  const stats = {decisions: 0, eligible: 0, applied: 0, chosen_mega: 0};
  const tracePath = path.resolve(repoRoot, result.trace_jsonl_path);
  if (!fs.existsSync(tracePath)) return stats;
  for (const line of fs.readFileSync(tracePath, 'utf8').split(/\r?\n/)) {
    if (!line) continue;
    const row = JSON.parse(line);
    if (row.side !== rlSide) continue;
    const policy = row.agent_diagnostics?.ppo_policy;
    if (!policy) continue;
    stats.decisions += 1;
    if (policy.mega_policy_eligible) stats.eligible += 1;
    if (policy.mega_policy_applied) stats.applied += 1;
    if (/\bmega\b/.test(String(row.chosen_action || ''))) stats.chosen_mega += 1;
  }
  return stats;
}

function addMegaPolicyStats(target, stats) {
  for (const key of ['decisions', 'eligible', 'applied', 'chosen_mega']) {
    target[key] += stats[key];
  }
}

function createRlAgent({args, pool, team, role = 'agent'}) {
  const opponent = role === 'opponent';
  const previewMode = opponent ? args.opponentPreviewMode : args.previewMode;
  const previewModel = opponent ? args.opponentTeamPreviewModel : args.teamPreviewModel;
  const modelsDir = opponent ? args.opponentModelsDir : args.modelsDir;
  const manifest = opponent ? args.opponentModelManifestData : args.modelManifestData;
  const useLearnedPreview = previewMode === 'learned';
  return createAgent('final_rl', {
    formatId: pool.format_id,
    modelPath: modelPathForTeam({modelsDir, manifest}, team.id),
    teamPreviewModelPath: useLearnedPreview ? previewModel : null,
    teamPreviewMode: previewMode === 'random' ? 'random' : 'model',
    pythonPath: args.pythonPath,
    torchDevice: args.torchDevice,
    epsilon: 0,
    topK: args.topK,
    megaPolicy: opponent ? 'model' : args.megaPolicy,
    sampleActions: false,
  });
}

function createOpponentAgent({args, pool, team}) {
  if (args.opponentAgent === 'rl') return createRlAgent({args, pool, team, role: 'opponent'});
  return createAgent(args.opponentAgent, {formatId: pool.format_id});
}

function ensureRow(table, teamId) {
  if (!table.has(teamId)) {
    table.set(teamId, {
      team_id: teamId,
      games: 0,
      wins: 0,
      losses: 0,
      unknown: 0,
      win_rate: 0,
      p1: {games: 0, wins: 0, losses: 0, unknown: 0, win_rate: 0},
      p2: {games: 0, wins: 0, losses: 0, unknown: 0, win_rate: 0},
    });
  }
  return table.get(teamId);
}

function record(table, teamId, result, side) {
  const row = ensureRow(table, teamId);
  row.games += 1;
  row[side].games += 1;
  if (result === 'win') row.wins += 1;
  else if (result === 'loss') row.losses += 1;
  else row.unknown += 1;
  if (result === 'win') row[side].wins += 1;
  else if (result === 'loss') row[side].losses += 1;
  else row[side].unknown += 1;
}

function finalize(table) {
  return [...table.values()]
    .map(row => ({
      ...row,
      win_rate: row.wins + row.losses ? row.wins / (row.wins + row.losses) : 0,
      p1: {
        ...row.p1,
        win_rate: row.p1.wins + row.p1.losses ? row.p1.wins / (row.p1.wins + row.p1.losses) : 0,
      },
      p2: {
        ...row.p2,
        win_rate: row.p2.wins + row.p2.losses ? row.p2.wins / (row.p2.wins + row.p2.losses) : 0,
      },
    }))
    .sort((a, b) => b.win_rate - a.win_rate || b.wins - a.wins || a.team_id.localeCompare(b.team_id));
}

function sideSummary(standings) {
  return Object.fromEntries(['p1', 'p2'].map(side => {
    const total = standings.reduce((accumulator, row) => {
      for (const key of ['games', 'wins', 'losses', 'unknown']) accumulator[key] += row[side][key];
      return accumulator;
    }, {games: 0, wins: 0, losses: 0, unknown: 0});
    total.win_rate = total.wins + total.losses ? total.wins / (total.wins + total.losses) : 0;
    return [side, total];
  }));
}

async function evaluate(args) {
  const pool = loadTeamPool();
  args.modelManifestData = loadModelManifest(args.modelManifest, 'model manifest');
  args.opponentModelManifestData = loadModelManifest(args.opponentModelManifest, 'opponent model manifest');
  const agentTeams = selectTeams(pool, args.agentTeams, '--agent-teams');
  const opponentTeams = selectTeams(pool, args.opponentTeams, '--opponent-teams');
  for (const team of agentTeams) {
    const modelPath = modelPathForTeam({modelsDir: args.modelsDir, manifest: args.modelManifestData}, team.id);
    if (!fs.existsSync(modelPath)) throw new Error(`Missing model for ${team.id}: ${modelPath}`);
  }
  if (args.previewMode === 'learned' && !args.teamPreviewModel) {
    throw new Error('--preview-mode learned requires --team-preview-model');
  }
  if (args.previewMode === 'learned' && !fs.existsSync(args.teamPreviewModel)) {
    throw new Error(`Missing team preview model: ${args.teamPreviewModel}`);
  }
  if (args.opponentAgent === 'rl') {
    if (!args.opponentModelsDir && !args.opponentModelManifest) {
      throw new Error('--opponent-agent rl requires --opponent-models-dir or --opponent-model-manifest');
    }
    for (const team of opponentTeams) {
      const modelPath = modelPathForTeam({
        modelsDir: args.opponentModelsDir,
        manifest: args.opponentModelManifestData,
      }, team.id);
      if (!fs.existsSync(modelPath)) throw new Error(`Missing opponent model for ${team.id}: ${modelPath}`);
    }
    if (args.opponentPreviewMode === 'learned' && !args.opponentTeamPreviewModel) {
      throw new Error('--opponent-preview-mode learned requires --opponent-team-preview-model');
    }
    if (args.opponentPreviewMode === 'learned' && !fs.existsSync(args.opponentTeamPreviewModel)) {
      throw new Error(`Missing opponent team preview model: ${args.opponentTeamPreviewModel}`);
    }
  }
  const outputs = ensureOutput(args);
  const table = new Map();
  const matchups = [];
  const megaPolicy = {decisions: 0, eligible: 0, applied: 0, chosen_mega: 0};
  let gameIndex = 0;

  for (const agentTeam of agentTeams) {
    for (const opponentTeam of opponentTeams) {
      const matchup = {
        agent_team: agentTeam.id,
        opponent_team: opponentTeam.id,
        games: [],
        wins: 0,
        losses: 0,
        unknown: 0,
      };
      for (let game = 1; game <= args.gamesPerPairing; game++) {
        for (const swapped of [false, true]) {
          if (swapped && !args.sideSwaps) continue;
          gameIndex += 1;
          const seed = `${args.seed}:${agentTeam.id}:vs:${opponentTeam.id}:${game}:${swapped ? 'p2' : 'p1'}`;
          const rng = makeRng(seed);
          const rlIsP1 = !swapped;
          const result = await runBattle({
            pool,
            seed,
            p1Team: rlIsP1 ? agentTeam : opponentTeam,
            p2Team: rlIsP1 ? opponentTeam : agentTeam,
            p1Lead: findLeadMode(rlIsP1 ? agentTeam : opponentTeam, null, rng),
            p2Lead: findLeadMode(rlIsP1 ? opponentTeam : agentTeam, null, rng),
            p1Agent: rlIsP1 ?
              createRlAgent({args, pool, team: agentTeam}) :
              createOpponentAgent({args, pool, team: opponentTeam}),
            p2Agent: rlIsP1 ?
              createOpponentAgent({args, pool, team: opponentTeam}) :
              createRlAgent({args, pool, team: agentTeam}),
            logDir: args.logDir,
            rng,
            rolloutMaxDecisions: args.rolloutMaxDecisions,
          });
          const side = winnerSide(result.winner);
          const rlWon = (rlIsP1 && side === 'p1') || (!rlIsP1 && side === 'p2');
          const outcome = side === 'unknown' ? 'unknown' : (rlWon ? 'win' : 'loss');
          addMegaPolicyStats(megaPolicy, megaPolicyStats(result, rlIsP1 ? 'p1' : 'p2'));
          record(table, agentTeam.id, outcome, rlIsP1 ? 'p1' : 'p2');
          if (outcome === 'win') matchup.wins += 1;
          else if (outcome === 'loss') matchup.losses += 1;
          else matchup.unknown += 1;
          matchup.games.push({
            game,
            rl_side: rlIsP1 ? 'p1' : 'p2',
            seed,
            winner: result.winner,
            winner_side: side,
            turns: result.turns,
            trace_jsonl_path: args.deleteBattleLogs ? null : result.trace_jsonl_path,
          });
          if (args.compactLogs || args.deleteBattleLogs) compactResultArtifacts(result, args.deleteBattleLogs);
          console.log(
            `eval ${gameIndex}: ${agentTeam.id} vs ${args.opponentAgent}(${opponentTeam.id}) ` +
            `outcome=${outcome} turns=${result.turns}`
          );
        }
      }
      matchups.push(matchup);
    }
  }

  const standings = finalize(table);
  const summary = {
    created_at: new Date().toISOString(),
    run_id: args.runId,
    models_dir: relativePath(args.modelsDir),
    model_manifest: args.modelManifest ? relativePath(args.modelManifest) : null,
    agent_teams: agentTeams.map(team => team.id),
    team_preview_model: args.previewMode === 'learned' ? relativePath(args.teamPreviewModel) : null,
    preview_mode: args.previewMode,
    opponent_agent: args.opponentAgent,
    opponent_models_dir: args.opponentModelsDir ? relativePath(args.opponentModelsDir) : null,
    opponent_model_manifest: args.opponentModelManifest ? relativePath(args.opponentModelManifest) : null,
    opponent_teams: opponentTeams.map(team => team.id),
    opponent_team_preview_model: args.opponentAgent === 'rl' && args.opponentPreviewMode === 'learned' ?
      relativePath(args.opponentTeamPreviewModel) :
      null,
    opponent_preview_mode: args.opponentAgent === 'rl' ? args.opponentPreviewMode : null,
    games_per_pairing: args.gamesPerPairing,
    side_swaps: args.sideSwaps,
    top_k: args.topK,
    mega_policy: args.megaPolicy,
    mega_policy_stats: {
      ...megaPolicy,
      applied_rate_when_eligible: megaPolicy.eligible ?
        megaPolicy.applied / megaPolicy.eligible :
        0,
      chosen_mega_rate_when_eligible: megaPolicy.eligible ?
        megaPolicy.chosen_mega / megaPolicy.eligible :
        0,
    },
    rollout_max_decisions: args.rolloutMaxDecisions,
    seed: args.seed,
    log_dir: args.deleteBattleLogs ? null : relativePath(args.logDir),
    compact_logs: args.compactLogs,
    delete_battle_logs: args.deleteBattleLogs,
    side_summary: sideSummary(standings),
    standings,
    matchups,
  };
  fs.writeFileSync(outputs.summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  console.log(`Wrote summary: ${relativePath(outputs.summaryPath)}`);
}

evaluate(parseArgs(process.argv.slice(2)))
  .finally(() => closeTorchPolicyScorers())
  .catch(error => {
    console.error(`FAIL ${error.stack || error.message}`);
    process.exit(1);
  });
