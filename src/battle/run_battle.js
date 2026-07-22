const fs = require('node:fs');
const path = require('node:path');
const {
  createBattleStream,
  dexForFormat,
  isTurnUpdate,
  requestFromChunk,
  spectatorLinesFromChunk,
  validateAndPackTeam,
  winnerFromChunk,
} = require('./showdown_protocol');
const {replayHtml} = require('./replay_export');
const {createRolloutSearch} = require('./rollout_search');
const {enumerateLegalActionChoices} = require('../agents/legal_actions');

const repoRoot = path.join(__dirname, '..', '..');

function hashSeed(value) {
  const text = String(value);
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function makeRng(seedValue) {
  let state = hashSeed(seedValue) || 0x9e3779b9;
  return {
    nextUint32() {
      state += 0x6d2b79f5;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0);
    },
    next() {
      return this.nextUint32() / 4294967296;
    },
    pick(items) {
      if (!items.length) throw new Error('Cannot pick from an empty list');
      return items[Math.floor(this.next() * items.length)];
    },
  };
}

function showdownSeed(rng) {
  return [rng.nextUint32(), rng.nextUint32(), rng.nextUint32(), rng.nextUint32()];
}

function findTeam(pool, selector, rng) {
  if (!selector) return rng.pick(pool.teams);
  const byId = pool.teams.find(team => team.id === selector);
  if (byId) return byId;
  const byName = pool.teams.find(team => team.name.toLowerCase() === selector.toLowerCase());
  if (byName) return byName;
  const index = Number(selector);
  if (Number.isInteger(index) && index >= 1 && index <= pool.teams.length) return pool.teams[index - 1];
  throw new Error(`Unknown team selector: ${selector}`);
}

function findLeadMode(team, selector, rng) {
  if (!selector) return rng.pick(team.lead_modes);
  const byId = team.lead_modes.find(mode => mode.id === selector);
  if (byId) return byId;
  const index = Number(selector);
  if (Number.isInteger(index) && index >= 1 && index <= team.lead_modes.length) {
    return team.lead_modes[index - 1];
  }
  throw new Error(`Unknown lead selector for ${team.id}: ${selector}`);
}

function loadTeamPool(poolPath = path.join(repoRoot, 'data', 'teams', 'team_pool.json')) {
  return JSON.parse(fs.readFileSync(poolPath, 'utf8'));
}

function loadPackedTeam(pool, team) {
  const importPath = path.join(repoRoot, team.import_file);
  const importText = fs.readFileSync(importPath, 'utf8');
  try {
    return validateAndPackTeam({formatId: pool.format_id, importText});
  } catch (error) {
    throw new Error(`${team.id} failed validation:\n${error.message}`);
  }
}

function safeLogName(value) {
  return String(value).replace(/[^a-zA-Z0-9_.-]+/g, '_');
}

function sideSummary(team, leadMode) {
  return {
    team_id: team.id,
    team_name: team.name,
    lead_id: leadMode.id,
    team_spec: leadMode.team_spec,
    leads: leadMode.leads,
    back: leadMode.back,
  };
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function sideFromIdent(ident) {
  const match = /^(p[12])([ab]):/.exec(ident || '');
  if (!match) return null;
  return {side: match[1], slot: match[2] === 'a' ? 0 : 1};
}

function speciesFromDetails(details) {
  return (details || '').split(',')[0].trim();
}

function createPublicState() {
  return {
    active: {
      p1: [null, null],
      p2: [null, null],
    },
  };
}

function applySpectatorLineToState(publicState, line) {
  const parts = line.split('|');
  const type = parts[1];
  if (!type) return;

  if (['switch', 'drag', 'replace'].includes(type)) {
    const parsed = sideFromIdent(parts[2]);
    if (!parsed) return;
    publicState.active[parsed.side][parsed.slot] = {
      ident: parts[2],
      species: speciesFromDetails(parts[3]),
      condition: parts[4] || null,
      fainted: false,
    };
    return;
  }

  if (['detailschange', '-formechange'].includes(type)) {
    const parsed = sideFromIdent(parts[2]);
    if (!parsed || !publicState.active[parsed.side][parsed.slot]) return;
    publicState.active[parsed.side][parsed.slot].species = speciesFromDetails(parts[3]);
    publicState.active[parsed.side][parsed.slot].fainted = false;
    return;
  }

  if (type === 'faint') {
    const parsed = sideFromIdent(parts[2]);
    if (!parsed || !publicState.active[parsed.side][parsed.slot]) return;
    publicState.active[parsed.side][parsed.slot].fainted = true;
    publicState.active[parsed.side][parsed.slot].condition = '0 fnt';
    return;
  }

  if (['-damage', '-heal', '-sethp'].includes(type)) {
    const parsed = sideFromIdent(parts[2]);
    if (!parsed || !publicState.active[parsed.side][parsed.slot]) return;
    const condition = parts[3] || null;
    publicState.active[parsed.side][parsed.slot].condition = condition;
    if (condition && /\bfnt\b/.test(condition)) {
      publicState.active[parsed.side][parsed.slot].fainted = true;
    }
  }
}

function errorSideFromChunk(chunk) {
  if (!chunk.startsWith('sideupdate\n') || !chunk.includes('|error|')) return null;
  return chunk.split('\n')[1] || null;
}

function agentDiagnostics(agent, context) {
  if (typeof agent?.getDiagnostics !== 'function') return null;
  const diagnostics = agent.getDiagnostics(context);
  if (diagnostics == null) return null;
  return deepClone(diagnostics);
}

async function runBattle({
  pool,
  seed,
  p1Team,
  p2Team,
  p1Lead,
  p2Lead,
  p1Agent,
  p2Agent,
  logDir,
  rng,
  rolloutMaxDecisions = 120,
}) {
  const battleRng = rng || makeRng(seed);
  const agentRngBySide = {
    p1: makeRng(`${seed}:p1-agent`),
    p2: makeRng(`${seed}:p2-agent`),
  };
  const battleSeed = showdownSeed(battleRng);
  const p1PackedTeam = loadPackedTeam(pool, p1Team);
  const p2PackedTeam = loadPackedTeam(pool, p2Team);
  const stream = createBattleStream();
  const protocolLog = [];
  const replayLines = [];
  const traceRows = [];
  const publicState = createPublicState();
  const dex = dexForFormat(pool.format_id);
  let turns = 0;
  let winner = null;
  const p1Name = `${p1Agent.displayName || p1Agent.name || 'Agent'} P1`;
  const p2Name = `${p2Agent.displayName || p2Agent.name || 'Agent'} P2`;
  const battleId = `${new Date().toISOString().replace(/[:.]/g, '-')}_${safeLogName(seed)}_${p1Team.id}_vs_${p2Team.id}`;
  const lastTraceIndexBySide = {p1: -1, p2: -1};
  const lastDecisionContextBySide = {p1: null, p2: null};
  const errorRecoveryCount = {p1: 0, p2: 0};
  const pendingRequestsBySide = {p1: null, p2: null};

  async function write(line) {
    protocolLog.push(line);
    await stream.write(line);
  }

  await write(`>start ${JSON.stringify({formatid: pool.format_id, seed: battleSeed})}`);
  await write(`>player p1 ${JSON.stringify({name: p1Name, team: p1PackedTeam})}`);
  await write(`>player p2 ${JSON.stringify({name: p2Name, team: p2PackedTeam})}`);

  while (true) {
    const chunk = await stream.read();
    if (chunk === null) break;
    protocolLog.push(chunk);

    const errorSide = errorSideFromChunk(chunk);
    if (errorSide) {
      errorRecoveryCount[errorSide] += 1;
      if (errorRecoveryCount[errorSide] > 5) {
        throw new Error(`Too many invalid choices for ${errorSide}; last error chunk:\n${chunk}`);
      }
      const badTraceIndex = lastTraceIndexBySide[errorSide];
      if (badTraceIndex >= 0) {
        traceRows.splice(badTraceIndex, 1);
        for (const side of ['p1', 'p2']) {
          if (lastTraceIndexBySide[side] > badTraceIndex) lastTraceIndexBySide[side] -= 1;
        }
      }
      const context = lastDecisionContextBySide[errorSide];
      if (context) {
        traceRows.push({
          battle_id: battleId,
          seed,
          format: pool.format_id,
          turn: turns,
          side: errorSide,
          agent: context.agent.name,
          team: context.team.id,
          lead: context.leadMode.id,
          request: deepClone(context.request),
          public_state: deepClone(context.publicState),
          legal_actions: ['default'],
          chosen_action: 'default',
          outcome_context: {winner: null},
          error_recovery: true,
        });
        lastTraceIndexBySide[errorSide] = traceRows.length - 1;
      }
      await write(`>${errorSide} default`);
      continue;
    }

    const spectatorLines = spectatorLinesFromChunk(chunk);
    for (const line of spectatorLines) {
      replayLines.push(line);
      applySpectatorLineToState(publicState, line);
    }

    if (isTurnUpdate(chunk)) turns += 1;
    winner = winnerFromChunk(chunk) || winner;

    for (const side of ['p1', 'p2']) {
      const request = requestFromChunk(chunk, side);
      if (request) pendingRequestsBySide[side] = request;
    }

    // Showdown emits the two private requests as separate stream chunks. Wait
    // for both before asking either agent to choose so rollout search receives
    // the same pre-action battle snapshot on both sides.
    if (!pendingRequestsBySide.p1 || !pendingRequestsBySide.p2) {
      if (chunk.startsWith('end\n')) break;
      continue;
    }

    const requestsBySide = {...pendingRequestsBySide};
    pendingRequestsBySide.p1 = null;
    pendingRequestsBySide.p2 = null;
    const teamsBySide = {p1: p1Team, p2: p2Team};
    const leadsBySide = {p1: p1Lead, p2: p2Lead};
    const battleSnapshot = stream.battle ? stream.battle.toJSON() : null;
    const decisions = [];

    for (const sideConfig of [
      {side: 'p1', agent: p1Agent, team: p1Team, leadMode: p1Lead},
      {side: 'p2', agent: p2Agent, team: p2Team, leadMode: p2Lead},
    ]) {
      const request = requestsBySide[sideConfig.side];
      if (!request) continue;

      const rolloutSearch = battleSnapshot ? createRolloutSearch({
        battleSnapshot,
        teams: teamsBySide,
        leadModes: leadsBySide,
        maxDecisions: rolloutMaxDecisions,
      }) : null;
      const battleState = {
        turns,
        team: sideConfig.team,
        leadMode: sideConfig.leadMode,
        teams: teamsBySide,
        leadModes: leadsBySide,
        requests: requestsBySide,
        publicState,
        rolloutSearch,
      };
      const legalActions = enumerateLegalActionChoices({
        side: sideConfig.side,
        request,
        battleState,
        dex,
      });
      const choice = await Promise.resolve(sideConfig.agent.chooseAction({
        side: sideConfig.side,
        request,
        battleState,
        rng: agentRngBySide[sideConfig.side],
      }));
      if (choice) {
        const diagnostics = agentDiagnostics(sideConfig.agent, {
          side: sideConfig.side,
          request,
          battleState,
          choice,
        });
        const traceRow = {
          battle_id: battleId,
          seed,
          format: pool.format_id,
          turn: turns,
          side: sideConfig.side,
          agent: sideConfig.agent.name,
          team: sideConfig.team.id,
          lead: sideConfig.leadMode.id,
          request: deepClone(request),
          public_state: deepClone(publicState),
          legal_actions: legalActions,
          chosen_action: choice,
          outcome_context: {winner: null},
        };
        if (diagnostics) traceRow.agent_diagnostics = diagnostics;
        traceRows.push(traceRow);
        lastTraceIndexBySide[sideConfig.side] = traceRows.length - 1;
        lastDecisionContextBySide[sideConfig.side] = {
          agent: sideConfig.agent,
          team: sideConfig.team,
          leadMode: sideConfig.leadMode,
          request,
          publicState: deepClone(publicState),
        };
        decisions.push({side: sideConfig.side, choice});
      }
    }

    for (const decision of decisions) {
      await write(`>${decision.side} ${decision.choice}`);
    }

    if (chunk.startsWith('end\n')) break;
  }

  const resolvedLogDir = path.resolve(repoRoot, logDir || path.join('logs', 'battles'));
  fs.mkdirSync(resolvedLogDir, {recursive: true});
  const protocolFile = path.join(resolvedLogDir, `${battleId}.protocol.txt`);
  const replayFile = path.join(resolvedLogDir, `${battleId}.replay.html`);
  const summaryFile = path.join(resolvedLogDir, `${battleId}.summary.json`);
  const traceFile = path.join(resolvedLogDir, `${battleId}.trace.jsonl`);
  fs.writeFileSync(protocolFile, protocolLog.join('\n\n') + '\n', 'utf8');
  fs.writeFileSync(replayFile, replayHtml({
    title: `${p1Name} vs ${p2Name}`,
    replayLog: replayLines.join('\n'),
  }), 'utf8');
  for (const row of traceRows) row.outcome_context.winner = winner;
  fs.writeFileSync(traceFile, traceRows.map(row => JSON.stringify(row)).join('\n') + '\n', 'utf8');

  const summary = {
    battle_id: battleId,
    seed,
    format: pool.format_id,
    p1_team: p1Team.id,
    p2_team: p2Team.id,
    p1_lead: p1Lead.id,
    p2_lead: p2Lead.id,
    winner,
    turn_count: turns,
    protocol_log_path: path.relative(repoRoot, protocolFile).replace(/\\/g, '/'),
    replay_html_path: path.relative(repoRoot, replayFile).replace(/\\/g, '/'),
    trace_jsonl_path: path.relative(repoRoot, traceFile).replace(/\\/g, '/'),
  };
  fs.writeFileSync(summaryFile, JSON.stringify(summary, null, 2) + '\n', 'utf8');

  return {
    winner,
    turns,
    seed,
    battle_id: battleId,
    showdown_seed: battleSeed,
    format_id: pool.format_id,
    p1: {...sideSummary(p1Team, p1Lead), agent_name: p1Agent.name},
    p2: {...sideSummary(p2Team, p2Lead), agent_name: p2Agent.name},
    protocol_log_path: summary.protocol_log_path,
    replay_html_path: summary.replay_html_path,
    summary_json_path: path.relative(repoRoot, summaryFile).replace(/\\/g, '/'),
    trace_jsonl_path: summary.trace_jsonl_path,
  };
}

module.exports = {
  findLeadMode,
  findTeam,
  loadTeamPool,
  makeRng,
  runBattle,
};
