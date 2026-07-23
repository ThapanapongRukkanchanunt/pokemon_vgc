const fs = require('node:fs');
const path = require('node:path');
const {Teams} = require('../vendor/pokemon-showdown/dist/sim/teams.js');
const {applySpectatorLineToState, createPublicState} = require('./battle/public_state');
const {replayHtml} = require('./battle/replay_export');
const {inferTeamSpreads} = require('./team_preview/spread_prior');

function toId(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function splitServerPayload(payload) {
  const text = String(payload || '').replace(/\r/g, '');
  if (!text.startsWith('>')) return [{roomId: '', lines: text.split('\n').filter(Boolean)}];
  return text.split(/\n(?=>)/).map(block => {
    const newline = block.indexOf('\n');
    if (newline < 0) return {roomId: block.slice(1), lines: []};
    return {
      roomId: block.slice(1, newline),
      lines: block.slice(newline + 1).split('\n').filter(Boolean),
    };
  });
}

function evString(evs) {
  if (!evs || typeof evs !== 'object') return '';
  const labels = {hp: 'HP', atk: 'Atk', def: 'Def', spa: 'SpA', spd: 'SpD', spe: 'Spe'};
  return Object.entries(labels)
    .filter(([stat]) => Number(evs[stat]) > 0)
    .map(([stat, label]) => `${evs[stat]} ${label}`)
    .join(' / ');
}

function teamSummaryFromSets(sets, id = 'ladder-opponent', name = 'Ladder opponent') {
  const normalizedSets = inferTeamSpreads((sets || []).map((set, index) => ({
    slot: index + 1,
    species: set.species || set.name || 'Unknown',
    item: set.item || '',
    ability: set.ability || '',
    nature: set.nature || '',
    evs: typeof set.evs === 'string' ? set.evs : evString(set.evs),
    moves: set.moves || [],
  })));
  const megas = normalizedSets
    .map(set => set.species)
    .filter(species => /-mega(?:-|$)/i.test(species));
  return {
    id,
    name,
    representative_mega: megas[0] || null,
    primary_megas: [...new Set(megas)],
    sets: normalizedSets,
  };
}

function teamSummaryFromPacked(packed, id = 'ladder-opponent', name = 'Ladder opponent') {
  const sets = Teams.unpack(packed);
  if (!sets) throw new Error('Could not unpack Showdown open team sheet');
  return teamSummaryFromSets(sets, id, name);
}

function partialSetFromPokeLine(line) {
  const parts = line.split('|');
  const details = parts[3] || '';
  return {
    species: details.split(',')[0].trim(),
    item: parts[4] && parts[4] !== 'item' ? parts[4] : '',
    ability: '',
    nature: '',
    evs: '',
    moves: [],
  };
}

function isReplayLine(line) {
  return ![
    '|request|',
    '|showteam|',
    '|queryresponse|',
    '|sentchoice|',
  ].some(prefix => line.startsWith(prefix));
}

class LadderBattle {
  constructor({roomId, username, ownTeam, agent, send, log = () => {}}) {
    this.roomId = roomId;
    this.username = username;
    this.ownTeam = ownTeam;
    this.agent = agent;
    this.send = send;
    this.log = log;
    this.ownSide = null;
    this.turns = 0;
    this.publicState = createPublicState();
    this.partialSets = {p1: [], p2: []};
    this.dynamicTeams = {p1: null, p2: null};
    this.lastRequest = null;
    this.lastRqid = null;
    this.decidedRqids = new Set();
    this.finished = false;
    this.result = null;
    this.timerStarted = false;
    this.otsAccepted = false;
    this.replayLines = [];
    this.opponentName = null;
  }

  roomSend(text) {
    this.send(`${this.roomId}|${text}`);
  }

  opponentSide() {
    return this.ownSide === 'p1' ? 'p2' : 'p1';
  }

  teamForSide(side) {
    if (side === this.ownSide) return this.ownTeam;
    return this.dynamicTeams[side] || {
      id: 'ladder-opponent',
      team_summary: teamSummaryFromSets(this.partialSets[side]),
    };
  }

  battleState() {
    const foeSide = this.opponentSide();
    return {
      turns: this.turns,
      team: this.ownTeam,
      leadMode: {id: 'ladder'},
      teams: {
        [this.ownSide]: this.ownTeam,
        [foeSide]: this.teamForSide(foeSide),
      },
      leadModes: {
        [this.ownSide]: {id: 'ladder'},
        [foeSide]: {id: 'unknown'},
      },
      requests: {[this.ownSide]: this.lastRequest},
      publicState: this.publicState,
      rolloutSearch: null,
    };
  }

  async choose(request) {
    if (!this.ownSide && request.side?.id) this.ownSide = request.side.id;
    if (!this.ownSide || request.wait) return;
    const rqid = request.rqid == null ? `turn-${this.turns}` : String(request.rqid);
    if (this.decidedRqids.has(rqid)) return;
    this.decidedRqids.add(rqid);
    this.lastRequest = request;
    this.lastRqid = request.rqid;
    let choice;
    try {
      choice = await Promise.resolve(this.agent.chooseAction({
        side: this.ownSide,
        request,
        battleState: this.battleState(),
        rng: null,
      }));
    } catch (error) {
      const suffix = request.rqid == null ? '' : `|${request.rqid}`;
      this.roomSend(`/choose default${suffix}`);
      this.log({
        type: 'inference_error_recovery',
        room_id: this.roomId,
        turn: this.turns,
        rqid: request.rqid,
        error: error.stack || error.message,
      });
      return;
    }
    if (!choice) return;
    const suffix = request.rqid == null ? '' : `|${request.rqid}`;
    this.roomSend(`/choose ${choice}${suffix}`);
    this.log({type: 'decision', room_id: this.roomId, turn: this.turns, rqid: request.rqid, choice});
  }

  async handleLines(lines) {
    for (const line of lines) {
      if (isReplayLine(line)) this.replayLines.push(line);
      if (line.startsWith('|player|')) {
        const [, , side, name] = line.split('|');
        if (toId(name) === toId(this.username)) {
          this.ownSide = side;
        } else if (name) {
          this.opponentName = name;
        }
      } else if (line.startsWith('|showteam|')) {
        const payload = line.slice('|showteam|'.length);
        const separator = payload.indexOf('|');
        if (separator < 0) throw new Error(`Malformed showteam line in ${this.roomId}`);
        const side = payload.slice(0, separator);
        const packed = payload.slice(separator + 1);
        this.dynamicTeams[side] = {
          id: side === this.ownSide ? this.ownTeam.id : 'ladder-opponent',
          team_summary: teamSummaryFromPacked(
            packed,
            side === this.ownSide ? this.ownTeam.id : 'ladder-opponent',
            side === this.ownSide ? this.ownTeam.name : 'Ladder opponent'
          ),
        };
      } else if (line.startsWith('|poke|')) {
        const [, , side] = line.split('|');
        this.partialSets[side].push(partialSetFromPokeLine(line));
      } else if (line.startsWith('|turn|')) {
        this.turns = Number(line.slice('|turn|'.length)) || this.turns;
      } else if (line.startsWith('|request|')) {
        const raw = line.slice('|request|'.length);
        if (raw && raw !== 'null') await this.choose(JSON.parse(raw));
      } else if (line.startsWith('|error|') && line.includes('[Invalid choice]') && this.lastRequest) {
        const suffix = this.lastRqid == null ? '' : `|${this.lastRqid}`;
        this.roomSend(`/choose default${suffix}`);
        this.log({type: 'invalid_choice_recovery', room_id: this.roomId, turn: this.turns, error: line});
      } else if (line.startsWith('|win|')) {
        const winner = line.slice('|win|'.length);
        this.finished = true;
        this.result = toId(winner) === toId(this.username) ? 'win' : 'loss';
      } else if (line === '|tie|') {
        this.finished = true;
        this.result = 'tie';
      }

      if (!line.startsWith('|request|')) applySpectatorLineToState(this.publicState, line);
      if (!this.timerStarted && line === '|start') {
        this.timerStarted = true;
        this.roomSend('/timer on');
      }
      if (!this.otsAccepted && line.startsWith('|uhtml|otsrequest|')) {
        this.otsAccepted = true;
        this.roomSend('/acceptopenteamsheets');
      }
    }
  }
}

class ShowdownLadderClient {
  constructor({
    username,
    password,
    packedTeam,
    formatId,
    ownTeam,
    agent,
    agentFactory = null,
    maxBattles = 1,
    websocketUrl = 'wss://sim3.psim.us/showdown/websocket',
    loginUrl = 'https://play.pokemonshowdown.com/api/login',
    logPath = null,
    mode = 'ladder',
    replayDir = null,
    rejectMismatchedChallenges = true,
  }) {
    this.username = username;
    this.password = password;
    this.packedTeam = packedTeam;
    this.formatId = formatId;
    this.ownTeam = ownTeam;
    this.agent = agent;
    this.agentFactory = agentFactory;
    this.maxBattles = maxBattles;
    this.websocketUrl = websocketUrl;
    this.loginUrl = loginUrl;
    this.logPath = logPath;
    this.mode = mode;
    this.replayDir = replayDir;
    this.rejectMismatchedChallenges = rejectMismatchedChallenges;
    this.rooms = new Map();
    this.completedBattles = 0;
    this.results = {wins: 0, losses: 0, ties: 0};
    this.loggedIn = false;
    this.searching = false;
    this.challengeActions = new Map();
    this.queue = Promise.resolve();
    this.done = null;
    if (!['ladder', 'challenge'].includes(this.mode)) {
      throw new Error(`Unknown Showdown client mode: ${this.mode}`);
    }
  }

  log(event) {
    const row = {created_at: new Date().toISOString(), ...event};
    if (this.logPath) {
      fs.mkdirSync(path.dirname(this.logPath), {recursive: true});
      fs.appendFileSync(this.logPath, `${JSON.stringify(row)}\n`, 'utf8');
    }
    if (['decision', 'protocol'].includes(event.type)) return;
    console.log(JSON.stringify(row));
  }

  send(message) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) throw new Error('Showdown WebSocket is not open');
    this.ws.send(message);
    this.log({type: 'send', message: message.includes('/trn ') ? '|/trn [redacted]' : message});
  }

  async login(challstr) {
    const body = new URLSearchParams({name: this.username, pass: this.password, challstr});
    const response = await fetch(this.loginUrl, {method: 'POST', body});
    if (!response.ok) throw new Error(`Showdown login HTTP ${response.status}`);
    const raw = await response.text();
    const data = JSON.parse(raw.startsWith(']') ? raw.slice(1) : raw);
    if (!data.assertion) throw new Error(data.error || 'Showdown login returned no assertion');
    this.send(`|/trn ${this.username},0,${data.assertion}`);
  }

  startSearch() {
    if (this.mode !== 'ladder') return;
    if (!this.loggedIn || this.searching || this.completedBattles >= this.maxBattles) return;
    this.searching = true;
    this.send(`|/utm ${this.packedTeam}`);
    this.send(`|/search ${this.formatId}`);
    this.log({type: 'search_started', format_id: this.formatId});
  }

  handleChallenge(challenger, formatId) {
    if (this.mode !== 'challenge' || !this.loggedIn) return;
    const challengerId = toId(challenger);
    if (!challengerId || this.challengeActions.has(challengerId)) return;
    if (toId(formatId) === toId(this.formatId)) {
      this.send(`|/utm ${this.packedTeam}`);
      this.send(`|/accept ${challenger}`);
      this.challengeActions.set(challengerId, 'accepted');
      this.log({
        type: 'challenge_accepted',
        challenger,
        format_id: formatId,
      });
    } else if (this.rejectMismatchedChallenges) {
      this.send(`|/reject ${challenger}`);
      this.challengeActions.set(challengerId, 'rejected');
      this.log({
        type: 'challenge_rejected',
        challenger,
        format_id: formatId,
        expected_format_id: this.formatId,
      });
    }
  }

  handleChallenges(update) {
    if (this.mode !== 'challenge' || !this.loggedIn) return;
    const challenges = update?.challengesFrom || {};
    const activeIds = new Set(Object.keys(challenges).map(toId));
    for (const challengerId of this.challengeActions.keys()) {
      if (!activeIds.has(challengerId)) this.challengeActions.delete(challengerId);
    }
    for (const [challenger, formatId] of Object.entries(challenges)) {
      this.handleChallenge(challenger, formatId);
    }
  }

  handlePmChallenge(line) {
    const parts = line.split('|');
    const challenger = (parts[2] || '').trim();
    const command = parts[4] || '';
    const match = command.match(/^\/challenge\s+([a-z0-9]+)$/i);
    if (!match) return;
    this.handleChallenge(challenger, parts[5] || match[1]);
  }

  async handleGlobal(lines) {
    for (const line of lines) {
      if (line.startsWith('|challstr|')) {
        await this.login(line.slice('|challstr|'.length));
      } else if (line.startsWith('|updateuser|')) {
        const parts = line.split('|');
        const named = parts[3] === '1';
        if (named && toId(parts[2]) === toId(this.username)) {
          this.loggedIn = true;
          this.log({type: 'login_succeeded', username: this.username});
          if (this.mode === 'ladder') {
            this.startSearch();
          } else {
            this.log({type: 'challenge_listening', format_id: this.formatId});
          }
        }
      } else if (line.startsWith('|nametaken|')) {
        throw new Error(`Showdown login failed: ${line.split('|').slice(3).join('|')}`);
      } else if (line.startsWith('|popup|')) {
        const message = line.slice('|popup|'.length);
        this.log({type: 'popup', message});
        if (/invalid|not a valid team|not allowed/i.test(message)) throw new Error(message);
      } else if (line.startsWith('|updatesearch|')) {
        const update = JSON.parse(line.slice('|updatesearch|'.length));
        this.searching = Array.isArray(update.searching) && update.searching.includes(this.formatId);
      } else if (line.startsWith('|updatechallenges|')) {
        this.handleChallenges(JSON.parse(line.slice('|updatechallenges|'.length)));
      } else if (line.startsWith('|pm|')) {
        this.handlePmChallenge(line);
      }
    }
  }

  saveLocalReplay(room) {
    if (!this.replayDir) return null;
    fs.mkdirSync(this.replayDir, {recursive: true});
    const safeRoomId = room.roomId.replace(/[^a-zA-Z0-9._-]+/g, '_');
    const protocolPath = path.join(this.replayDir, `${safeRoomId}.protocol.txt`);
    const replayPath = path.join(this.replayDir, `${safeRoomId}.replay.html`);
    const replayLog = `${room.replayLines.join('\n')}\n`;
    fs.writeFileSync(protocolPath, replayLog, 'utf8');
    fs.writeFileSync(replayPath, replayHtml({
      title: room.roomId,
      replayLog,
      replayId: room.roomId.replace(/^battle-/, ''),
    }), 'utf8');
    return {protocolPath, replayPath};
  }

  async handleRoom(roomId, lines) {
    let room = this.rooms.get(roomId);
    if (!room && lines.includes('|init|battle')) {
      this.searching = false;
      room = new LadderBattle({
        roomId,
        username: this.username,
        ownTeam: this.ownTeam,
        agent: this.agentFactory ? this.agentFactory() : this.agent,
        send: message => this.send(message),
        log: event => this.log(event),
      });
      this.rooms.set(roomId, room);
      this.log({type: 'battle_started', room_id: roomId});
    }
    if (!room) return;
    await room.handleLines(lines);
    if (room.opponentName) this.challengeActions.delete(toId(room.opponentName));
    if (!room.finished || room.reported) return;
    room.reported = true;
    this.completedBattles += 1;
    if (room.result === 'win') this.results.wins += 1;
    else if (room.result === 'loss') this.results.losses += 1;
    else this.results.ties += 1;
    let localReplay = null;
    let localReplayError = null;
    try {
      localReplay = this.saveLocalReplay(room);
    } catch (error) {
      localReplayError = error.stack || error.message;
    }
    let serverReplayRequested = false;
    try {
      room.roomSend('/savereplay');
      serverReplayRequested = true;
    } catch (error) {
      this.log({type: 'server_replay_error', room_id: roomId, error: error.stack || error.message});
    }
    this.log({
      type: 'battle_finished',
      room_id: roomId,
      result: room.result,
      turns: room.turns,
      server_replay_requested: serverReplayRequested,
      replay_url: `https://replay.pokemonshowdown.com/${roomId.replace(/^battle-/, '')}`,
      local_protocol_path: localReplay?.protocolPath || null,
      local_replay_path: localReplay?.replayPath || null,
      local_replay_error: localReplayError,
    });
    if (this.maxBattles > 0 && this.completedBattles >= this.maxBattles) {
      this.finish();
    } else if (this.mode === 'ladder') {
      setTimeout(() => this.startSearch(), 1000);
    }
  }

  async handlePayload(payload) {
    for (const block of splitServerPayload(payload)) {
      this.log({type: 'protocol', room_id: block.roomId, lines: block.lines});
      if (block.roomId) await this.handleRoom(block.roomId, block.lines);
      else await this.handleGlobal(block.lines);
    }
  }

  finish(reason = 'completed') {
    if (!this.done) return;
    const summary = {battles: this.completedBattles, ...this.results, reason};
    const {resolve} = this.done;
    this.done = null;
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.close(1000, 'completed');
    resolve(summary);
  }

  stop(reason = 'stopped') {
    this.finish(reason);
  }

  run() {
    if (typeof WebSocket !== 'function') throw new Error('Node.js 22 or newer is required for global WebSocket support');
    return new Promise((resolve, reject) => {
      this.done = {resolve, reject};
      this.ws = new WebSocket(this.websocketUrl);
      this.ws.addEventListener('open', () => this.log({type: 'connected', url: this.websocketUrl}));
      this.ws.addEventListener('message', event => {
        this.queue = this.queue.then(() => this.handlePayload(event.data)).catch(error => {
          if (!this.done) return;
          const pending = this.done;
          this.done = null;
          pending.reject(error);
          this.ws.close();
        });
      });
      this.ws.addEventListener('error', () => {
        if (!this.done) return;
        const pending = this.done;
        this.done = null;
        pending.reject(new Error('Showdown WebSocket error'));
      });
      this.ws.addEventListener('close', event => {
        if (!this.done) return;
        const pending = this.done;
        this.done = null;
        pending.reject(new Error(`Showdown WebSocket closed before completion (${event.code})`));
      });
    });
  }
}

module.exports = {
  LadderBattle,
  ShowdownLadderClient,
  splitServerPayload,
  teamSummaryFromPacked,
  teamSummaryFromSets,
  toId,
};
