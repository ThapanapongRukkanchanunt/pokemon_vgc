const fs = require('node:fs');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {dexForFormat} = require('../battle/showdown_protocol');
const {HMMBeliefState} = require('../belief/hmm_belief');
const {enumerateLegalActions} = require('./legal_actions');

const repoRoot = path.join(__dirname, '..', '..');
const scorerCache = new Map();
const teamSummaryCache = new Map();

function resolveRepoPath(filePath) {
  if (!filePath) throw new Error('PPO policy agent requires a modelPath checkpoint');
  return path.isAbsolute(filePath) ? filePath : path.resolve(repoRoot, filePath);
}

function requestType(request) {
  if (!request || typeof request !== 'object') return 'other';
  if (request.teamPreview) return 'team_preview';
  if (Array.isArray(request.forceSwitch) && request.forceSwitch.some(Boolean)) return 'force_switch';
  if (Array.isArray(request.active) && request.active.length) return 'move';
  if (request.wait) return 'wait';
  return 'other';
}

function battleStateForSide(battleState, side) {
  if (!battleState) return battleState;
  return {
    ...battleState,
    team: battleState.teams?.[side] || battleState.team,
    leadMode: battleState.leadModes?.[side] || battleState.leadMode,
  };
}

function exampleFromBattle({side, request, battleState, diagnostics = null}) {
  const sideBattleState = battleStateForSide(battleState, side) || {};
  return {
    request_type: requestType(request),
    side,
    team: sideBattleState.team?.id || 'unknown',
    lead: sideBattleState.leadMode?.id || 'unknown',
    turn: sideBattleState.turns || battleState?.turns || 0,
    state: {
      request,
      public_state: sideBattleState.publicState || battleState?.publicState || {},
    },
    agent_diagnostics: diagnostics || undefined,
  };
}

function pythonExecutable() {
  return process.platform === 'win32' ? 'python' : 'python3';
}

function finiteOr(value, fallback) {
  return Number.isFinite(value) ? value : fallback;
}

function clamp01(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(1, number));
}

function pickIndex(probabilities, rng) {
  const total = probabilities.reduce((sum, value) => sum + Math.max(0, finiteOr(value, 0)), 0);
  if (!(total > 0)) return 0;
  let cursor = (rng ? rng.next() : Math.random()) * total;
  for (let index = 0; index < probabilities.length; index++) {
    cursor -= Math.max(0, finiteOr(probabilities[index], 0));
    if (cursor <= 0) return index;
  }
  return probabilities.length - 1;
}

function bestIndex(scores) {
  let best = 0;
  let bestScore = -Infinity;
  scores.forEach((score, index) => {
    const value = finiteOr(score, -Infinity);
    if (value > bestScore) {
      best = index;
      bestScore = value;
    }
  });
  return best;
}

function topKIndices(scores, k) {
  return scores
    .map((score, index) => ({index, score: finiteOr(score, -Infinity)}))
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .slice(0, Math.max(1, k))
    .map(entry => entry.index);
}

function parseSetBlock(block, slot) {
  const lines = block.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  const header = lines[0] || '';
  const [speciesPart, itemPart = ''] = header.split(/\s+@\s+/);
  const set = {
    slot,
    species: speciesPart.trim(),
    item: itemPart.trim(),
    ability: '',
    nature: '',
    evs: '',
    moves: [],
  };
  for (const line of lines.slice(1)) {
    if (line.startsWith('Ability:')) {
      set.ability = line.slice('Ability:'.length).trim();
    } else if (line.startsWith('EVs:')) {
      set.evs = line.slice('EVs:'.length).trim();
    } else if (line.endsWith(' Nature')) {
      set.nature = line.slice(0, -' Nature'.length).trim();
    } else if (line.startsWith('- ')) {
      set.moves.push(line.slice(2).trim());
    }
  }
  return set;
}

function loadTeamSummary(team) {
  if (!team?.import_file) return null;
  if (teamSummaryCache.has(team.id)) return teamSummaryCache.get(team.id);
  const importPath = path.resolve(repoRoot, team.import_file);
  const text = fs.readFileSync(importPath, 'utf8');
  const sets = text
    .split(/\n\s*\n/)
    .map((block, index) => parseSetBlock(block, index + 1))
    .filter(set => set.species);
  const summary = {
    id: team.id,
    name: team.name,
    representative_mega: team.representative_mega || null,
    primary_megas: team.primary_megas || [],
    sets,
  };
  teamSummaryCache.set(team.id, summary);
  return summary;
}

function activeSpecies(publicState, side) {
  return new Set((publicState?.active?.[side] || [])
    .filter(active => active && !active.fainted && active.species)
    .map(active => active.species));
}

function predictedBackMons(teamSummary, publicState, foeSide) {
  if (!teamSummary) return [];
  const actives = activeSpecies(publicState, foeSide);
  return teamSummary.sets
    .filter(set => !actives.has(set.species))
    .map(set => ({
      slot: set.slot,
      species: set.species,
      item: set.item,
      ability: set.ability,
      nature: set.nature,
      evs: set.evs,
    }));
}

function teamContextForSide({side, battleState}) {
  const ownSide = side;
  const foeSide = side === 'p1' ? 'p2' : 'p1';
  const ownTeam = loadTeamSummary(battleState?.teams?.[ownSide]);
  const foeTeam = loadTeamSummary(battleState?.teams?.[foeSide]);
  return {
    source: 'team_pool_roster',
    own_team: ownTeam,
    opponent_team: foeTeam,
    predicted_opponent_back: predictedBackMons(foeTeam, battleState?.publicState, foeSide),
  };
}

class PpoPolicyScorer {
  constructor({modelPath, pythonPath = null, device = 'auto'} = {}) {
    this.modelPath = resolveRepoPath(modelPath);
    if (!fs.existsSync(this.modelPath)) {
      throw new Error(`PPO policy checkpoint not found: ${this.modelPath}`);
    }
    this.pythonPath = pythonPath || process.env.POKEMON_RL_PYTHON || pythonExecutable();
    this.device = device || process.env.POKEMON_RL_TORCH_DEVICE || 'auto';
    this.nextId = 1;
    this.pending = new Map();
    this.buffer = '';
    this.stderr = '';
    this.ready = false;
    this.readyPromise = null;
    this.closed = false;
    this.closePromise = null;
    this.start();
  }

  start() {
    const scriptPath = path.join(repoRoot, 'scripts', 'torch_ppo_server.py');
    this.child = spawn(this.pythonPath, [
      scriptPath,
      '--checkpoint', this.modelPath,
      '--device', this.device,
    ], {
      cwd: repoRoot,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.child.stdout.setEncoding('utf8');
    this.child.stderr.setEncoding('utf8');
    this.child.stdout.on('data', chunk => this.onStdout(chunk));
    this.child.stderr.on('data', chunk => {
      this.stderr += chunk;
      if (this.stderr.length > 8000) this.stderr = this.stderr.slice(-8000);
    });
    this.child.on('exit', (code, signal) => {
      const error = new Error(
        `PPO scorer exited code=${code} signal=${signal || ''}: ${this.stderr.trim()}`
      );
      for (const {reject} of this.pending.values()) reject(error);
      this.pending.clear();
      if (this._rejectReady && !this.ready) this._rejectReady(error);
    });
    this.readyPromise = new Promise((resolve, reject) => {
      this._resolveReady = resolve;
      this._rejectReady = reject;
    });
  }

  close() {
    if (this.closePromise) return this.closePromise;
    this.closed = true;
    this.closePromise = new Promise(resolve => {
      if (!this.child || this.child.exitCode !== null) {
        resolve();
        return;
      }
      const timer = setTimeout(() => {
        try {
          this.child.kill();
        } catch (error) {
          // The process may already be gone.
        }
      }, 1000);
      this.child.once('exit', () => {
        clearTimeout(timer);
        resolve();
      });
      try {
        this.child.stdin.end();
      } catch (error) {
        try {
          this.child.kill();
        } catch (killError) {
          // Shutdown is best-effort.
        }
      }
    });
    return this.closePromise;
  }

  onStdout(chunk) {
    this.buffer += chunk;
    for (;;) {
      const newline = this.buffer.indexOf('\n');
      if (newline < 0) break;
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (!line) continue;
      let message;
      try {
        message = JSON.parse(line);
      } catch (error) {
        this.stderr += `\nInvalid PPO scorer JSON: ${line}`;
        continue;
      }
      if (message.ready) {
        this.ready = true;
        if (this._resolveReady) this._resolveReady(message);
        continue;
      }
      const pending = this.pending.get(message.id);
      if (!pending) continue;
      this.pending.delete(message.id);
      if (message.ok) pending.resolve(message);
      else pending.reject(new Error(message.error || 'PPO scorer failed'));
    }
  }

  async score({example, legalActions}) {
    if (this.closed) throw new Error('PPO scorer is closed');
    await this.readyPromise;
    const id = this.nextId++;
    const payload = {id, example, legal_actions: legalActions};
    return new Promise((resolve, reject) => {
      this.pending.set(id, {resolve, reject});
      this.child.stdin.write(`${JSON.stringify(payload)}\n`, error => {
        if (!error) return;
        this.pending.delete(id);
        reject(error);
      });
    });
  }
}

function scorerKey(options) {
  return [
    resolveRepoPath(options.modelPath),
    options.pythonPath || process.env.POKEMON_RL_PYTHON || pythonExecutable(),
    options.device || process.env.POKEMON_RL_TORCH_DEVICE || 'auto',
  ].join('|');
}

function getScorer(options) {
  const key = scorerKey(options);
  if (!scorerCache.has(key)) scorerCache.set(key, new PpoPolicyScorer(options));
  return scorerCache.get(key);
}

async function closePpoPolicyScorers() {
  const scorers = [...scorerCache.values()];
  scorerCache.clear();
  await Promise.allSettled(scorers.map(scorer => scorer.close()));
}

function createPpoPolicyAgent({
  modelPath,
  teamPreviewModelPath = null,
  pythonPath = null,
  torchDevice = null,
  formatId = 'vgc',
  sampleActions = false,
  epsilon = 0,
  topK = 1,
  fallbackConfidence = 0,
  useHmmBelief = true,
} = {}) {
  const resolvedModelPath = resolveRepoPath(modelPath);
  const scorer = getScorer({modelPath: resolvedModelPath, pythonPath, device: torchDevice});
  const resolvedPreviewModelPath = teamPreviewModelPath ? resolveRepoPath(teamPreviewModelPath) : null;
  const previewScorer = resolvedPreviewModelPath ?
    getScorer({modelPath: resolvedPreviewModelPath, pythonPath, device: torchDevice}) :
    scorer;
  const relativeModelPath = path.relative(repoRoot, resolvedModelPath).replace(/\\/g, '/');
  const relativePreviewModelPath = resolvedPreviewModelPath ?
    path.relative(repoRoot, resolvedPreviewModelPath).replace(/\\/g, '/') :
    relativeModelPath;
  const belief = useHmmBelief ? new HMMBeliefState() : null;
  const explorationEpsilon = clamp01(epsilon);
  const requestedTopK = Math.max(1, Number.isInteger(topK) ? topK : Number(topK) || 1);

  async function scorePolicyRequest({side, request, battleState, diagnostics = null}) {
    const dex = dexForFormat(formatId);
    const sideBattleState = battleStateForSide(battleState, side);
    const legalActions = enumerateLegalActions({side, request, battleState: sideBattleState, dex});
    if (!legalActions.length) {
      return {
        sideBattleState,
        legalActions,
        choices: [],
        probabilities: [],
        scores: [],
        response: {value: 0, entropy: 0},
        teamContext: teamContextForSide({side, battleState: sideBattleState}),
        requestIsTeamPreview: requestType(request) === 'team_preview',
      };
    }
    const choices = legalActions.map(action => action.choice);
    const teamContext = teamContextForSide({side, battleState: sideBattleState});
    const example = exampleFromBattle({side, request, battleState: sideBattleState, diagnostics});
    example.state.team_context = teamContext;
    const requestIsTeamPreview = requestType(request) === 'team_preview';
    const activeScorer = requestIsTeamPreview ? previewScorer : scorer;
    const response = await activeScorer.score({example, legalActions: choices});
    return {
      sideBattleState,
      legalActions,
      choices,
      teamContext,
      requestIsTeamPreview,
      response,
      probabilities: (response.probabilities || []).map(value => finiteOr(Number(value), 0)),
      scores: (response.scores || []).map(value => finiteOr(Number(value), -Infinity)),
    };
  }

  async function selectTopModelAction({side, request, battleState}) {
    const scored = await scorePolicyRequest({side, request, battleState, diagnostics: null});
    if (!scored.choices.length) return null;
    const index = bestIndex(scored.scores);
    return {
      index,
      choice: scored.choices[index],
      score: scored.scores[index],
      probability: scored.probabilities[index] || 0,
    };
  }

  return {
    name: 'ppo_policy_agent',
    displayName: 'FinalRL',
    modelPath: relativeModelPath,
    teamPreviewModelPath: relativePreviewModelPath,
    selector: sampleActions ? 'ppo_sample' : 'ppo_greedy',
    lastDiagnostics: null,
    async chooseAction({side, request, battleState, rng}) {
      if (request.wait) return null;
      const sideBattleState = battleStateForSide(battleState, side);
      const hmmDiagnostics = belief ? {hmm_belief: belief.update({side, request, battleState: sideBattleState})} : null;
      const scored = await scorePolicyRequest({side, request, battleState: sideBattleState, diagnostics: hmmDiagnostics});
      if (!scored.choices.length) return null;
      const {
        choices,
        teamContext,
        requestIsTeamPreview,
        response,
        probabilities,
        scores,
      } = scored;
      const topIndices = topKIndices(scores, Math.min(requestedTopK, choices.length));
      let actionIndex = sampleActions ? pickIndex(probabilities, rng) : bestIndex(scores);
      let rolloutSearch = null;
      if (
        !sampleActions &&
        !requestIsTeamPreview &&
        requestedTopK > 1 &&
        sideBattleState?.rolloutSearch &&
        typeof sideBattleState.rolloutSearch.evaluateCandidates === 'function'
      ) {
        rolloutSearch = await sideBattleState.rolloutSearch.evaluateCandidates({
          side,
          candidates: topIndices.map(index => ({
            index,
            action: choices[index],
            model_score: scores[index],
            model_probability: probabilities[index] || 0,
          })),
          selectTopAction: selectTopModelAction,
        });
        if (Number.isInteger(rolloutSearch?.best?.index)) {
          actionIndex = rolloutSearch.best.index;
        }
      }
      let explored = false;
      if (explorationEpsilon > 0 && (rng ? rng.next() : Math.random()) < explorationEpsilon) {
        actionIndex = rng ? Math.floor(rng.next() * choices.length) : Math.floor(Math.random() * choices.length);
        explored = true;
      }
      const uniformProbability = 1 / choices.length;
      let modelProbability = probabilities[actionIndex] || 0;
      let selectedProbability =
        explorationEpsilon > 0 ?
          (1 - explorationEpsilon) * modelProbability + explorationEpsilon * uniformProbability :
          modelProbability;
      if (fallbackConfidence > 0 && selectedProbability < fallbackConfidence) {
        actionIndex = bestIndex(scores);
        modelProbability = probabilities[actionIndex] || 0;
        selectedProbability =
          explorationEpsilon > 0 ?
            (1 - explorationEpsilon) * modelProbability + explorationEpsilon * uniformProbability :
            modelProbability;
      }
      const logProb = Math.log(Math.max(selectedProbability, 1e-12));
      this.lastDiagnostics = {
        ...(hmmDiagnostics || {}),
        team_context: teamContext,
        ppo_policy: {
          action_index: actionIndex,
          log_prob: logProb,
          value_prediction: finiteOr(Number(response.value), 0),
          entropy: finiteOr(Number(response.entropy), 0),
          selected_probability: selectedProbability,
          model_probability: modelProbability,
          epsilon: explorationEpsilon,
          epsilon_explored: explored,
          top_k: topIndices.map(index => ({
            index,
            action: choices[index],
            score: scores[index],
            probability: probabilities[index] || 0,
          })),
          rollout_search: rolloutSearch ? {
            enabled: !!rolloutSearch.enabled,
            max_decisions: rolloutSearch.max_decisions,
            best_index: rolloutSearch.best?.index ?? null,
            best_action: rolloutSearch.best?.action ?? null,
            best_score: Number.isFinite(rolloutSearch.best?.rollout_score) ?
              rolloutSearch.best.rollout_score :
              null,
            candidates: (rolloutSearch.candidates || []).map(candidate => ({
              index: candidate.index,
              action: candidate.action,
              model_score: candidate.model_score,
              rollout_score: Number.isFinite(candidate.rollout_score) ? candidate.rollout_score : null,
              status: candidate.status,
              winner_side: candidate.winner_side,
              turns: candidate.turns,
              decisions: candidate.decisions,
              error: candidate.error,
            })),
          } : undefined,
          sampled: !!sampleActions,
          model_path: requestIsTeamPreview ? relativePreviewModelPath : relativeModelPath,
          battle_model_path: relativeModelPath,
          team_preview_model_path: relativePreviewModelPath,
        },
      };
      return choices[actionIndex] || choices[0];
    },
    getDiagnostics() {
      return this.lastDiagnostics;
    },
  };
}

module.exports = {
  closePpoPolicyScorers,
  createPpoPolicyAgent,
  exampleFromBattle,
};
