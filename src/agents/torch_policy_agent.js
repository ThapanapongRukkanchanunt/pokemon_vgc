const fs = require('node:fs');
const path = require('node:path');
const {spawn} = require('node:child_process');
const {dexForFormat} = require('../battle/showdown_protocol');
const {enumerateLegalActions} = require('./legal_actions');
const {PolicySelector} = require('../selectors');

const repoRoot = path.join(__dirname, '..', '..');
const scorerCache = new Map();

function resolveRepoPath(filePath) {
  if (!filePath) throw new Error('PyTorch policy agent requires a modelPath checkpoint');
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

function exampleFromBattle({side, request, battleState}) {
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
  };
}

function pythonExecutable() {
  return process.platform === 'win32' ? 'python' : 'python3';
}

class TorchPolicyScorer {
  constructor({modelPath, pythonPath = null, device = 'auto'} = {}) {
    this.modelPath = resolveRepoPath(modelPath);
    if (!fs.existsSync(this.modelPath)) {
      throw new Error(`PyTorch policy checkpoint not found: ${this.modelPath}`);
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
    const scriptPath = path.join(repoRoot, 'scripts', 'torch_policy_server.py');
    this.child = spawn(this.pythonPath, [
      scriptPath,
      '--policy-checkpoint', this.modelPath,
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
        `PyTorch scorer exited code=${code} signal=${signal || ''}: ${this.stderr.trim()}`
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
          // Ignore shutdown races; the process may already be gone.
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
          // Ignore shutdown races; the caller only needs the scorer closed.
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
        this.stderr += `\nInvalid scorer JSON: ${line}`;
        continue;
      }
      if (message.ready) {
        this.ready = true;
        if (this._resolveReady) this._resolveReady(message);
        continue;
      }
      const id = message.id;
      const pending = this.pending.get(id);
      if (!pending) continue;
      this.pending.delete(id);
      if (message.ok) pending.resolve(message);
      else pending.reject(new Error(message.error || 'PyTorch scorer failed'));
    }
  }

  async score({example, legalActions}) {
    if (this.closed) throw new Error('PyTorch scorer is closed');
    await this.readyPromise;
    const id = this.nextId++;
    const payload = {
      id,
      example,
      legal_actions: legalActions,
    };
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
  if (!scorerCache.has(key)) scorerCache.set(key, new TorchPolicyScorer(options));
  return scorerCache.get(key);
}

async function closeTorchPolicyScorers() {
  const scorers = [...scorerCache.values()];
  scorerCache.clear();
  await Promise.allSettled(scorers.map(scorer => scorer.close()));
}

function createTorchPolicyAgent({
  modelPath,
  pythonPath = null,
  torchDevice = null,
  formatId = 'vgc',
} = {}) {
  const resolvedModelPath = resolveRepoPath(modelPath);
  const scorer = getScorer({modelPath: resolvedModelPath, pythonPath, device: torchDevice});
  const selector = new PolicySelector();
  return {
    name: 'torch_policy_agent',
    displayName: 'TorchPolicy',
    modelPath: path.relative(repoRoot, resolvedModelPath).replace(/\\/g, '/'),
    selector: selector.name,
    async chooseAction({side, request, battleState, rng}) {
      if (request.wait) return null;
      const dex = dexForFormat(formatId);
      const sideBattleState = battleStateForSide(battleState, side);
      const legalActions = enumerateLegalActions({side, request, battleState: sideBattleState, dex});
      if (!legalActions.length) return null;
      const choices = legalActions.map(action => action.choice);
      const example = exampleFromBattle({side, request, battleState: sideBattleState});
      const response = await scorer.score({example, legalActions: choices});
      const modelScores = choices.map((choice, index) => ({
        choice,
        score: Number(response.scores?.[index]),
      }));
      const selected = selector.choose({
        request,
        legalActions,
        modelScores,
        rng,
      });
      return selected ? selected.choice : rng.pick(legalActions).choice;
    },
  };
}

module.exports = {
  closeTorchPolicyScorers,
  createTorchPolicyAgent,
};
