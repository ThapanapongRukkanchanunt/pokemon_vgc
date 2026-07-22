const path = require('node:path');

const repoRoot = path.join(__dirname, '..', '..');
const showdownRoot = path.join(repoRoot, 'vendor', 'pokemon-showdown');
const {
  Battle,
  extractChannelMessages,
} = require(path.join(showdownRoot, 'dist', 'sim', 'battle.js'));
const {applySpectatorLineToState, createPublicState} = require('./public_state');

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function sideId(index) {
  return `p${index + 1}`;
}

function publicStateFromBattle(battle) {
  const publicState = createPublicState();
  const spectatorLines = extractChannelMessages((battle.log || []).join('\n'), [0])[0];
  for (const line of spectatorLines) {
    applySpectatorLineToState(publicState, line);
  }
  return publicState;
}

function requestsFromBattle(battle) {
  const requests = {};
  for (const [index, side] of battle.sides.entries()) {
    if (side.activeRequest) requests[sideId(index)] = deepClone(side.activeRequest);
  }
  return requests;
}

function battleStateFromBattle({battle, teams, leadModes}) {
  return {
    turns: battle.turn || 0,
    teams,
    leadModes,
    requests: requestsFromBattle(battle),
    publicState: publicStateFromBattle(battle),
  };
}

function normalizeChoice(choice) {
  if (!choice) return null;
  if (typeof choice === 'string') return choice;
  return choice.choice || choice.action || null;
}

function winnerSide(battle, perspectiveSide) {
  if (!battle.winner) return 'unknown';
  const perspective = battle.getSide(perspectiveSide);
  if (battle.winner === perspective.name) return perspectiveSide;
  for (const [index, side] of battle.sides.entries()) {
    if (battle.winner === side.name) return sideId(index);
  }
  return 'unknown';
}

function rolloutScore(battle, perspectiveSide) {
  const side = winnerSide(battle, perspectiveSide);
  if (side === perspectiveSide) return 1;
  if (side === 'unknown') return 0;
  return -1;
}

async function chooseLaterAction({battle, side, teams, leadModes, selectTopAction}) {
  const request = battle.getSide(side).activeRequest;
  if (!request) return {ok: true, choice: null};
  const battleState = battleStateFromBattle({battle, teams, leadModes});
  const selected = await selectTopAction({
    side,
    request: deepClone(request),
    battleState,
  });
  const choice = normalizeChoice(selected) || 'default';
  if (battle.choose(side, choice)) return {ok: true, choice};
  if (choice !== 'default' && battle.choose(side, 'default')) {
    return {ok: true, choice: 'default', fallback: true};
  }
  return {ok: false, choice};
}

async function runCandidate({
  snapshotText,
  candidate,
  side,
  teams,
  leadModes,
  selectTopAction,
  maxDecisions,
}) {
  const battle = Battle.fromJSON(JSON.parse(snapshotText));
  const choice = normalizeChoice(candidate);
  const ok = choice && battle.choose(side, choice);
  if (!ok) {
    return {
      ...candidate,
      status: 'invalid_initial_choice',
      rollout_score: -Infinity,
      winner: null,
      winner_side: 'unknown',
      turns: battle.turn || 0,
      decisions: 0,
    };
  }

  let decisions = 1;
  while (!battle.ended && decisions < maxDecisions) {
    const pendingSides = battle.sides
      .map((battleSide, index) => ({id: sideId(index), battleSide}))
      .filter(entry => entry.battleSide.activeRequest && !entry.battleSide.isChoiceDone());

    if (!pendingSides.length) break;

    for (const entry of pendingSides) {
      if (battle.ended || !entry.battleSide.activeRequest || entry.battleSide.isChoiceDone()) continue;
      const result = await chooseLaterAction({
        battle,
        side: entry.id,
        teams,
        leadModes,
        selectTopAction,
      });
      decisions += 1;
      if (!result.ok) {
        return {
          ...candidate,
          status: 'invalid_later_choice',
          invalid_choice: result.choice,
          rollout_score: -Infinity,
          winner: battle.winner || null,
          winner_side: winnerSide(battle, side),
          turns: battle.turn || 0,
          decisions,
        };
      }
      if (decisions >= maxDecisions) break;
    }
  }

  const status = battle.ended ? 'ended' : 'max_decisions';
  return {
    ...candidate,
    status,
    rollout_score: rolloutScore(battle, side),
    winner: battle.winner || null,
    winner_side: winnerSide(battle, side),
    turns: battle.turn || 0,
    decisions,
  };
}

function createRolloutSearch({
  battleSnapshot,
  teams,
  leadModes,
  maxDecisions = 80,
} = {}) {
  const snapshotText = typeof battleSnapshot === 'string' ?
    battleSnapshot :
    JSON.stringify(battleSnapshot);

  return {
    maxDecisions,
    async evaluateCandidates({side, candidates, selectTopAction}) {
      const candidateList = (candidates || []).filter(candidate => candidate && candidate.action);
      if (!candidateList.length || typeof selectTopAction !== 'function') {
        return {enabled: false, max_decisions: maxDecisions, candidates: [], best: null};
      }

      const results = [];
      for (const candidate of candidateList) {
        try {
          results.push(await runCandidate({
            snapshotText,
            candidate,
            side,
            teams,
            leadModes,
            selectTopAction,
            maxDecisions,
          }));
        } catch (error) {
          results.push({
            ...candidate,
            status: 'error',
            error: error.message,
            rollout_score: -Infinity,
            winner: null,
            winner_side: 'unknown',
            turns: 0,
            decisions: 0,
          });
        }
      }

      const best = results
        .filter(result => Number.isFinite(result.rollout_score))
        .sort((a, b) => (
          b.rollout_score - a.rollout_score ||
          Number(b.model_score || 0) - Number(a.model_score || 0) ||
          Number(a.index || 0) - Number(b.index || 0)
        ))[0] || null;

      return {
        enabled: true,
        max_decisions: maxDecisions,
        candidates: results,
        best,
      };
    },
  };
}

module.exports = {
  createRolloutSearch,
  publicStateFromBattle,
};
