const fs = require('node:fs');
const path = require('node:path');
const {Teams} = require('../../vendor/pokemon-showdown/dist/sim/teams.js');

const repoRoot = path.join(__dirname, '..', '..');
let cachedPrior = null;

function toId(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function evString(evs) {
  if (!evs || typeof evs !== 'object') return '';
  const labels = {hp: 'HP', atk: 'Atk', def: 'Def', spa: 'SpA', spd: 'SpD', spe: 'Spe'};
  return Object.entries(labels)
    .filter(([stat]) => Number(evs[stat]) > 0)
    .map(([stat, label]) => `${evs[stat]} ${label}`)
    .join(' / ');
}

function loadCuratedSpreadPrior() {
  if (cachedPrior) return cachedPrior;
  const pool = JSON.parse(fs.readFileSync(path.join(repoRoot, 'data', 'teams', 'team_pool.json'), 'utf8'));
  cachedPrior = pool.teams.flatMap(team => {
    const importText = fs.readFileSync(path.join(repoRoot, team.import_file), 'utf8');
    return (Teams.import(importText) || []).map(set => ({
      team_id: team.id,
      species: set.species || set.name || '',
      item: set.item || '',
      ability: set.ability || '',
      moves: set.moves || [],
      nature: set.nature || '',
      evs: evString(set.evs),
    }));
  });
  return cachedPrior;
}

function matchScore(observed, candidate) {
  if (toId(observed.species) !== toId(candidate.species)) return -Infinity;
  let score = 1;
  if (observed.item) score += toId(observed.item) === toId(candidate.item) ? 8 : -4;
  if (observed.ability) score += toId(observed.ability) === toId(candidate.ability) ? 6 : -3;
  const candidateMoves = new Set((candidate.moves || []).map(toId));
  for (const move of observed.moves || []) score += candidateMoves.has(toId(move)) ? 2 : -1;
  return score;
}

function inferSpread(set, prior = loadCuratedSpreadPrior()) {
  if (set.nature && set.evs) return {...set, spread_source: 'observed'};
  const ranked = prior
    .map(entry => ({entry, score: matchScore(set, entry)}))
    .filter(result => Number.isFinite(result.score))
    .sort((a, b) => b.score - a.score || a.entry.team_id.localeCompare(b.entry.team_id));
  if (!ranked.length) return {...set, spread_source: 'unknown'};
  const selected = ranked[0];
  const observedMoves = (set.moves || []).filter(Boolean);
  const exactFingerprint = !!set.item && !!set.ability && observedMoves.length >= 4 &&
    toId(set.item) === toId(selected.entry.item) &&
    toId(set.ability) === toId(selected.entry.ability) &&
    observedMoves.every(move => selected.entry.moves.map(toId).includes(toId(move)));
  return {
    ...set,
    nature: set.nature || selected.entry.nature,
    evs: set.evs || selected.entry.evs,
    spread_source: 'curated_mb_prior',
    spread_confidence: exactFingerprint ? 'high' : (ranked.length === 1 ? 'medium' : 'low'),
    spread_reference_team: selected.entry.team_id,
  };
}

function inferTeamSpreads(sets, prior = loadCuratedSpreadPrior()) {
  return (sets || []).map(set => inferSpread(set, prior));
}

module.exports = {
  inferSpread,
  inferTeamSpreads,
  loadCuratedSpreadPrior,
  matchScore,
};
