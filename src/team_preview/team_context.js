const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.join(__dirname, '..', '..');
const teamSummaryCache = new Map();

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
  if (team?.team_summary) return team.team_summary;
  if (Array.isArray(team?.sets)) {
    return {
      id: team.id || 'dynamic-team',
      name: team.name || team.id || 'Dynamic team',
      representative_mega: team.representative_mega || null,
      primary_megas: team.primary_megas || [],
      sets: team.sets,
    };
  }
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

function normalizedSpecies(species) {
  return String(species || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function speciesSet(values) {
  return new Set(Array.from(values || []).map(normalizedSpecies).filter(Boolean));
}

function predictedBackMons(teamSummary, publicState, foeSide) {
  if (!teamSummary) return [];
  const actives = speciesSet(activeSpecies(publicState, foeSide));
  const revealed = speciesSet(publicState?.revealed?.[foeSide]);
  const fainted = speciesSet(publicState?.fainted?.[foeSide]);
  const selectedKnown = revealed.size >= 4;
  return teamSummary.sets
    .filter(set => {
      const species = normalizedSpecies(set.species);
      if (actives.has(species) || fainted.has(species)) return false;
      return !selectedKnown || revealed.has(species);
    })
    .map(set => ({
      slot: set.slot,
      species: set.species,
      item: set.item,
      ability: set.ability,
      nature: set.nature,
      evs: set.evs,
      revealed: revealed.has(normalizedSpecies(set.species)),
      selected_known: selectedKnown,
    }));
}

function teamContextFromTeams({ownTeam, opponentTeam, publicState = null, foeSide = null}) {
  const opponentSummary = loadTeamSummary(opponentTeam);
  return {
    source: 'team_pool_roster',
    own_team: loadTeamSummary(ownTeam),
    opponent_team: opponentSummary,
    predicted_opponent_back: foeSide ? predictedBackMons(opponentSummary, publicState, foeSide) : [],
  };
}

function teamContextForSide({side, battleState}) {
  const foeSide = side === 'p1' ? 'p2' : 'p1';
  return teamContextFromTeams({
    ownTeam: battleState?.teams?.[side],
    opponentTeam: battleState?.teams?.[foeSide],
    publicState: battleState?.publicState,
    foeSide,
  });
}

module.exports = {
  loadTeamSummary,
  teamContextForSide,
  teamContextFromTeams,
};
