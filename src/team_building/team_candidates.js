const fs = require('node:fs');
const path = require('node:path');
const {dexForFormat} = require('../battle/showdown_protocol');

const repoRoot = path.join(__dirname, '..', '..');
const showdownRoot = path.join(repoRoot, 'vendor', 'pokemon-showdown');
const {Teams} = require(path.join(showdownRoot, 'dist', 'sim', 'teams.js'));
const {TeamValidator} = require(path.join(showdownRoot, 'dist', 'sim', 'team-validator.js'));

const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const DEFAULT_GENERATOR_OPTIONS = {
  minMegas: 1,
  maxMegas: 2,
  uniqueItems: true,
  maxAttempts: 5000,
};

function toId(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function hashString(value) {
  const text = String(value);
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

function relativePath(filePath) {
  return path.relative(repoRoot, filePath).replace(/\\/g, '/');
}

function resolveRepoPath(filePath) {
  return path.isAbsolute(filePath) ? filePath : path.resolve(repoRoot, filePath);
}

function loadPool(poolPath = path.join(repoRoot, 'data', 'teams', 'team_pool.json')) {
  return JSON.parse(fs.readFileSync(resolveRepoPath(poolPath), 'utf8'));
}

function cleanImportText(text) {
  return text.split(/\r?\n/).map(line => line.trimEnd()).join('\n').trim() + '\n';
}

function teamImportText(sets) {
  return cleanImportText(Teams.export(sets.map(set => clone(set)), {removeNicknames: true}));
}

function teamHash(sets) {
  return hashString(Teams.pack(sets.map(set => clone(set))));
}

function moveIds(set) {
  return (set.moves || []).map(move => toId(move));
}

function itemInfo(dex, itemName) {
  const item = dex.items.get(itemName || '');
  return {
    id: item?.exists ? item.id : toId(itemName),
    name: item?.exists ? item.name : (itemName || ''),
    megaSpecies: item?.megaStone || null,
    isMega: !!item?.megaStone,
  };
}

function speciesInfo(dex, speciesName) {
  const species = dex.species.get(speciesName || '');
  return {
    id: species?.exists ? species.id : toId(speciesName),
    name: species?.exists ? species.name : (speciesName || ''),
    types: species?.types || [],
  };
}

function hasAnyMove(set, ids) {
  const moves = new Set(moveIds(set));
  return ids.some(id => moves.has(id));
}

function classifySet({set, dex}) {
  const item = itemInfo(dex, set.item);
  let offensivePower = 0;
  let spreadDamage = false;
  let priorityDamage = false;

  for (const moveName of set.moves || []) {
    const move = dex.moves.get(moveName);
    if (!move?.exists || move.category === 'Status') continue;
    offensivePower += move.basePower || (typeof move.damage === 'number' ? move.damage : 0);
    if (['allAdjacentFoes', 'allAdjacent', 'all'].includes(move.target)) spreadDamage = true;
    if ((move.priority || 0) > 0) priorityDamage = true;
  }

  const speedControl = hasAnyMove(set, ['tailwind', 'trickroom', 'icywind', 'electroweb', 'bulldoze']);
  const fakeOut = hasAnyMove(set, ['fakeout']);
  const redirection = hasAnyMove(set, ['followme', 'ragepowder']);
  const pivot = hasAnyMove(set, ['uturn', 'voltswitch', 'flipturn', 'partingshot', 'batonpass', 'shedtail']);
  const setup = hasAnyMove(set, [
    'swordsdance',
    'dragondance',
    'quiverdance',
    'nastyplot',
    'calmmind',
    'shellsmash',
    'bulkup',
  ]);
  const protect = hasAnyMove(set, ['protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker']);
  const support = speedControl || fakeOut || redirection || pivot || protect;

  return {
    mega: item.isMega,
    speed_control: speedControl,
    fake_out: fakeOut,
    redirection,
    pivot,
    setup,
    protect,
    support,
    spread_damage: spreadDamage,
    priority_damage: priorityDamage,
    offense_score: offensivePower,
  };
}

function leadRoleScore(entry, role) {
  if (role === 'speed') return entry.roles.speed_control ? 100 : (entry.roles.support ? 20 : 0);
  if (role === 'fakeout') return entry.roles.fake_out ? 100 : (entry.roles.pivot ? 35 : 0);
  if (role === 'support') return entry.roles.support ? 70 : 0;
  if (role === 'offense') {
    return entry.roles.offense_score + (entry.roles.mega ? 80 : 0) + (entry.roles.spread_damage ? 30 : 0);
  }
  if (role === 'setup') return entry.roles.setup ? 100 : leadRoleScore(entry, 'offense') * 0.25;
  return 0;
}

function pickSlot(entries, usedSlots, role) {
  let best = null;
  for (let index = 0; index < entries.length; index++) {
    const slot = index + 1;
    if (usedSlots.has(slot)) continue;
    const score = leadRoleScore(entries[index], role);
    if (!best || score > best.score) best = {slot, score};
  }
  if (best) {
    usedSlots.add(best.slot);
    return best.slot;
  }
  for (let slot = 1; slot <= entries.length; slot++) {
    if (!usedSlots.has(slot)) {
      usedSlots.add(slot);
      return slot;
    }
  }
  throw new Error('No unused slots left while building lead mode');
}

function makeLeadMode({id, entries, roles}) {
  const usedSlots = new Set();
  const slots = roles.map(role => pickSlot(entries, usedSlots, role));
  const species = slots.map(slot => entries[slot - 1].species);
  return {
    id,
    team_spec: slots.join(''),
    leads: species.slice(0, 2),
    back: species.slice(2, 4),
  };
}

function buildLeadModes(entries) {
  const modes = [
    makeLeadMode({id: 'tempo_pressure', entries, roles: ['speed', 'offense', 'fakeout', 'support']}),
    makeLeadMode({id: 'fakeout_setup', entries, roles: ['fakeout', 'setup', 'speed', 'offense']}),
    makeLeadMode({id: 'balanced_four', entries, roles: ['support', 'offense', 'speed', 'fakeout']}),
  ];
  const seen = new Set();
  return modes.map((mode, index) => {
    if (!seen.has(mode.team_spec)) {
      seen.add(mode.team_spec);
      return mode;
    }
    const fallbackSpec = ['1234', '1256', '3456'][index] || '1234';
    const fallbackSlots = fallbackSpec.split('').map(Number);
    return {
      id: mode.id,
      team_spec: fallbackSpec,
      leads: fallbackSlots.slice(0, 2).map(slot => entries[slot - 1].species),
      back: fallbackSlots.slice(2, 4).map(slot => entries[slot - 1].species),
    };
  });
}

function loadSetCatalog({pool = null, poolPath = null} = {}) {
  const loadedPool = pool || loadPool(poolPath || path.join(repoRoot, 'data', 'teams', 'team_pool.json'));
  const dex = dexForFormat(loadedPool.format_id);
  const seen = new Set();
  const sets = [];

  for (const sourceTeam of loadedPool.teams || []) {
    const importPath = resolveRepoPath(sourceTeam.import_file);
    const parsed = Teams.import(fs.readFileSync(importPath, 'utf8')) || [];
    parsed.forEach((set, index) => {
      const canonical = Teams.pack([set]);
      if (seen.has(canonical)) return;
      seen.add(canonical);
      const species = speciesInfo(dex, set.species);
      const item = itemInfo(dex, set.item);
      const roles = classifySet({set, dex});
      sets.push({
        set_id: `${species.id}_${hashString(canonical)}`,
        source_team_id: sourceTeam.id,
        source_team_name: sourceTeam.name,
        source_slot: index + 1,
        species: species.name,
        species_id: species.id,
        item: item.name,
        item_id: item.id,
        ability: set.ability || '',
        moves: (set.moves || []).slice(),
        mega_species: item.megaSpecies,
        roles,
        set: clone(set),
      });
    });
  }

  return {
    format_id: loadedPool.format_id,
    format_name: loadedPool.format_name,
    source_team_count: loadedPool.teams?.length || 0,
    set_count: sets.length,
    sets,
  };
}

function shuffle(items, rng) {
  const shuffled = items.slice();
  for (let index = shuffled.length - 1; index > 0; index--) {
    const swapIndex = Math.floor(rng.next() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }
  return shuffled;
}

function canAddEntry(entry, selected, options) {
  if (selected.some(other => other.species_id === entry.species_id)) return false;
  if (options.uniqueItems && entry.item_id && selected.some(other => other.item_id === entry.item_id)) {
    return false;
  }
  return true;
}

function validateTeamSets({sets, formatId}) {
  if (!Array.isArray(sets) || sets.length !== 6) return ['team must contain exactly six sets'];
  const problems = TeamValidator.get(formatId).validateTeam(sets);
  return problems || [];
}

function sampleCandidateEntries({catalog, rng, options = {}}) {
  const mergedOptions = {...DEFAULT_GENERATOR_OPTIONS, ...options};
  const selected = [];
  for (const entry of shuffle(catalog.sets, rng)) {
    if (!canAddEntry(entry, selected, mergedOptions)) continue;
    selected.push(entry);
    if (selected.length === 6) break;
  }
  if (selected.length !== 6) return null;

  const megaCount = selected.filter(entry => entry.roles.mega).length;
  if (megaCount < mergedOptions.minMegas || megaCount > mergedOptions.maxMegas) return null;

  const problems = validateTeamSets({
    sets: selected.map(entry => clone(entry.set)),
    formatId: catalog.format_id,
  });
  return problems.length ? null : selected;
}

function candidateFromEntries({entries, id, name, formatId}) {
  const sets = entries.map(entry => clone(entry.set));
  const hash = teamHash(sets);
  return {
    id,
    name,
    format_id: formatId,
    hash,
    entries: entries.slice(),
    sets,
    species: entries.map(entry => entry.species),
    items: entries.map(entry => entry.item || ''),
    source_set_ids: entries.map(entry => entry.set_id),
    source_team_ids: [...new Set(entries.map(entry => entry.source_team_id))],
    primary_megas: entries
      .filter(entry => entry.roles.mega)
      .map(entry => entry.mega_species || `${entry.species}-Mega`),
    lead_modes: buildLeadModes(entries),
    validation: 'passed',
  };
}

function generateRandomCandidates({catalog, rng, count, idPrefix = 'phase7-cand', options = {}}) {
  const mergedOptions = {...DEFAULT_GENERATOR_OPTIONS, ...options};
  const candidates = [];
  const seenHashes = new Set();
  let attempts = 0;

  while (candidates.length < count && attempts < mergedOptions.maxAttempts) {
    attempts += 1;
    const entries = sampleCandidateEntries({catalog, rng, options: mergedOptions});
    if (!entries) continue;
    const candidate = candidateFromEntries({
      entries,
      id: `${idPrefix}-${String(candidates.length + 1).padStart(3, '0')}`,
      name: `Phase7 Candidate ${String(candidates.length + 1).padStart(3, '0')}`,
      formatId: catalog.format_id,
    });
    if (seenHashes.has(candidate.hash)) continue;
    seenHashes.add(candidate.hash);
    candidates.push(candidate);
  }

  if (candidates.length < count) {
    throw new Error(`Generated ${candidates.length}/${count} candidates after ${attempts} attempts`);
  }
  return {candidates, attempts};
}

function fillCandidateEntries({entries, catalog, rng, options = {}}) {
  const mergedOptions = {...DEFAULT_GENERATOR_OPTIONS, ...options};
  const selected = [];
  for (const entry of shuffle(entries, rng)) {
    if (selected.length >= 6) break;
    if (canAddEntry(entry, selected, mergedOptions)) selected.push(entry);
  }
  for (const entry of shuffle(catalog.sets, rng)) {
    if (selected.length >= 6) break;
    if (canAddEntry(entry, selected, mergedOptions)) selected.push(entry);
  }
  if (selected.length !== 6) return null;
  const megaCount = selected.filter(entry => entry.roles.mega).length;
  if (megaCount < mergedOptions.minMegas || megaCount > mergedOptions.maxMegas) return null;
  const problems = validateTeamSets({
    sets: selected.map(entry => clone(entry.set)),
    formatId: catalog.format_id,
  });
  return problems.length ? null : selected;
}

function mutateEntries({parentEntries, catalog, rng, options = {}}) {
  const indexToReplace = Math.floor(rng.next() * parentEntries.length);
  const base = parentEntries.filter((_, index) => index !== indexToReplace);
  return fillCandidateEntries({entries: base, catalog, rng, options});
}

function crossoverEntries({leftEntries, rightEntries, catalog, rng, options = {}}) {
  const mixed = [];
  for (let index = 0; index < 6; index++) {
    mixed.push((rng.next() < 0.5 ? leftEntries : rightEntries)[index]);
  }
  return fillCandidateEntries({entries: mixed, catalog, rng, options});
}

function teamRecordForCandidate({candidate, importFile}) {
  return {
    id: candidate.id,
    name: candidate.name,
    import_file: importFile,
    hash: candidate.hash,
    source_set_ids: candidate.source_set_ids,
    source_team_ids: candidate.source_team_ids,
    species: candidate.species,
    items: candidate.items,
    primary_megas: candidate.primary_megas,
    validation: 'passed',
    lead_modes: candidate.lead_modes,
  };
}

function writeCandidatePool({
  sourcePool,
  sourcePoolPath,
  candidates,
  outDir,
  seed,
  generator,
  overwrite = false,
}) {
  const resolvedOutDir = resolveRepoPath(outDir);
  const importsDir = path.join(resolvedOutDir, 'imports');
  const poolPath = path.join(resolvedOutDir, 'candidates.json');
  if (!overwrite && fs.existsSync(poolPath)) {
    throw new Error(`Candidate pool exists at ${relativePath(poolPath)}; pass --overwrite to replace it`);
  }
  fs.mkdirSync(importsDir, {recursive: true});

  const teams = candidates.map(candidate => {
    const importPath = path.join(importsDir, `${candidate.id}.txt`);
    fs.writeFileSync(importPath, teamImportText(candidate.sets), 'utf8');
    const importFile = relativePath(importPath);
    candidate.import_file = importFile;
    return teamRecordForCandidate({candidate, importFile});
  });

  const pool = {
    format_id: sourcePool.format_id,
    format_name: sourcePool.format_name,
    team_preview_policy: 'generated_role_lead_modes',
    source_pool: sourcePoolPath ? relativePath(resolveRepoPath(sourcePoolPath)) : null,
    created_at: new Date().toISOString(),
    seed,
    generator,
    teams,
  };
  fs.writeFileSync(poolPath, `${JSON.stringify(pool, null, 2)}\n`, 'utf8');
  return {pool, poolPath};
}

function validateLeadModes(team) {
  const problems = [];
  for (const mode of team.lead_modes || []) {
    if (!/^[1-6]{4}$/.test(mode.team_spec)) {
      problems.push(`${mode.id}: team_spec must be four digits between 1 and 6`);
      continue;
    }
    if (new Set(mode.team_spec).size !== 4) {
      problems.push(`${mode.id}: team_spec must contain four unique slots`);
    }
  }
  return problems;
}

function validatePoolTeam({pool, team}) {
  const importPath = resolveRepoPath(team.import_file);
  const parsedTeam = Teams.import(fs.readFileSync(importPath, 'utf8'));
  const validationProblems = TeamValidator.get(pool.format_id).validateTeam(parsedTeam) || [];
  return validationProblems.concat(validateLeadModes(team));
}

module.exports = {
  DEFAULT_GENERATOR_OPTIONS,
  candidateFromEntries,
  cleanImportText,
  clone,
  crossoverEntries,
  generateRandomCandidates,
  hashString,
  loadPool,
  loadSetCatalog,
  mutateEntries,
  relativePath,
  resolveRepoPath,
  sampleCandidateEntries,
  teamHash,
  teamImportText,
  toId,
  validatePoolTeam,
  validateTeamSets,
  writeCandidatePool,
};
