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
    active: {p1: [null, null], p2: [null, null]},
    revealed: {p1: [], p2: []},
    fainted: {p1: [], p2: []},
  };
}

function addUnique(list, value) {
  if (value && !list.includes(value)) list.push(value);
}

function rememberSpecies(publicState, side, species) {
  if (!publicState.revealed) publicState.revealed = {p1: [], p2: []};
  if (!publicState.revealed[side]) publicState.revealed[side] = [];
  addUnique(publicState.revealed[side], species);
}

function rememberFainted(publicState, side, species) {
  if (!publicState.fainted) publicState.fainted = {p1: [], p2: []};
  if (!publicState.fainted[side]) publicState.fainted[side] = [];
  addUnique(publicState.fainted[side], species);
}

function replaceRememberedSpecies(publicState, side, oldSpecies, newSpecies) {
  for (const key of ['revealed', 'fainted']) {
    const list = publicState[key]?.[side];
    if (!list) continue;
    const index = list.indexOf(oldSpecies);
    if (index >= 0) list[index] = newSpecies;
  }
  rememberSpecies(publicState, side, newSpecies);
}

function applySpectatorLineToState(publicState, line) {
  const parts = line.split('|');
  const type = parts[1];
  if (!type) return;

  if (['switch', 'drag', 'replace'].includes(type)) {
    const parsed = sideFromIdent(parts[2]);
    if (!parsed) return;
    const species = speciesFromDetails(parts[3]);
    publicState.active[parsed.side][parsed.slot] = {
      ident: parts[2],
      species,
      condition: parts[4] || null,
      fainted: false,
    };
    rememberSpecies(publicState, parsed.side, species);
    return;
  }

  if (['detailschange', '-formechange'].includes(type)) {
    const parsed = sideFromIdent(parts[2]);
    if (!parsed || !publicState.active[parsed.side][parsed.slot]) return;
    const species = speciesFromDetails(parts[3]);
    const oldSpecies = publicState.active[parsed.side][parsed.slot].species;
    publicState.active[parsed.side][parsed.slot].species = species;
    publicState.active[parsed.side][parsed.slot].fainted = false;
    replaceRememberedSpecies(publicState, parsed.side, oldSpecies, species);
    return;
  }

  if (type === 'faint') {
    const parsed = sideFromIdent(parts[2]);
    if (!parsed || !publicState.active[parsed.side][parsed.slot]) return;
    const active = publicState.active[parsed.side][parsed.slot];
    active.fainted = true;
    active.condition = '0 fnt';
    rememberFainted(publicState, parsed.side, active.species);
    return;
  }

  if (['-damage', '-heal', '-sethp'].includes(type)) {
    const parsed = sideFromIdent(parts[2]);
    if (!parsed || !publicState.active[parsed.side][parsed.slot]) return;
    const active = publicState.active[parsed.side][parsed.slot];
    const condition = parts[3] || null;
    active.condition = condition;
    if (condition && /\bfnt\b/.test(condition)) {
      active.fainted = true;
      rememberFainted(publicState, parsed.side, active.species);
    }
  }
}

module.exports = {
  applySpectatorLineToState,
  createPublicState,
  sideFromIdent,
  speciesFromDetails,
};
