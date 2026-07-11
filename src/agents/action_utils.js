const {canonicalTeamPreviewChoice} = require('../team_preview/preview_actions');

function isFainted(pokemon) {
  return /\bfnt\b/.test(pokemon?.condition || '');
}

function switchableSlots(request, reservedSlots = new Set()) {
  if (!request.side?.pokemon) return [];
  const slots = [];
  request.side.pokemon.forEach((pokemon, index) => {
    const slot = index + 1;
    if (!pokemon.active && !pokemon.reviving && !isFainted(pokemon) && !reservedSlots.has(slot)) {
      slots.push(slot);
    }
  });
  return slots;
}

function canSwitchActive(activeData, request, reservedSlots = new Set()) {
  return !activeData?.trapped &&
    !activeData?.maybeTrapped &&
    switchableSlots(request, reservedSlots).length > 0;
}

function chooseSwitch(request, rng, reservedSlots = new Set()) {
  const slots = switchableSlots(request, reservedSlots);
  if (!slots.length) return 'pass';
  const slot = rng.pick(slots);
  reservedSlots.add(slot);
  return `switch ${slot}`;
}

function chooseForcedSwitches(request, rng, reservedSlots = new Set()) {
  return request.forceSwitch
    .map(needed => needed ? chooseSwitch(request, rng, reservedSlots) : 'pass')
    .join(', ');
}

function teamPreviewChoice(battleState) {
  return canonicalTeamPreviewChoice(`team ${battleState.leadMode.team_spec}`);
}

function isTeamPreviewRequest(request) {
  return request.requestType === 'teampreview' || !!request.teamPreview;
}

function isMoveRequest(request) {
  return request.requestType === 'move' || !!request.active;
}

function enabledMoveSlots(activeData) {
  return (activeData.moves || [])
    .map((moveSlot, index) => ({moveSlot, moveIndex: index + 1}))
    .filter(entry => !entry.moveSlot.disabled);
}

function megaEvolutionCandidates(request) {
  const candidates = [];
  request.active?.forEach((activeData, index) => {
    if (activeData?.canMegaEvo) candidates.push(index);
  });
  return candidates;
}

function chooseMegaSlot(request, rng, probability = 1) {
  const candidates = megaEvolutionCandidates(request);
  if (!candidates.length || rng.next() >= probability) return -1;
  return rng.pick(candidates);
}

function makeMoveChoice({moveIndex, target = null, mega = false}) {
  const targetText = target ? ` ${target}` : '';
  const megaText = mega ? ' mega' : '';
  return `move ${moveIndex}${targetText}${megaText}`;
}

function sideOpponent(side) {
  return side === 'p1' ? 'p2' : 'p1';
}

function speciesNameFromDetails(details) {
  if (!details) return null;
  return details.split(',')[0].trim();
}

function ownActivePokemon(request) {
  return (request.side?.pokemon || []).filter(pokemon => pokemon.active);
}

function ownActivePokemonAt(request, activeIndex) {
  return ownActivePokemon(request)[activeIndex] || null;
}

function isFaintedActiveSlot(request, activeIndex) {
  const pokemon = ownActivePokemonAt(request, activeIndex);
  return !!pokemon && isFainted(pokemon);
}

function canActiveSlotAct(request, activeIndex, activeData = null) {
  const pokemon = ownActivePokemonAt(request, activeIndex);
  if (!pokemon || pokemon.commanding || isFainted(pokemon)) return false;
  if (activeData?.moves?.length) return true;
  return true;
}

function ownActiveTargets(request) {
  return ownActivePokemon(request).map((pokemon, slot) => ({slot, pokemon}));
}

function ownActiveSpeciesName(request, activeIndex) {
  return speciesNameFromDetails(ownActivePokemonAt(request, activeIndex)?.details);
}

function foeActiveTargets({side, battleState, dex}) {
  return (battleState.publicState?.active?.[sideOpponent(side)] || [])
    .map((active, slot) => ({
      slot,
      species: active ? dex.species.get(active.species) : null,
      condition: active?.condition || null,
      fainted: !!active?.fainted,
    }))
    .filter(active => active.species?.exists && !active.fainted);
}

function targetGroupsForMove(move, foeActives) {
  if (['allAdjacentFoes', 'allAdjacent', 'all'].includes(move.target)) {
    return [{target: null, targets: foeActives, spread: true}];
  }
  if (['normal', 'any', 'adjacentFoe', 'randomNormal'].includes(move.target)) {
    return foeActives.map(target => ({target: foeTargetLoc(target), targets: [target], spread: false}));
  }
  return [{target: null, targets: [], spread: false}];
}

function moveTargetChoices({move, activeIndex, foeActives, ownActives}) {
  if (['normal', 'any', 'adjacentFoe'].includes(move.target)) {
    return foeActives.map(target => foeTargetLoc(target));
  }
  if (move.target === 'adjacentAlly') {
    return ownActives
      .filter(target => target.slot !== activeIndex)
      .map(target => allyTargetLoc(target));
  }
  if (move.target === 'adjacentAllyOrSelf') {
    return ownActives.map(target => allyTargetLoc(target));
  }
  return [null];
}

function foeTargetLoc(target) {
  return String(target.slot + 1);
}

function allyTargetLoc(target) {
  return `-${target.slot + 1}`;
}

function basePowerForMove(move) {
  if (move.category === 'Status') return 0;
  if (typeof move.damage === 'number') return move.damage;
  if (move.damage === 'level') return 50;
  if (!move.basePower) return 0;
  if (Array.isArray(move.multihit)) return move.basePower * 3;
  return move.basePower;
}

function typeEffectivenessMultiplier(dex, move, targetSpecies) {
  if (!targetSpecies?.exists || !move.type) return 1;
  if (!dex.getImmunity(move.type, targetSpecies.types)) return 0;
  return Math.pow(2, dex.getEffectiveness(move.type, targetSpecies.types));
}

function moveDamageScore({dex, move, userSpecies, target, spread}) {
  const basePower = basePowerForMove(move);
  if (!basePower) return 0;

  const effectiveness = typeEffectivenessMultiplier(dex, move, target.species);
  const stab = userSpecies?.types?.includes(move.type) ? 1.5 : 1;
  const spreadPenalty = spread ? 0.75 : 1;
  return basePower * effectiveness * stab * spreadPenalty;
}

module.exports = {
  basePowerForMove,
  canActiveSlotAct,
  canSwitchActive,
  chooseForcedSwitches,
  chooseMegaSlot,
  chooseSwitch,
  enabledMoveSlots,
  foeActiveTargets,
  foeTargetLoc,
  isFainted,
  isFaintedActiveSlot,
  isMoveRequest,
  isTeamPreviewRequest,
  makeMoveChoice,
  moveTargetChoices,
  moveDamageScore,
  ownActiveTargets,
  ownActiveSpeciesName,
  sideOpponent,
  speciesNameFromDetails,
  switchableSlots,
  targetGroupsForMove,
  teamPreviewChoice,
  typeEffectivenessMultiplier,
};
