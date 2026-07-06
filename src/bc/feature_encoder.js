const {isFainted, isTeamPreviewRequest, speciesNameFromDetails, switchableSlots} = require('../agents/action_utils');

const ENCODER_VERSION = 1;

function toId(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function hashString(value) {
  const text = String(value);
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function hashFeature(feature, featureDim) {
  if (!Number.isInteger(featureDim) || featureDim <= 0) {
    throw new Error(`featureDim must be a positive integer, got ${featureDim}`);
  }
  return hashString(feature) % featureDim;
}

function requestTypeFromRequest(request) {
  if (!request || typeof request !== 'object') return 'other';
  if (isTeamPreviewRequest(request)) return 'team_preview';
  if (Array.isArray(request.forceSwitch) && request.forceSwitch.some(Boolean)) return 'force_switch';
  if (Array.isArray(request.active) && request.active.length) return 'move';
  if (request.wait) return 'wait';
  return 'other';
}

function sideOpponent(side) {
  return side === 'p1' ? 'p2' : 'p1';
}

function activePokemon(request) {
  return (request.side?.pokemon || []).filter(pokemon => pokemon.active);
}

function actionableActiveIndices(request) {
  const ownActive = activePokemon(request);
  return (request.active || [])
    .map((_, activeIndex) => ({activeIndex, pokemon: ownActive[activeIndex]}))
    .filter(entry => !entry.pokemon || !isFainted(entry.pokemon))
    .map(entry => entry.activeIndex);
}

function hpBucket(condition) {
  if (!condition || /\bfnt\b/.test(condition)) return 'fnt';
  const match = String(condition).match(/^(\d+)\/(\d+)/);
  if (!match) return 'unknown';
  const current = Number(match[1]);
  const max = Number(match[2]);
  if (!max) return 'unknown';
  const ratio = current / max;
  if (ratio <= 0.25) return '0_25';
  if (ratio <= 0.5) return '25_50';
  if (ratio <= 0.75) return '50_75';
  return '75_100';
}

function turnBucket(turn) {
  const numericTurn = Number(turn);
  if (!Number.isFinite(numericTurn) || numericTurn <= 0) return '0';
  if (numericTurn <= 2) return '1_2';
  if (numericTurn <= 5) return '3_5';
  if (numericTurn <= 10) return '6_10';
  return '11_plus';
}

function addFeature(features, key, value = 1, interact = true) {
  if (value === 0 || key == null || key === '') return;
  features.push({key: String(key), value, interact});
}

function stableDigest(value) {
  if (value == null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableDigest).join(',')}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map(key => `${JSON.stringify(key)}:${stableDigest(value[key])}`).join(',')}}`;
}

function stateSignature(context) {
  return hashString(stableDigest({
    side: context.side,
    requestType: context.requestType || requestTypeFromRequest(context.request),
    request: context.request,
    publicState: context.publicState,
  })).toString(36);
}

function stateFeaturesFromContext(context, options = {}) {
  const request = context.request || {};
  const publicState = context.publicState || {};
  const requestType = context.requestType || requestTypeFromRequest(request);
  const side = context.side || request.side?.id || 'unknown';
  const features = [];

  addFeature(features, `request_type:${requestType}`);
  addFeature(features, `side:${side}`);
  addFeature(features, `team:${context.team || 'unknown'}`);
  addFeature(features, `lead:${context.lead || 'unknown'}`);
  addFeature(features, `turn_bucket:${turnBucket(context.turn)}`);

  if (options.includeStateSignature !== false) {
    addFeature(features, `state_sig:${stateSignature({...context, requestType})}`);
  }

  const ownPokemon = request.side?.pokemon || [];
  const ownActive = activePokemon(request);
  addFeature(features, `own_team_size:${ownPokemon.length}`);
  addFeature(features, `own_active_count:${ownActive.length}`);
  addFeature(features, `own_alive_count:${ownPokemon.filter(pokemon => !isFainted(pokemon)).length}`);

  ownPokemon.forEach((pokemon, index) => {
    const species = toId(speciesNameFromDetails(pokemon.details) || pokemon.ident);
    if (!species) return;
    addFeature(features, `own_species:${species}`, 1, false);
    addFeature(features, `own_slot${index + 1}:species:${species}`);
    addFeature(features, `own_slot${index + 1}:hp:${hpBucket(pokemon.condition)}`);
    if (pokemon.active) addFeature(features, `own_active_species:${species}`);
    if (!pokemon.active && !isFainted(pokemon)) addFeature(features, `own_bench_species:${species}`);
    for (const moveId of pokemon.moves || []) {
      addFeature(features, `own_known_move:${toId(moveId)}`, 1, false);
    }
  });

  (request.active || []).forEach((activeData, activeIndex) => {
    if (activeData?.canMegaEvo) addFeature(features, `active${activeIndex}:can_mega`);
    (activeData?.moves || []).forEach((move, moveIndex) => {
      const moveId = toId(move.id || move.move);
      if (!moveId) return;
      addFeature(features, `available_move:${moveId}`);
      addFeature(features, `active${activeIndex}:move${moveIndex + 1}:${moveId}`);
      addFeature(features, `active${activeIndex}:move${moveIndex + 1}:target:${move.target || 'unknown'}`);
      if (move.disabled) addFeature(features, `active${activeIndex}:move${moveIndex + 1}:disabled`);
    });
  });

  if (Array.isArray(request.forceSwitch)) {
    addFeature(features, `force_switch:${request.forceSwitch.map(Boolean).join('_')}`);
  }

  if (request.teamPreview) {
    addFeature(features, `team_preview_size:${request.maxChosenTeamSize || ownPokemon.length}`);
  }

  const ownSide = side;
  const foeSide = sideOpponent(side);
  for (const [publicSide, actives] of Object.entries(publicState.active || {})) {
    (actives || []).forEach((active, slot) => {
      const species = toId(active?.species);
      if (!species) return;
      const relation = publicSide === ownSide ? 'public_own' : (publicSide === foeSide ? 'public_foe' : `public_${publicSide}`);
      addFeature(features, `${relation}_slot${slot}:species:${species}`);
      addFeature(features, `${relation}_species:${species}`);
      if (active.fainted) addFeature(features, `${relation}_slot${slot}:fainted`);
    });
  }

  return features;
}

function parseCommand(command) {
  const parts = command.trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return {kind: 'empty'};
  const kind = parts[0];
  if (kind === 'move') {
    const moveIndex = Number(parts[1]);
    const target = parts.slice(2).find(part => /^-?\d+$/.test(part)) || null;
    return {
      kind,
      moveIndex: Number.isInteger(moveIndex) ? moveIndex : null,
      target,
      mega: parts.includes('mega'),
    };
  }
  if (kind === 'switch') {
    const slot = Number(parts[1]);
    return {kind, slot: Number.isInteger(slot) ? slot : null};
  }
  if (kind === 'team') return {kind, spec: parts[1] || ''};
  if (kind === 'pass') return {kind};
  return {kind: 'other', raw: command.trim()};
}

function moveIdForAction(request, activeIndex, moveIndex) {
  if (!Number.isInteger(moveIndex) || moveIndex < 1) return null;
  const move = request.active?.[activeIndex]?.moves?.[moveIndex - 1];
  return toId(move?.id || move?.move);
}

function switchSpeciesForAction(request, slot) {
  if (!Number.isInteger(slot) || slot < 1) return null;
  return toId(speciesNameFromDetails(request.side?.pokemon?.[slot - 1]?.details));
}

function actionFeaturesForChoice({choice, request}) {
  const text = String(choice || '').trim().toLowerCase().replace(/\s+/g, ' ');
  const features = [];
  addFeature(features, 'bias', 1, false);
  addFeature(features, `choice:${text}`);

  if (text.startsWith('team ')) {
    const spec = text.slice('team '.length).trim();
    addFeature(features, 'kind:team');
    addFeature(features, `team_spec:${spec}`);
    spec.split('').forEach((slot, index) => {
      addFeature(features, `team_pick_pos${index + 1}:${slot}`);
      if (index < 2) addFeature(features, `team_lead_pos${index + 1}:${slot}`);
    });
    if (spec.length >= 2) addFeature(features, `team_lead_pair:${spec.slice(0, 2)}`);
    return features;
  }

  const commands = text.split(',').map(command => command.trim()).filter(Boolean);
  const activeIndices = actionableActiveIndices(request || {});
  const kinds = [];
  commands.forEach((commandText, commandIndex) => {
    const activeIndex = activeIndices[commandIndex] ?? commandIndex;
    const command = parseCommand(commandText);
    kinds.push(command.kind);
    addFeature(features, `slot${activeIndex}:kind:${command.kind}`);
    addFeature(features, `kind:${command.kind}`);

    if (command.kind === 'move') {
      addFeature(features, `move_index:${command.moveIndex}`);
      addFeature(features, `slot${activeIndex}:move_index:${command.moveIndex}`);
      const moveId = moveIdForAction(request || {}, activeIndex, command.moveIndex);
      if (moveId) {
        addFeature(features, `move_id:${moveId}`);
        addFeature(features, `slot${activeIndex}:move_id:${moveId}`);
      }
      if (command.target) {
        addFeature(features, `target:${command.target}`);
        addFeature(features, `slot${activeIndex}:target:${command.target}`);
      }
      if (command.mega) {
        addFeature(features, 'mega');
        addFeature(features, `slot${activeIndex}:mega`);
      }
    } else if (command.kind === 'switch') {
      addFeature(features, `switch_slot:${command.slot}`);
      addFeature(features, `slot${activeIndex}:switch_slot:${command.slot}`);
      const species = switchSpeciesForAction(request || {}, command.slot);
      if (species) {
        addFeature(features, `switch_species:${species}`);
        addFeature(features, `slot${activeIndex}:switch_species:${species}`);
      }
    }
  });

  if (kinds.length) addFeature(features, `command_kinds:${kinds.join('+')}`);
  return features;
}

function featureEntriesForStateAction({stateFeatures, actionFeatures, featureDim}) {
  const entries = new Map();

  function addHashed(feature, value) {
    const index = hashFeature(feature, featureDim);
    entries.set(index, (entries.get(index) || 0) + value);
  }

  for (const actionFeature of actionFeatures) {
    addHashed(`a:${actionFeature.key}`, actionFeature.value);
  }

  const interactingState = stateFeatures.filter(feature => feature.interact !== false);
  const interactingAction = actionFeatures.filter(feature => feature.interact !== false);
  for (const stateFeature of interactingState) {
    for (const actionFeature of interactingAction) {
      addHashed(`x:${stateFeature.key}|${actionFeature.key}`, stateFeature.value * actionFeature.value);
    }
  }

  const norm = Math.sqrt([...entries.values()].reduce((sum, value) => sum + value * value, 0)) || 1;
  return [...entries.entries()].map(([index, value]) => ({index, value: value / norm}));
}

function contextFromExample(example) {
  return {
    side: example.side,
    turn: example.turn,
    team: example.team,
    lead: example.lead,
    requestType: example.request_type,
    request: example.state?.request,
    publicState: example.state?.public_state,
  };
}

function contextFromBattle({side, request, battleState}) {
  return {
    side,
    turn: battleState?.turns || 0,
    team: battleState?.team?.id,
    lead: battleState?.leadMode?.id,
    requestType: requestTypeFromRequest(request),
    request,
    publicState: battleState?.publicState,
  };
}

function legalChoiceText(action) {
  return typeof action === 'string' ? action : action.choice;
}

function featurizeChoice({context, choice, featureDim, encoderOptions = {}, stateFeatures = null}) {
  const resolvedStateFeatures = stateFeatures || stateFeaturesFromContext(context, encoderOptions);
  const actionFeatures = actionFeaturesForChoice({choice, request: context.request});
  return featureEntriesForStateAction({
    stateFeatures: resolvedStateFeatures,
    actionFeatures,
    featureDim,
  });
}

function switchableSpeciesFeatures(request) {
  return switchableSlots(request).map(slot => switchSpeciesForAction(request, slot)).filter(Boolean);
}

module.exports = {
  ENCODER_VERSION,
  actionFeaturesForChoice,
  contextFromBattle,
  contextFromExample,
  featureEntriesForStateAction,
  featurizeChoice,
  hashFeature,
  hashString,
  legalChoiceText,
  requestTypeFromRequest,
  stateFeaturesFromContext,
  switchableSpeciesFeatures,
  toId,
};
