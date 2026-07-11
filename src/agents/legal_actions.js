const {
  canActiveSlotAct,
  canSwitchActive,
  enabledMoveSlots,
  foeActiveTargets,
  isFaintedActiveSlot,
  isMoveRequest,
  isTeamPreviewRequest,
  makeMoveChoice,
  moveTargetChoices,
  ownActiveTargets,
  switchableSlots,
} = require('./action_utils');
const {enumerateCanonicalTeamPreviewActions} = require('../team_preview/preview_actions');

function moveChoicesForActive({activeData, activeIndex, side, request, battleState, dex, megaUsed}) {
  if (!canActiveSlotAct(request, activeIndex, activeData)) {
    return [{choice: 'pass', decisions: [{kind: 'pass', activeIndex}]}];
  }

  const foeActives = foeActiveTargets({side, battleState, dex});
  const ownActives = ownActiveTargets(request);
  const choices = [];

  for (const {moveSlot, moveIndex} of enabledMoveSlots(activeData)) {
    const move = dex.moves.get(moveSlot.id || moveSlot.move);
    if (!move.exists) {
      choices.push({
        choice: makeMoveChoice({moveIndex}),
        decisions: [{
          kind: 'move',
          activeIndex,
          moveIndex,
          moveId: moveSlot.id || moveSlot.move,
          target: null,
          mega: false,
        }],
      });
      continue;
    }

    const targetChoices = Object.prototype.hasOwnProperty.call(moveSlot, 'target')
      ? moveTargetChoices({move: {...move, target: moveSlot.target || move.target}, activeIndex, foeActives, ownActives})
      : [null];
    for (const target of targetChoices) {
      choices.push({
        choice: makeMoveChoice({moveIndex, target}),
        decisions: [{
          kind: 'move',
          activeIndex,
          moveIndex,
          moveId: move.id,
          target,
          mega: false,
        }],
      });

      if (!megaUsed && activeData.canMegaEvo) {
        choices.push({
          choice: makeMoveChoice({moveIndex, target, mega: true}),
          decisions: [{
            kind: 'move',
            activeIndex,
            moveIndex,
            moveId: move.id,
            target,
            mega: true,
          }],
        });
      }
    }
  }

  return choices.length ? choices : [{choice: 'pass', decisions: [{kind: 'pass', activeIndex}]}];
}

function switchChoicesForActive({activeData, activeIndex, request, reservedSlots}) {
  if (!canActiveSlotAct(request, activeIndex, activeData) || !canSwitchActive(activeData, request, reservedSlots)) {
    return [];
  }
  return switchableSlots(request, reservedSlots).map(slot => ({
    choice: `switch ${slot}`,
    decisions: [{kind: 'switch', activeIndex, slot}],
    reservesSwitchSlot: slot,
  }));
}

function enumerateMoveActions({side, request, battleState, dex}) {
  const actions = [];

  function walk(activeIndex, parts, decisions, reservedSlots, megaUsed) {
    if (activeIndex >= request.active.length) {
      actions.push({choice: parts.join(', '), decisions});
      return;
    }

    if (isFaintedActiveSlot(request, activeIndex)) {
      walk(activeIndex + 1, parts, decisions, reservedSlots, megaUsed);
      return;
    }

    const activeData = request.active[activeIndex];
    if (!activeData) {
      walk(activeIndex + 1, [...parts, 'pass'], [...decisions, {kind: 'pass', activeIndex}], reservedSlots, megaUsed);
      return;
    }

    const activeChoices = [
      ...moveChoicesForActive({activeData, activeIndex, side, request, battleState, dex, megaUsed}),
      ...switchChoicesForActive({activeData, activeIndex, request, reservedSlots}),
    ];

    for (const activeChoice of activeChoices) {
      const nextReservedSlots = new Set(reservedSlots);
      if (activeChoice.reservesSwitchSlot) nextReservedSlots.add(activeChoice.reservesSwitchSlot);
      const nextMegaUsed = megaUsed || activeChoice.decisions.some(decision => decision.mega);
      walk(
        activeIndex + 1,
        [...parts, activeChoice.choice],
        decisions.concat(activeChoice.decisions),
        nextReservedSlots,
        nextMegaUsed
      );
    }
  }

  walk(0, [], [], new Set(), false);
  return actions;
}

function enumerateForceSwitchActions(request) {
  const actions = [];

  function walk(index, parts, decisions, reservedSlots) {
    if (index >= request.forceSwitch.length) {
      actions.push({choice: parts.join(', '), decisions});
      return;
    }

    if (!request.forceSwitch[index]) {
      walk(index + 1, [...parts, 'pass'], [...decisions, {kind: 'pass', activeIndex: index}], reservedSlots);
      return;
    }

    const slots = switchableSlots(request, reservedSlots);
    if (!slots.length) {
      walk(index + 1, [...parts, 'pass'], [...decisions, {kind: 'pass', activeIndex: index}], reservedSlots);
      return;
    }

    for (const slot of slots) {
      const nextReservedSlots = new Set(reservedSlots);
      nextReservedSlots.add(slot);
      walk(
        index + 1,
        [...parts, `switch ${slot}`],
        [...decisions, {kind: 'switch', activeIndex: index, slot}],
        nextReservedSlots
      );
    }
  }

  walk(0, [], [], new Set());
  return actions;
}

function enumerateLegalActions({side, request, battleState, dex}) {
  if (request.wait) return [];
  if (isTeamPreviewRequest(request)) return enumerateCanonicalTeamPreviewActions(request);
  if (request.forceSwitch?.some(Boolean)) return enumerateForceSwitchActions(request);
  if (isMoveRequest(request)) return enumerateMoveActions({side, request, battleState, dex});
  return [{choice: 'default', decisions: [{kind: 'default'}]}];
}

function enumerateLegalActionChoices(args) {
  return enumerateLegalActions(args).map(action => action.choice);
}

module.exports = {
  enumerateLegalActionChoices,
  enumerateLegalActions,
};
