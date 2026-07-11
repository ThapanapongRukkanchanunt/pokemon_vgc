function combinations(items, size) {
  if (size === 0) return [[]];
  if (size < 0 || size > items.length) return [];
  const selections = [];
  for (let index = 0; index <= items.length - size; index++) {
    for (const rest of combinations(items.slice(index + 1), size - 1)) {
      selections.push([items[index], ...rest]);
    }
  }
  return selections;
}

function canonicalTeamPreviewChoice(choice) {
  const match = String(choice || '').trim().match(/^team\s+(\d+)$/i);
  if (!match) return choice;
  const slots = [...match[1]];
  if (slots.length < 2) return `team ${slots.sort().join('')}`;
  const leads = slots.slice(0, 2).sort();
  const backs = slots.slice(2).sort();
  return `team ${leads.concat(backs).join('')}`;
}

function enumerateCanonicalTeamPreviewActions(request) {
  const teamSize = request.maxChosenTeamSize || request.side?.pokemon?.length || 0;
  const slots = (request.side?.pokemon || []).map((_, index) => index + 1);
  if (teamSize <= 0 || teamSize > slots.length) return [];

  const leadSize = Math.min(2, teamSize);
  const actions = [];
  for (const leads of combinations(slots, leadSize)) {
    const leadSet = new Set(leads);
    const remaining = slots.filter(slot => !leadSet.has(slot));
    for (const backs of combinations(remaining, teamSize - leadSize)) {
      const selection = leads.concat(backs);
      actions.push({
        choice: `team ${selection.join('')}`,
        decisions: [{kind: 'team', slots: selection}],
      });
    }
  }
  return actions;
}

module.exports = {
  canonicalTeamPreviewChoice,
  combinations,
  enumerateCanonicalTeamPreviewActions,
};
