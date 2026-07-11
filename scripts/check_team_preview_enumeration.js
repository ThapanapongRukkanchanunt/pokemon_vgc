const assert = require('node:assert/strict');
const {enumerateLegalActions} = require('../src/agents/legal_actions');
const {canonicalTeamPreviewChoice} = require('../src/team_preview/preview_actions');

const request = {
  teamPreview: true,
  maxChosenTeamSize: 4,
  side: {pokemon: Array.from({length: 6}, (_, index) => ({ident: `p1: slot-${index + 1}`}))},
};
const actions = enumerateLegalActions({side: 'p1', request, battleState: {}, dex: null});
assert.equal(actions.length, 90);
assert.equal(new Set(actions.map(action => action.choice)).size, 90);
assert(actions.every(action => action.choice === canonicalTeamPreviewChoice(action.choice)));
assert.equal(canonicalTeamPreviewChoice('team 5264'), 'team 2546');
assert.equal(canonicalTeamPreviewChoice('team 2564'), 'team 2546');
console.log('PASS canonical VGC preview enumeration: 15 lead pairs * 6 back pairs = 90 actions');
