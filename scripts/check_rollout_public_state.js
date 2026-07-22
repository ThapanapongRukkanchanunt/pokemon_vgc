const assert = require('node:assert/strict');
const {publicStateFromBattle} = require('../src/battle/rollout_search');

const hiddenSideData = [{
  active: [{
    name: 'Hidden Active Name',
    species: {name: 'Hidden Active Species'},
    hp: 123,
    maxhp: 157,
    fainted: false,
  }],
}];

const battle = {
  sides: hiddenSideData,
  log: [
    '|switch|p1a: Alpha|Alpha, L50|157/157',
    '|switch|p1b: Beta|Beta, L50|171/171',
    '|switch|p2a: Gamma|Gamma, L50|100/100',
    '|switch|p2b: Delta|Delta, L50|100/100',
    '|split|p2',
    '|-damage|p2a: Gamma|73/157',
    '|-damage|p2a: Gamma|47/100',
    '|switch|p2a: Epsilon|Epsilon, L50|100/100',
    '|detailschange|p2a: Epsilon|Epsilon-Mega, L50',
    '|faint|p2b: Delta',
  ],
};

const state = publicStateFromBattle(battle);
assert.deepEqual(state.revealed.p1, ['Alpha', 'Beta']);
assert.deepEqual(state.revealed.p2, ['Gamma', 'Delta', 'Epsilon-Mega']);
assert.deepEqual(state.fainted.p2, ['Delta']);
assert.equal(state.active.p2[0].species, 'Epsilon-Mega');
assert.equal(state.active.p2[1].condition, '0 fnt');
assert(!JSON.stringify(state).includes('Hidden'), 'hidden simulator data must not enter public state');
assert(!JSON.stringify(state).includes('73/157'), 'opponent-private exact HP must not enter spectator state');

console.log('PASS rollout state is reconstructed from public spectator history');
