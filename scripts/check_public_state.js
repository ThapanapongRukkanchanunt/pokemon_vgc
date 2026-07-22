const assert = require('node:assert/strict');
const {applySpectatorLineToState, createPublicState} = require('../src/battle/public_state');
const {teamContextFromTeams} = require('../src/team_preview/team_context');

const ownTeam = {id: 'own', sets: [{slot: 1, species: 'Ownmon'}]};
const opponentTeam = {
  id: 'foe',
  sets: ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta'].map((species, index) => ({
    slot: index + 1,
    species,
    item: '',
    ability: '',
    nature: '',
    evs: '',
  })),
};
const state = createPublicState();

function predicted() {
  return teamContextFromTeams({ownTeam, opponentTeam, publicState: state, foeSide: 'p2'})
    .predicted_opponent_back.map(set => set.species);
}

assert.equal(predicted().length, 6, 'team preview should retain all six candidates');
applySpectatorLineToState(state, '|switch|p2a: Alpha|Alpha, L50|100/100');
applySpectatorLineToState(state, '|switch|p2b: Beta|Beta, L50|100/100');
assert.deepEqual(predicted(), ['Gamma', 'Delta', 'Epsilon', 'Zeta']);
applySpectatorLineToState(state, '|switch|p2a: Gamma|Gamma, L50|100/100');
assert.deepEqual(predicted(), ['Alpha', 'Delta', 'Epsilon', 'Zeta']);
applySpectatorLineToState(state, '|detailschange|p2a: Gamma|Gamma-Mega, L50');
assert.equal(state.revealed.p2.length, 3, 'forme changes must not count as a second selected Pokemon');
applySpectatorLineToState(state, '|detailschange|p2a: Gamma|Gamma, L50');
applySpectatorLineToState(state, '|switch|p2b: Delta|Delta, L50|100/100');
assert.deepEqual(predicted(), ['Alpha', 'Beta'], 'four reveals should identify the selected back exactly');
applySpectatorLineToState(state, '|faint|p2a: Gamma');
assert.deepEqual(predicted(), ['Alpha', 'Beta'], 'fainted Pokemon must not return to predicted back');

console.log('PASS public reveal history and opponent-back inference');
