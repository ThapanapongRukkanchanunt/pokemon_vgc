const assert = require('node:assert/strict');
const fs = require('node:fs');
const {Teams} = require('../vendor/pokemon-showdown/dist/sim/teams.js');
const {
  LadderBattle,
  splitServerPayload,
  teamSummaryFromPacked,
} = require('../src/showdown_ladder');

const blocks = splitServerPayload('>battle-test-1\n|init|battle\n|turn|1\n>lobby\n|users|0');
assert.equal(blocks.length, 2);
assert.equal(blocks[0].roomId, 'battle-test-1');
assert.deepEqual(blocks[0].lines, ['|init|battle', '|turn|1']);

const importText = fs.readFileSync('data/teams/imports/mb_006_ladder_blastoise_delphox.txt', 'utf8');
const packed = Teams.pack(Teams.import(importText));
const summary = teamSummaryFromPacked(packed);
assert.equal(summary.sets.length, 6);
assert.ok(summary.sets[0].nature);
assert.ok(summary.sets[0].evs);

const otsPacked = Teams.pack(Teams.import(importText).map(set => ({...set, nature: '', evs: null, ivs: null})));
const otsSummary = teamSummaryFromPacked(otsPacked);
assert.equal(otsSummary.sets[0].nature, summary.sets[0].nature);
assert.equal(otsSummary.sets[0].evs, summary.sets[0].evs);
assert.equal(otsSummary.sets[0].spread_source, 'curated_mb_prior');
assert.equal(otsSummary.sets[0].spread_confidence, 'high');

const sent = [];
let receivedState = null;
const battle = new LadderBattle({
  roomId: 'battle-test-1',
  username: 'VGC Bot',
  ownTeam: {id: 'mb-006', name: 'Test team', sets: summary.sets},
  agent: {
    async chooseAction({battleState}) {
      receivedState = battleState;
      return 'team 1234';
    },
  },
  send: message => sent.push(message),
});
const previewRequest = {
  rqid: 7,
  teamPreview: true,
  maxChosenTeamSize: 4,
  side: {
    id: 'p1',
    pokemon: summary.sets.map((set, index) => ({
      ident: `p1: ${set.species}`,
      details: `${set.species}, L50`,
      condition: '100/100',
      active: false,
      moves: set.moves,
    })),
  },
};

(async () => {
  await battle.handleLines([
    '|player|p1|VGC Bot|1',
    '|player|p2|Opponent|1',
    `|showteam|p2|${packed}`,
    '|uhtml|otsrequest|buttons',
    '|start',
    `|request|${JSON.stringify(previewRequest)}`,
  ]);
  assert.equal(battle.ownSide, 'p1');
  assert.ok(sent.includes('battle-test-1|/acceptopenteamsheets'));
  assert.ok(sent.includes('battle-test-1|/timer on'));
  assert.ok(sent.includes('battle-test-1|/choose team 1234|7'));
  assert.equal(receivedState.teams.p1.id, 'mb-006');
  assert.equal(receivedState.teams.p2.id, 'ladder-opponent');
  assert.equal(receivedState.teams.p2.team_summary.sets.length, 6);
  console.log('PASS Showdown payload, OTS, timer, rqid choice, and packed-team handling');
})().catch(error => {
  console.error(error.stack || error.message);
  process.exit(1);
});
