const path = require('node:path');

const showdownRoot = path.join(__dirname, '..', 'vendor', 'pokemon-showdown');
const {BattleStream} = require(path.join(showdownRoot, 'dist', 'sim', 'battle-stream.js'));
const {Teams} = require(path.join(showdownRoot, 'dist', 'sim', 'teams.js'));
const {TeamValidator} = require(path.join(showdownRoot, 'dist', 'sim', 'team-validator.js'));

const formatid = 'vgc';

const teamText = `
Charizard @ Charizardite Y
Ability: Blaze
Level: 50
- Heat Wave
- Solar Beam
- Protect
- Air Slash

Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
- Fake Out
- Flare Blitz
- Parting Shot
- Darkest Lariat

Garchomp @ Soft Sand
Ability: Rough Skin
Level: 50
- Earthquake
- Dragon Claw
- Protect
- Rock Slide

Whimsicott @ Focus Sash
Ability: Prankster
Level: 50
- Tailwind
- Moonblast
- Encore
- Protect

Kingambit @ Black Glasses
Ability: Defiant
Level: 50
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Protect

Rotom-Wash @ Leftovers
Ability: Levitate
Level: 50
- Hydro Pump
- Thunderbolt
- Will-O-Wisp
- Protect
`;

const team = Teams.import(teamText);
const validationProblems = TeamValidator.get(formatid).validateTeam(team);
if (validationProblems) {
  throw new Error(validationProblems.join('\n'));
}

const packedTeam = Teams.pack(team);
const stream = new BattleStream({noCatch: true});

let turns = 0;
let winner = null;

void (async () => {
  while (true) {
    const chunk = await stream.read();
    if (chunk === null) break;

    if (chunk.includes('|turn|')) turns += 1;
    const winLine = chunk.split('\n').find(line => line.startsWith('|win|'));
    if (winLine) winner = winLine.slice('|win|'.length);

    for (const player of ['p1', 'p2']) {
      if (!chunk.startsWith(`sideupdate\n${player}\n`) || !chunk.includes('|request|')) continue;
      const requestLine = chunk.split('\n').find(line => line.startsWith('|request|'));
      if (!requestLine) continue;
      const request = JSON.parse(requestLine.slice('|request|'.length));
      if (request.wait) continue;
      if (request.requestType === 'teampreview') {
        stream.write(`>${player} team 1234`);
      } else {
        stream.write(`>${player} default`);
      }
    }

    if (chunk.startsWith('end\n')) {
      console.log(JSON.stringify({formatid, turns, winner}, null, 2));
      break;
    }
  }
})();

stream.write(`>start {"formatid":"${formatid}","seed":[1,2,3,4]}`);
stream.write(`>player p1 {"name":"Smoke A","team":${JSON.stringify(packedTeam)}}`);
stream.write(`>player p2 {"name":"Smoke B","team":${JSON.stringify(packedTeam)}}`);
