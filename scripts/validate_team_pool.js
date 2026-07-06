const fs = require('node:fs');
const path = require('node:path');

const repoRoot = path.join(__dirname, '..');
const showdownRoot = path.join(repoRoot, 'vendor', 'pokemon-showdown');
const {Teams} = require(path.join(showdownRoot, 'dist', 'sim', 'teams.js'));
const {TeamValidator} = require(path.join(showdownRoot, 'dist', 'sim', 'team-validator.js'));

const poolPath = path.join(repoRoot, 'data', 'teams', 'team_pool.json');
const pool = JSON.parse(fs.readFileSync(poolPath, 'utf8'));
const validator = TeamValidator.get(pool.format_id);

let failed = false;

for (const team of pool.teams) {
  const importPath = path.join(repoRoot, team.import_file);
  const text = fs.readFileSync(importPath, 'utf8');
  const parsedTeam = Teams.import(text);
  const validationProblems = validator.validateTeam(parsedTeam);

  const leadProblems = [];
  for (const mode of team.lead_modes || []) {
    if (!/^[1-6]{4}$/.test(mode.team_spec)) {
      leadProblems.push(`${mode.id}: team_spec must be four digits between 1 and 6`);
      continue;
    }
    if (new Set(mode.team_spec).size !== 4) {
      leadProblems.push(`${mode.id}: team_spec must contain four unique slots`);
    }
  }

  if (validationProblems || leadProblems.length) {
    failed = true;
    console.log(`FAIL ${team.id} ${team.name}`);
    for (const problem of validationProblems || []) console.log(`  team: ${problem}`);
    for (const problem of leadProblems) console.log(`  lead: ${problem}`);
  } else {
    console.log(`PASS ${team.id} ${team.name}`);
  }
}

if (failed) process.exit(1);
