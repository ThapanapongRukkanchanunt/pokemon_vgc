const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {loadFinalAgentPackage, sha256} = require('../src/final_agent_package');

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'pokemon-vgc-final-package-'));
const teamSource = path.join(__dirname, '..', 'data', 'teams', 'imports', 'mb_006_ladder_blastoise_delphox.txt');
const battlePath = path.join(root, 'battle_checkpoint.pt');
const previewPath = path.join(root, 'preview_checkpoint.pt');
const teamPath = path.join(root, 'team.txt');

try {
  fs.writeFileSync(battlePath, 'battle checkpoint fixture');
  fs.writeFileSync(previewPath, 'preview checkpoint fixture');
  fs.copyFileSync(teamSource, teamPath);
  const manifest = {
    package_version: 1,
    format_id: 'gen9championsvgc2026regmb',
    team_id: 'mb-006',
    team_name: 'Fixture',
    battle_checkpoint: path.basename(battlePath),
    preview_checkpoint: path.basename(previewPath),
    team_import: path.basename(teamPath),
    sha256: {
      battle_checkpoint: sha256(battlePath),
      preview_checkpoint: sha256(previewPath),
      team_import: sha256(teamPath),
    },
  };
  fs.writeFileSync(path.join(root, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);

  const loaded = loadFinalAgentPackage(root);
  assert.equal(loaded.manifest.team_id, 'mb-006');
  assert.equal(loaded.sets.length, 6);

  fs.appendFileSync(previewPath, 'tampered');
  assert.throws(() => loadFinalAgentPackage(root), /SHA-256 mismatch/);
  console.log('PASS final-agent package parsing and SHA-256 verification');
} finally {
  fs.rmSync(root, {recursive: true, force: true});
}
