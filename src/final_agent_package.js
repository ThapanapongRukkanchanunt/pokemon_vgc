const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {Teams} = require('../vendor/pokemon-showdown/dist/sim/teams.js');

function sha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function packageMember(packageDir, relativePath, label) {
  if (!relativePath || path.isAbsolute(relativePath)) {
    throw new Error(`Invalid ${label} path in package manifest`);
  }
  const resolved = path.resolve(packageDir, relativePath);
  const relative = path.relative(packageDir, resolved);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`${label} leaves the package directory`);
  }
  if (!fs.existsSync(resolved)) throw new Error(`Missing packaged ${label}: ${resolved}`);
  return resolved;
}

function verifyHash(filePath, expected, label) {
  if (!/^[a-f0-9]{64}$/i.test(expected || '')) {
    throw new Error(`Missing or invalid ${label} SHA-256 in package manifest`);
  }
  const actual = sha256(filePath);
  if (actual.toLowerCase() !== expected.toLowerCase()) {
    throw new Error(`SHA-256 mismatch for packaged ${label}`);
  }
}

function loadFinalAgentPackage(packageDir) {
  const resolvedDir = path.resolve(packageDir);
  const manifestPath = path.join(resolvedDir, 'manifest.json');
  if (!fs.existsSync(manifestPath)) throw new Error(`Missing package manifest: ${manifestPath}`);
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  if (manifest.package_version !== 1) {
    throw new Error(`Unsupported final-agent package version: ${manifest.package_version}`);
  }
  for (const field of ['format_id', 'team_id', 'battle_checkpoint', 'preview_checkpoint', 'team_import']) {
    if (!manifest[field]) throw new Error(`Package manifest is missing ${field}`);
  }

  const battleModelPath = packageMember(resolvedDir, manifest.battle_checkpoint, 'battle checkpoint');
  const previewModelPath = packageMember(resolvedDir, manifest.preview_checkpoint, 'preview checkpoint');
  const teamImportPath = packageMember(resolvedDir, manifest.team_import, 'team import');
  verifyHash(battleModelPath, manifest.sha256?.battle_checkpoint, 'battle checkpoint');
  verifyHash(previewModelPath, manifest.sha256?.preview_checkpoint, 'preview checkpoint');
  verifyHash(teamImportPath, manifest.sha256?.team_import, 'team import');

  const teamImportText = fs.readFileSync(teamImportPath, 'utf8');
  const sets = Teams.import(teamImportText);
  if (!sets?.length) throw new Error('Packaged team import could not be parsed');
  return {
    packageDir: resolvedDir,
    manifestPath,
    manifest,
    battleModelPath,
    previewModelPath,
    teamImportPath,
    teamImportText,
    sets,
  };
}

module.exports = {loadFinalAgentPackage, sha256};
