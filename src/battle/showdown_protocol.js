const path = require('node:path');

const repoRoot = path.join(__dirname, '..', '..');
const showdownRoot = path.join(repoRoot, 'vendor', 'pokemon-showdown');
const {BattleStream} = require(path.join(showdownRoot, 'dist', 'sim', 'battle-stream.js'));
const {Dex} = require(path.join(showdownRoot, 'dist', 'sim', 'dex.js'));
const {extractChannelMessages} = require(path.join(showdownRoot, 'dist', 'sim', 'battle.js'));
const {Teams} = require(path.join(showdownRoot, 'dist', 'sim', 'teams.js'));
const {TeamValidator} = require(path.join(showdownRoot, 'dist', 'sim', 'team-validator.js'));

function requestFromChunk(chunk, side) {
  if (!chunk.startsWith(`sideupdate\n${side}\n`) || !chunk.includes('|request|')) return null;
  const requestLine = chunk.split('\n').find(line => line.startsWith('|request|'));
  if (!requestLine) return null;
  return JSON.parse(requestLine.slice('|request|'.length));
}

function winnerFromChunk(chunk) {
  const winLine = chunk.split('\n').find(line => line.startsWith('|win|'));
  return winLine ? winLine.slice('|win|'.length) : null;
}

function isTurnUpdate(chunk) {
  return chunk.includes('|turn|');
}

function createBattleStream(options = {}) {
  return new BattleStream({noCatch: true, ...options});
}

function dexForFormat(formatId) {
  return Dex.forFormat(formatId);
}

function canonicalFormatId(formatId) {
  const format = dexForFormat(formatId).formats.get(formatId);
  if (!format?.exists) throw new Error(`Unknown Showdown format: ${formatId}`);
  return format.id;
}

function spectatorLinesFromChunk(chunk) {
  if (!chunk.startsWith('update\n')) return [];
  const data = chunk.slice('update\n'.length);
  return extractChannelMessages(data, [0])[0];
}

function validateAndPackTeam({formatId, importText}) {
  const parsedTeam = Teams.import(importText);
  const problems = TeamValidator.get(formatId).validateTeam(parsedTeam);
  if (problems) throw new Error(problems.join('\n'));
  return Teams.pack(parsedTeam);
}

module.exports = {
  canonicalFormatId,
  createBattleStream,
  dexForFormat,
  requestFromChunk,
  spectatorLinesFromChunk,
  validateAndPackTeam,
  winnerFromChunk,
  isTurnUpdate,
};
