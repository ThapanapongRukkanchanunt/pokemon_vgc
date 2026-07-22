function clamp01(value) {
  return Math.max(0, Math.min(1, Number(value) || 0));
}

function payoffKey(currentTeamId, snapshotId, opponentTeamId) {
  return `${currentTeamId}|${snapshotId}|${opponentTeamId}`;
}

function smoothedWinRate(record = null, priorGames = 2, priorWinRate = 0.5) {
  const games = Math.max(0, Number(record?.games) || 0);
  const wins = Math.max(0, Number(record?.wins) || 0);
  const pseudoGames = Math.max(0, Number(priorGames) || 0);
  const denominator = games + pseudoGames;
  if (!(denominator > 0)) return clamp01(priorWinRate);
  return clamp01((wins + pseudoGames * clamp01(priorWinRate)) / denominator);
}

function pfspWeight(record, {exponent = 2, minimum = 1e-3, priorGames = 2} = {}) {
  const winRate = smoothedWinRate(record, priorGames, 0.5);
  return Math.max(minimum, Math.pow(1 - winRate, Math.max(0, Number(exponent) || 0)));
}

function weightedPick(items, weights, rng) {
  if (!items.length) throw new Error('Cannot sample an empty PFSP pool');
  const normalized = weights.map(weight => Math.max(0, Number(weight) || 0));
  const total = normalized.reduce((sum, weight) => sum + weight, 0);
  if (!(total > 0)) return items[Math.floor(rng.next() * items.length)];
  let cursor = rng.next() * total;
  for (let index = 0; index < items.length; index++) {
    cursor -= normalized[index];
    if (cursor <= 0) return items[index];
  }
  return items.at(-1);
}

function selectHistoricalOpponent({currentTeamId, candidates, payoffs, rng, exponent = 2, priorGames = 2}) {
  const eligible = candidates.filter(candidate => candidate.team.id !== currentTeamId);
  if (!eligible.length) throw new Error(`No historical opponents available for ${currentTeamId}`);
  const weights = eligible.map(candidate => pfspWeight(
    payoffs.get(payoffKey(currentTeamId, candidate.snapshot.id, candidate.team.id)),
    {exponent, priorGames}
  ));
  return weightedPick(eligible, weights, rng);
}

module.exports = {
  payoffKey,
  pfspWeight,
  selectHistoricalOpponent,
  smoothedWinRate,
  weightedPick,
};
