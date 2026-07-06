const {
  spectatorLinesFromChunk,
} = require('./showdown_protocol');

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function protocolParts(line) {
  if (!line.startsWith('|')) return {type: '', parts: [line]};
  const parts = line.split('|');
  return {type: parts[1] || '', parts};
}

function pokemonName(ident) {
  return String(ident || '').replace(/^p[12][a-z]?:\s*/, '');
}

function detailsName(details) {
  return String(details || '').split(',')[0].trim();
}

function renderFallbackLine(line) {
  if (!line) return '<div class="spacer battle-history"><br></div>';

  const {type, parts} = protocolParts(line);
  const raw = `<code>${escapeHtml(line)}</code>`;

  switch (type) {
  case 'tier':
    return `<div><small>Format:</small><br><strong>${escapeHtml(parts[2] || '')}</strong><br>${raw}</div>`;
  case 'rated':
    return `<div class="rated"><strong>Rated battle</strong><br>${raw}</div>`;
  case 'rule':
    return `<div><small><em>${escapeHtml(parts[2] || 'Rule')}:</em> ${escapeHtml(parts.slice(3).join('|'))}</small><br>${raw}</div>`;
  case 'player':
    return `<div class="chat battle-history"><strong>${escapeHtml(parts[2] || 'Player')}:</strong> ${escapeHtml(parts[3] || '')}<br>${raw}</div>`;
  case 'poke':
    return `<div class="chat battle-history"><strong>${escapeHtml(parts[2] || '')} team:</strong> ${escapeHtml(detailsName(parts[3]))}<br>${raw}</div>`;
  case 'teampreview':
    return `<div class="chat battle-history"><strong>Team preview</strong> (${escapeHtml(parts[2] || '')})<br>${raw}</div>`;
  case 'start':
    return `<div class="battle-history"><strong>Battle started.</strong><br>${raw}</div>`;
  case 'turn':
    return `<h2 class="battle-history">Turn ${escapeHtml(parts[2] || '')}</h2><div class="battle-history">${raw}</div>`;
  case 'switch':
  case 'drag':
  case 'replace':
    return `<div class="battle-history">${escapeHtml(pokemonName(parts[2]))} entered as <strong>${escapeHtml(detailsName(parts[3]))}</strong>.<br>${raw}</div>`;
  case 'detailschange':
  case '-formechange':
    return `<div class="battle-history">${escapeHtml(pokemonName(parts[2]))} changed to <strong>${escapeHtml(detailsName(parts[3]))}</strong>.<br>${raw}</div>`;
  case 'move':
    return `<div class="battle-history">${escapeHtml(pokemonName(parts[2]))} used <strong>${escapeHtml(parts[3] || '')}</strong>${parts[4] ? ` on ${escapeHtml(pokemonName(parts[4]))}` : ''}.<br>${raw}</div>`;
  case '-damage':
    return `<div class="battle-history"><small>${escapeHtml(pokemonName(parts[2]))}: ${escapeHtml(parts[3] || '')}</small><br>${raw}</div>`;
  case '-heal':
    return `<div class="battle-history"><small>${escapeHtml(pokemonName(parts[2]))} healed to ${escapeHtml(parts[3] || '')}</small><br>${raw}</div>`;
  case 'faint':
    return `<div class="battle-history"><strong>${escapeHtml(pokemonName(parts[2]))} fainted.</strong><br>${raw}</div>`;
  case 'win':
    return `<div class="battle-history"><strong>${escapeHtml(parts[2] || '')} won the battle!</strong><br>${raw}</div>`;
  case 'clearpoke':
  case 'gametype':
  case 'gen':
  case 'teamsize':
  case 'upkeep':
  case 't:':
    return `<div class="battle-history protocol-line">${raw}</div>`;
  default:
    return `<div class="battle-history protocol-line">${raw}</div>`;
  }
}

function fallbackBattleLogHtml(replayLog) {
  const lines = String(replayLog || '').split(/\r?\n/);
  const renderedLines = lines.map(renderFallbackLine).join('');
  return `<div class="battle-log battle-log-inline"><div class="inner"><div class="battle-options"><div style="padding-top: 3px; padding-right: 3px; text-align: right"><button class="icon button" name="openBattleOptions" title="Options">Battle Options</button></div></div><div class="inner message-log">${renderedLines}</div><div class="inner-preempt message-log"></div></div></div>`;
}

function replayIdFromTitle(title) {
  return String(title || 'local-replay')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'local-replay';
}

function replayHtml({title, replayLog, replayId}) {
  const safeReplayId = replayId || replayIdFromTitle(title);
  const fallbackHtml = fallbackBattleLogHtml(replayLog);
  return `<!DOCTYPE html>
<meta charset="utf-8" />
<!-- version 1 -->
<title>${escapeHtml(title)}</title>
<style>
html,body {font-family:Verdana, sans-serif;font-size:10pt;margin:0;padding:0;}body{padding:12px 0;} .battle-log {font-family:Verdana, sans-serif;font-size:10pt;} .battle-log-inline {border:1px solid #AAAAAA;background:#EEF2F5;color:black;max-width:760px;margin:0 auto 80px;padding-bottom:5px;} .battle-log .inner {padding:4px 8px 0px 8px;} .battle-log .inner-preempt {padding:0 8px 4px 8px;} .battle-log h2 {margin:0.5em -8px;padding:4px 8px;border:1px solid #AAAAAA;background:#E0E7EA;border-left:0;border-right:0;font-family:Verdana, sans-serif;font-size:13pt;} .battle-log .chat {vertical-align:middle;padding:3px 0 3px 0;font-size:8pt;} .battle-log .chat strong {color:#40576A;} .battle-log code {white-space:pre-wrap;word-break:break-word;border:1px solid #C0C0C0;background:#F8F8F8;color:#222;padding:0 2px;font-family:Consolas, monospace;font-size:8pt;} .battle-log .rated {padding:3px 4px;} .battle-log .rated strong {color:white;background:#89A;padding:1px 4px;border-radius:4px;} .spacer {margin-top:0.5em;} .protocol-line {color:#3A4A66;} .subtle {color:#3A4A66;}
</style>
<div class="wrapper replay-wrapper" style="max-width:1180px;margin:0 auto">
<input type="hidden" name="replayid" value="${escapeHtml(safeReplayId)}" />
<div class="battle"></div><div class="battle-log"></div><div class="replay-controls"></div><div class="replay-controls-2"></div>
<h1 style="font-weight:normal;text-align:center"><strong>${escapeHtml(title)}</strong></h1>
<script type="text/plain" class="battle-log-data">${escapeHtml(replayLog)}</script>
</div>
${fallbackHtml}
<script>
let daily = Math.floor(Date.now()/1000/60/60/24);document.write('<script src="https://play.pokemonshowdown.com/js/replay-embed.js?version'+daily+'"></'+'script>');
</script>
`;
}

function protocolChunks(protocolText) {
  const chunks = [];
  let current = null;
  const lines = String(protocolText || '').replace(/\r\n/g, '\n').split('\n');

  for (const line of lines) {
    // Blank lines can be meaningful inside Showdown split-channel blocks.
    const startsChunk = line === 'update' || line === 'sideupdate' || line === 'end' || line.startsWith('>');
    if (startsChunk) {
      if (current) chunks.push(current.join('\n'));
      current = [line];
    } else if (current) {
      current.push(line);
    } else if (line.trim()) {
      current = [line];
    }
  }

  if (current) chunks.push(current.join('\n'));
  return chunks.filter(Boolean);
}

function replayLinesFromProtocolText(protocolText) {
  const replayLines = [];
  for (const chunk of protocolChunks(protocolText)) {
    replayLines.push(...spectatorLinesFromChunk(chunk));
  }
  return replayLines;
}

function replayHtmlFromProtocolText({title, protocolText}) {
  const replayLog = replayLinesFromProtocolText(protocolText).join('\n');
  return replayHtml({
    title,
    replayLog,
    replayId: replayIdFromTitle(title),
  });
}

module.exports = {
  fallbackBattleLogHtml,
  replayHtml,
  replayHtmlFromProtocolText,
  replayLinesFromProtocolText,
};
