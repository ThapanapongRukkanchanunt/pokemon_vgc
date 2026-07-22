const websocketUrl = process.argv[2] || 'wss://sim3.psim.us/showdown/websocket';
const expectedFormatId = process.argv[3] || 'gen9championsvgc2026regmb';
const timeoutMs = Number(process.env.SHOWDOWN_SMOKE_TIMEOUT_MS || 15000);

if (typeof WebSocket !== 'function') {
  console.error('FAIL Node.js 22 or newer is required for global WebSocket support');
  process.exit(1);
}

const socket = new WebSocket(websocketUrl);
let receivedChallstr = false;
let formatAvailable = false;
const timer = setTimeout(() => {
  console.error(
    `FAIL Showdown readiness within ${timeoutMs}ms ` +
    `(challstr=${receivedChallstr}, format_${expectedFormatId}=${formatAvailable})`
  );
  socket.close();
  process.exitCode = 1;
}, timeoutMs);

socket.addEventListener('message', event => {
  const data = String(event.data);
  if (data.includes('|challstr|')) receivedChallstr = true;
  if (data.includes('|formats|')) {
    const normalized = data.toLowerCase().replace(/[^a-z0-9]+/g, '');
    formatAvailable = normalized.includes(expectedFormatId);
  }
  if (!receivedChallstr || !formatAvailable) return;
  clearTimeout(timer);
  console.log(`PASS Showdown handshake and searchable format ${expectedFormatId}: ${websocketUrl}`);
  socket.close(1000, 'smoke complete');
});
socket.addEventListener('error', () => {
  clearTimeout(timer);
  console.error(`FAIL Showdown WebSocket connection: ${websocketUrl}`);
  process.exitCode = 1;
});
