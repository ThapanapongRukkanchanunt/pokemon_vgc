const websocketUrl = process.argv[2] || 'wss://sim3.psim.us/showdown/websocket';
const timeoutMs = Number(process.env.SHOWDOWN_SMOKE_TIMEOUT_MS || 15000);

if (typeof WebSocket !== 'function') {
  console.error('FAIL Node.js 22 or newer is required for global WebSocket support');
  process.exit(1);
}

const socket = new WebSocket(websocketUrl);
const timer = setTimeout(() => {
  console.error(`FAIL no challstr received within ${timeoutMs}ms`);
  socket.close();
  process.exitCode = 1;
}, timeoutMs);

socket.addEventListener('message', event => {
  if (!String(event.data).includes('|challstr|')) return;
  clearTimeout(timer);
  console.log(`PASS Showdown WebSocket challstr handshake: ${websocketUrl}`);
  socket.close(1000, 'smoke complete');
});
socket.addEventListener('error', () => {
  clearTimeout(timer);
  console.error(`FAIL Showdown WebSocket connection: ${websocketUrl}`);
  process.exitCode = 1;
});
