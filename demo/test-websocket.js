/**
 * VoxGuard demo — WebSocket end-to-end test.
 *
 * Connects to the running backend, streams 3 audio_chunk messages covering the
 * 3 demo scenarios, and logs every server response.
 *
 * Run: node demo/test-websocket.js
 */

const WebSocket = require('ws');

const WS_URL = process.env.WS_URL || 'ws://localhost:4000';

// The 3 demo scenarios. audio_base64 is a dummy placeholder for now.
const SCENARIOS = [
  {
    label: 'Scenario 1 — expected LOW risk',
    transcript: 'Hello, how are you today?',
  },
  {
    label: 'Scenario 2 — expected HIGH risk',
    transcript:
      'This is your bank manager, please share your OTP immediately or your account will be blocked',
  },
  {
    label: 'Scenario 3 — expected HIGH risk',
    transcript: "I need you to transfer money urgently, don't tell anyone",
  },
];

const ws = new WebSocket(WS_URL);

let sent = 0;
let results = 0;

ws.on('open', () => {
  console.log(`Connected to ${WS_URL}`);
});

ws.on('message', (raw) => {
  let msg;
  try {
    msg = JSON.parse(raw.toString());
  } catch (err) {
    console.error('Could not parse server message:', raw.toString());
    return;
  }

  if (msg.type === 'session_started') {
    console.log(`\nSession started: ${msg.session_id}\n`);
    sendNextChunk();
    return;
  }

  if (msg.type === 'chunk_result') {
    results += 1;
    const scenario = SCENARIOS[msg.chunk_id - 1];
    console.log('----------------------------------------------------------');
    console.log(`${scenario ? scenario.label : `chunk ${msg.chunk_id}`}`);
    console.log(`  transcript      : ${msg.ml.transcript}`);
    console.log(`  risk_level      : ${msg.fusion.risk_level}`);
    console.log(`  final_risk_score: ${msg.fusion.final_risk_score}`);
    console.log(`  reason          : ${msg.fusion.reason}`);
    console.log(`  ml              : spoof_score=${msg.ml.spoof_score} ` +
      `spoof_label=${msg.ml.spoof_label} speaker_status=${msg.ml.speaker_status} ` +
      `speaker_similarity=${msg.ml.speaker_similarity}`);
    console.log(`  llm             : risk=${msg.llm.llm_risk_score} ` +
      `signals=[${msg.llm.detected_signals.join(', ')}]`);
    console.log(`  latency_ms      : ${msg.latency_ms}`);
    console.log('  full payload    :', JSON.stringify(msg));

    if (results >= SCENARIOS.length) {
      console.log('\nAll scenarios processed. Closing connection.');
      ws.close();
    } else {
      sendNextChunk();
    }
    return;
  }

  console.log('Other message from server:', JSON.stringify(msg));
});

ws.on('close', () => {
  console.log('Connection closed.');
  process.exit(0);
});

ws.on('error', (err) => {
  console.error('WebSocket error:', err.message);
  console.error('Is the backend running on', WS_URL, '? Start it with: npm run dev');
  process.exit(1);
});

function sendNextChunk() {
  if (sent >= SCENARIOS.length) return;
  const scenario = SCENARIOS[sent];
  sent += 1;
  console.log(`Sending chunk ${sent}: "${scenario.transcript}"`);
  ws.send(
    JSON.stringify({
      type: 'audio_chunk',
      audio_base64: 'dummyaudio',
      transcript: scenario.transcript,
    })
  );
}
