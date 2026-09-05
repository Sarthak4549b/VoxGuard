# VoxGuard Demo Scripts

Standalone Node.js scripts (plain JS) for exercising the VoxGuard backend
during demos. They depend only on the `ws` package, which is already installed
in the project's `node_modules`.

## Prerequisites

1. Copy the env file and (optionally) add an Anthropic API key for real LLM scoring:

   ```
   cp .env.example .env
   ```

   Without `ANTHROPIC_API_KEY` set, the Claude scorer falls back to
   `llm_risk_score: 0`. Without the ML service running on `ML_API_URL`, the ML
   bridge falls back to a neutral `spoof_score: 0.5`. The pipeline still runs
   end to end in both cases.

2. Start the backend in another terminal:

   ```
   npm run dev
   ```

   Wait for the log line: `VoxGuard backend running on port 4000`.

## test-websocket.js

Connects to `ws://localhost:4000`, waits for the `session_started` message,
then streams the 3 demo scenarios as `audio_chunk` messages (one at a time,
each sent after the previous result comes back):

| # | Transcript | Expected risk |
|---|------------|---------------|
| 1 | `Hello, how are you today?` | LOW |
| 2 | `This is your bank manager, please share your OTP immediately or your account will be blocked` | HIGH |
| 3 | `I need you to transfer money urgently, don't tell anyone` | HIGH |

`audio_base64` is sent as the dummy string `dummyaudio`.

For every server response it logs `risk_level`, `final_risk_score`, `reason`,
the ML/LLM sub-scores, `latency_ms`, and the full JSON payload. After the 3rd
`chunk_result` it closes the connection.

### Run

```
node demo/test-websocket.js
```

Override the target with an env var if needed:

```
WS_URL=ws://localhost:5000 node demo/test-websocket.js
```

> Note: the expected HIGH ratings for scenarios 2 and 3 require the Claude
> scorer to be active (valid `ANTHROPIC_API_KEY`). With the scorer and ML
> service both offline, all three scenarios score `LOW` (`final_risk_score`
> `0.3`) because only the neutral ML fallback contributes.
