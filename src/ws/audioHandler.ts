import type http from 'http';
import { randomUUID } from 'crypto';
import { WebSocket, WebSocketServer } from 'ws';
import { createSession, endSession, saveChunkAnalysis } from '../db/queries';
import type { ChunkAnalysis } from '../types/contracts';
import { callMLAnalyze } from '../services/mlBridge';
import { scoreTranscript } from '../services/claudeScorer';
import { fuseScores } from '../services/fusionEngine';
import { LatencyTracker } from '../utils/latencyTracker';
import { logger } from '../utils/logger';

interface IncomingMessage {
  type: string;
  audio_base64?: string;
  transcript?: string;
}

export function setupWebSocket(server: http.Server): void {
  const wss = new WebSocketServer({ server });

  wss.on('connection', (ws: WebSocket) => {
    const session_id = randomUUID();
    let chunk_id = 0;
    const tracker = new LatencyTracker(session_id);

    createSession(session_id);
    logger.info(`New session started: ${session_id}`);
    ws.send(JSON.stringify({ type: 'session_started', session_id }));

    ws.on('message', (raw) => {
      let message: IncomingMessage;
      try {
        message = JSON.parse(raw.toString());
      } catch (err) {
        logger.error(err, 'Failed to parse WebSocket message');
        return;
      }

      if (message.type !== 'audio_chunk') {
        return;
      }

      chunk_id += 1;
      const currentChunkId = chunk_id;
      logger.info(`Received chunk ${currentChunkId} for session ${session_id}`);

      void (async () => {
        tracker.start(currentChunkId);

        const ml = await callMLAnalyze(
          message.audio_base64 || '',
          message.transcript || ''
        );
        const llm = await scoreTranscript(ml.transcript);
        const fusion = fuseScores(ml, llm);
        const latency_ms = tracker.end(currentChunkId);

        const chunkAnalysis: ChunkAnalysis = {
          session_id,
          chunk_id: currentChunkId,
          ml,
          llm,
          fusion,
          latency_ms,
          timestamp: new Date().toISOString(),
        };

        try {
          saveChunkAnalysis(chunkAnalysis);
        } catch (err) {
          logger.error(err, 'Failed to persist chunk analysis');
        }

        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'chunk_result', ...chunkAnalysis }));
        }
      })();
    });

    ws.on('close', () => {
      endSession(session_id);
      logger.info(`Session ended: ${session_id}`);
      logger.info(
        { session_id, ...tracker.getSummary() },
        `Latency summary for session ${session_id}`
      );
    });

    ws.on('error', (err) => {
      logger.error(err, `WebSocket error for session ${session_id}`);
      endSession(session_id);
    });
  });
}
