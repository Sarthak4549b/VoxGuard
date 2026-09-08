import type http from 'http';
import { randomUUID } from 'crypto';
import { WebSocket, WebSocketServer } from 'ws';
import { createSession, endSession, saveChunkAnalysis } from '../db/queries';
import type { ChunkAnalysis, FusionResult } from '../types/contracts';
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

    // --- Session-level "sticky" risk state -----------------------------------
    // The UI only renders the latest chunk, so a one-chunk HIGH alert followed
    // by silent chunks (llm_risk = 0 -> LOW) was being visually overwritten.
    // We keep the running peak here and stream it alongside the per-chunk data.
    //
    // Only fraud CONTENT (LLM detected_signals) is trusted to move the session
    // risk. Speaker MISMATCH and spoof_strong are far too noisy on compressed
    // browser audio without reliable speaker enrollment - speaker similarity
    // dips below threshold at random and spoof_score saturates near 1.0 even
    // for genuine speech - so acoustic signals ALONE never lift the session
    // above LOW. A chunk with no fraud signals is treated as noise.
    const RISK_RANK: Record<FusionResult['risk_level'], number> = {
      LOW: 0,
      MEDIUM: 1,
      HIGH: 2,
    };
    // Representative score to park the session at after a decay step, kept
    // inside the target level's band so it does not immediately re-escalate.
    const LEVEL_FLOOR: Record<FusionResult['risk_level'], number> = {
      LOW: 0,
      MEDIUM: 0.4,
      HIGH: 0.7,
    };
    // A fraud-content peak is sticky: it decays one level only after this many
    // consecutive clean-OR-silent chunks (no fraud words spoken). Fraud content
    // is the only thing that can ever elevate the session, so this is the only
    // decay cadence in play.
    const DECAY_AFTER_CLEAN_CHUNKS = 5;

    let session_peak_risk_score = 0;
    let session_peak_risk_level: FusionResult['risk_level'] = 'LOW';
    let session_peak_reason = 'No elevated risk detected yet';
    const accumulated_signals = new Set<string>();
    let consecutive_clean_chunks = 0;

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

        // The transcript that feeds the fraud scorer: prefer the ML server's
        // Whisper output, fall back to any client-supplied transcript. Keep
        // ml.transcript in sync so the persisted/streamed value matches what
        // was actually scored.
        const transcriptForScoring = ml.transcript || message.transcript || '';
        ml.transcript = transcriptForScoring;
        console.log(
          `[audioHandler] chunk ${currentChunkId} | transcript received ` +
            `(${transcriptForScoring.length} chars): ` +
            `${JSON.stringify(transcriptForScoring)}`
        );

        const llm = await scoreTranscript(transcriptForScoring);
        console.log(
          `[audioHandler] chunk ${currentChunkId} | llm_risk_score=` +
            `${llm.llm_risk_score} signals=[${llm.detected_signals.join(', ')}]`
        );

        const fusion = fuseScores(ml, llm);
        console.log(
          `[audioHandler] chunk ${currentChunkId} | risk_hint=${ml.risk_hint} ` +
            `spoof_strong=${ml.spoof_strong} -> final_risk_score=` +
            `${fusion.final_risk_score} risk_level=${fusion.risk_level} ` +
            `reason="${fusion.reason}"`
        );
        const latency_ms = tracker.end(currentChunkId);

        // --- Update session-level sticky risk --------------------------------
        // Only fraud CONTENT drives the session risk. A silent chunk (empty
        // transcript) carries no fraud, so it does NOT escalate - but while
        // the session is elevated it DOES count as a clean chunk and advances
        // the decay streak (normal conversation is full of pauses, and those
        // pauses were previously freezing an elevated session forever).
        const isSilentChunk = transcriptForScoring.trim().length < 3;
        const fraudContent = llm.detected_signals.length > 0;

        if (!isSilentChunk && fraudContent) {
          for (const signal of llm.detected_signals) {
            accumulated_signals.add(signal);
          }

          // Fraud words are present, so trust fusion's level (MEDIUM/HIGH) as
          // is. Adopt this chunk as the new session peak when it out-ranks or
          // out-scores the current peak.
          const isElevated =
            fusion.risk_level === 'MEDIUM' || fusion.risk_level === 'HIGH';
          if (
            isElevated &&
            (RISK_RANK[fusion.risk_level] > RISK_RANK[session_peak_risk_level] ||
              fusion.final_risk_score > session_peak_risk_score)
          ) {
            session_peak_risk_score = fusion.final_risk_score;
            session_peak_risk_level = fusion.risk_level;
            session_peak_reason = fusion.reason;
            console.log(
              `[audioHandler] chunk ${currentChunkId} | session peak escalated ` +
                `-> ${session_peak_risk_level} score=${session_peak_risk_score} ` +
                `(fraud content) reason="${session_peak_reason}"`
            );
          }
        } else if (!isSilentChunk && fusion.risk_level === 'HIGH') {
          // Fusion scored this chunk HIGH on acoustic signals alone - a noisy
          // speaker MISMATCH and/or a saturated spoof_score - with no fraud
          // words. On browser audio those signals are unreliable, so this is
          // ignored by design and must not touch the session risk.
          console.log(
            `[audioHandler] chunk ${currentChunkId} | ignoring acoustic-only ` +
              `HIGH (no fraud content): risk_hint=${ml.risk_hint} ` +
              `spoof_strong=${ml.spoof_strong} speaker_status=${ml.speaker_status}`
          );
        }

        // Decay: a clean OR silent chunk advances the streak. "Clean" now just
        // means no fraud words were spoken - acoustic noise is irrelevant.
        // After DECAY_AFTER_CLEAN_CHUNKS such chunks an elevated session (which
        // can only ever be fraud-content based) drops one level.
        const chunkIsClean = isSilentChunk || !fraudContent;
        consecutive_clean_chunks = chunkIsClean
          ? consecutive_clean_chunks + 1
          : 0;

        if (
          consecutive_clean_chunks >= DECAY_AFTER_CLEAN_CHUNKS &&
          session_peak_risk_level !== 'LOW'
        ) {
          const decayed: FusionResult['risk_level'] =
            session_peak_risk_level === 'HIGH' ? 'MEDIUM' : 'LOW';
          session_peak_risk_level = decayed;
          session_peak_risk_score = LEVEL_FLOOR[decayed];
          session_peak_reason =
            `Risk decayed to ${decayed} after ${consecutive_clean_chunks} ` +
            `clean/silent chunks with no fraud content`;
          consecutive_clean_chunks = 0;
          console.log(
            `[audioHandler] chunk ${currentChunkId} | session risk decayed ` +
              `-> ${session_peak_risk_level} score=${session_peak_risk_score}`
          );
        }

        console.log(
          `[audioHandler] chunk ${currentChunkId} | silent=${isSilentChunk} ` +
            `fraud_content=${fraudContent} clean_streak=${consecutive_clean_chunks} -> ` +
            `session_risk_score=${session_peak_risk_score} ` +
            `session_risk_level=${session_peak_risk_level}`
        );

        const chunkAnalysis: ChunkAnalysis = {
          session_id,
          chunk_id: currentChunkId,
          ml,
          llm,
          fusion,
          latency_ms,
          timestamp: new Date().toISOString(),
          session_risk_score: session_peak_risk_score,
          session_risk_level: session_peak_risk_level,
          session_reason: session_peak_reason,
          session_signals: Array.from(accumulated_signals),
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
