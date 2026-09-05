import { getDB } from './init';
import type { ChunkAnalysis } from '../types/contracts';

export function createSession(sessionId: string): void {
  const db = getDB();
  db.prepare(
    `INSERT INTO sessions (id, created_at, status) VALUES (?, ?, 'active')`
  ).run(sessionId, new Date().toISOString());
}

export function endSession(sessionId: string): void {
  const db = getDB();
  db.prepare(
    `UPDATE sessions SET ended_at = ?, status = 'ended' WHERE id = ?`
  ).run(new Date().toISOString(), sessionId);
}

export function saveChunkAnalysis(data: ChunkAnalysis): void {
  const db = getDB();
  db.prepare(
    `INSERT INTO chunk_analyses (
      session_id, chunk_id,
      spoof_score, speaker_similarity, speaker_status, spoof_label, transcript,
      llm_risk_score, detected_signals,
      final_risk_score, risk_level, reason,
      latency_ms, timestamp
    ) VALUES (
      @session_id, @chunk_id,
      @spoof_score, @speaker_similarity, @speaker_status, @spoof_label, @transcript,
      @llm_risk_score, @detected_signals,
      @final_risk_score, @risk_level, @reason,
      @latency_ms, @timestamp
    )`
  ).run({
    session_id: data.session_id,
    chunk_id: data.chunk_id,
    spoof_score: data.ml.spoof_score,
    speaker_similarity: data.ml.speaker_similarity,
    speaker_status: data.ml.speaker_status,
    spoof_label: data.ml.spoof_label,
    transcript: data.ml.transcript,
    llm_risk_score: data.llm.llm_risk_score,
    detected_signals: JSON.stringify(data.llm.detected_signals),
    final_risk_score: data.fusion.final_risk_score,
    risk_level: data.fusion.risk_level,
    reason: data.fusion.reason,
    latency_ms: data.latency_ms,
    timestamp: data.timestamp,
  });
}

export function getSessionHistory(limit: number): object[] {
  const db = getDB();
  return db
    .prepare(
      `SELECT
        s.id AS session_id,
        s.created_at,
        s.ended_at,
        s.status,
        c.id AS analysis_id,
        c.chunk_id,
        c.spoof_score,
        c.speaker_similarity,
        c.speaker_status,
        c.spoof_label,
        c.transcript,
        c.llm_risk_score,
        c.detected_signals,
        c.final_risk_score,
        c.risk_level,
        c.reason,
        c.latency_ms,
        c.timestamp
      FROM sessions s
      LEFT JOIN chunk_analyses c ON c.session_id = s.id
      WHERE s.id IN (
        SELECT id FROM sessions ORDER BY created_at DESC LIMIT ?
      )
      ORDER BY s.created_at DESC, c.chunk_id ASC`
    )
    .all(limit) as object[];
}
