export type RiskHint = 'LOW' | 'HIGH' | 'NEUTRAL';

export interface MLAnalyzeResponse {
  speaker_similarity: number;
  speaker_status: 'MATCH' | 'MISMATCH' | 'UNKNOWN';
  spoof_score: number;
  spoof_label: 'GENUINE' | 'SPOOF';
  /**
   * True only when the raw spoof_score is near-certain AND the speaker is a
   * MISMATCH. The raw spoof_score saturates on compressed browser audio, so
   * fusion must use this calibrated flag, never spoof_score directly.
   */
  spoof_strong: boolean;
  /**
   * How the ML server's speaker signal should steer escalation:
   *   LOW     - enrolled speaker MATCH; ignore spoof_score entirely
   *   HIGH    - speaker MISMATCH; treat as possible cloned voice
   *   NEUTRAL - no enrollment; score on content (+ spoof_strong) only
   * Fusion respects this before any weighted formula.
   */
  risk_hint: RiskHint;
  transcript: string;
}

export interface LLMRiskScore {
  llm_risk_score: number;
  detected_signals: string[];
}

export interface FusionResult {
  final_risk_score: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  reason: string;
}

export interface ChunkAnalysis {
  session_id: string;
  chunk_id: number;
  ml: MLAnalyzeResponse;
  llm: LLMRiskScore;
  fusion: FusionResult;
  latency_ms: number;
  timestamp: string;
  /**
   * Session-level "sticky" risk. Once a chunk escalates to MEDIUM/HIGH the peak
   * is carried forward across later chunks (including silent ones) so a single
   * fraud utterance is not overwritten by trailing silence.
   *
   * ONLY fraud content (llm.detected_signals non-empty) can raise the session
   * peak. Speaker MISMATCH and spoof_strong are too noisy on browser audio
   * without reliable enrollment, so acoustic signals alone keep the session at
   * LOW.
   *
   * Decay: a clean OR silent chunk (no fraud words spoken) advances the streak;
   * after 5 consecutive such chunks an elevated session drops one level
   * (HIGH -> MEDIUM -> LOW).
   */
  session_risk_score: number;
  session_risk_level: FusionResult['risk_level'];
  session_reason: string;
  /** Union of every detected_signals value seen so far in this session. */
  session_signals: string[];
}
