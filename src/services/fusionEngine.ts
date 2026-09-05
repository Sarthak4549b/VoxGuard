import type { MLAnalyzeResponse, LLMRiskScore, FusionResult } from '../types/contracts';

/**
 * Combines the acoustic (ML) and semantic (LLM) risk signals into a single
 * final risk score, level, and human-readable reason.
 */
export function fuseScores(
  ml: MLAnalyzeResponse,
  llm: LLMRiskScore
): FusionResult {
  // Step 1 - base weighted score
  const base_score = 0.6 * ml.spoof_score + 0.4 * llm.llm_risk_score;

  // Step 2 - speaker mismatch override
  let final_risk_score: number;
  let reason: string;
  if (ml.speaker_status === 'MISMATCH' && ml.spoof_score > 0.4) {
    final_risk_score = Math.max(base_score, 0.75);
    reason = 'Speaker mismatch detected with elevated spoof score';
  } else {
    final_risk_score = base_score;
    reason = 'Weighted fusion of acoustic and LLM scores';
  }

  // Step 3 - risk level
  let risk_level: FusionResult['risk_level'];
  if (final_risk_score >= 0.7) {
    risk_level = 'HIGH';
  } else if (final_risk_score >= 0.4) {
    risk_level = 'MEDIUM';
  } else {
    risk_level = 'LOW';
  }

  // Step 4 - append detected signals to reason
  if (llm.detected_signals.length > 0) {
    reason += `. Signals: ${llm.detected_signals.join(', ')}`;
  }

  return { final_risk_score, risk_level, reason };
}
