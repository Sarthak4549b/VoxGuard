import type { MLAnalyzeResponse, LLMRiskScore, FusionResult } from '../types/contracts';

/**
 * Combines the acoustic (ML) and semantic (LLM) risk signals into a single
 * final risk score, level, and human-readable reason.
 *
 * The ML server's `risk_hint` is authoritative and is applied BEFORE any
 * weighted formula. The raw `spoof_score` is never blended in directly - it
 * saturates near 1.0 on compressed browser audio even for genuine speakers, so
 * the only acoustic signal fusion trusts is the calibrated `spoof_strong` flag.
 *
 * Strict priority:
 *   1. risk_hint === 'LOW'     -> verified genuine speaker; content only. A low
 *                                 fraud score stays safe (x0.4), but clear
 *                                 fraud content (llm_risk_score >= 0.6) is
 *                                 still allowed to escalate all the way to HIGH.
 *   2. risk_hint === 'HIGH'    -> speaker mismatch; floor the score at 0.75.
 *   3. risk_hint === 'NEUTRAL' -> no enrollment; score on content, nudged up
 *                                 only when spoof_strong is set.
 */
export function fuseScores(
  ml: MLAnalyzeResponse,
  llm: LLMRiskScore
): FusionResult {
  const llmRisk = llm.llm_risk_score;
  const spoofStrong = ml.spoof_strong === true;

  let final_risk_score: number;
  let reason: string;

  switch (ml.risk_hint) {
    case 'LOW':
      // Only the LLM content matters - spoof_score is ignored entirely.
      if (llmRisk >= 0.6) {
        // Verified speaker, but the words are clearly fraudulent: let it
        // escalate (can reach HIGH). No cap.
        final_risk_score = llmRisk;
        reason = 'Genuine speaker but fraudulent content detected';
      } else {
        // Normal safe case for a verified speaker.
        final_risk_score = llmRisk * 0.4;
        reason = 'Verified genuine speaker';
      }
      break;

    case 'HIGH':
      final_risk_score = Math.max(0.75, llmRisk);
      reason = 'Speaker mismatch - possible cloned voice';
      break;

    case 'NEUTRAL':
    default:
      final_risk_score = llmRisk * 0.6 + (spoofStrong ? 0.4 : 0);
      reason = 'No enrolled speaker - scoring on content';
      break;
  }

  // Risk level straight off the final score.
  let risk_level: FusionResult['risk_level'];
  if (final_risk_score >= 0.7) {
    risk_level = 'HIGH';
  } else if (final_risk_score >= 0.4) {
    risk_level = 'MEDIUM';
  } else {
    risk_level = 'LOW';
  }

  // Append any detected fraud signals for context (does not change the score).
  if (llm.detected_signals.length > 0) {
    reason += `. Signals: ${llm.detected_signals.join(', ')}`;
  }

  console.log(
    `[fusionEngine] risk_hint=${ml.risk_hint} llm_risk_score=${llmRisk} ` +
      `spoof_strong=${spoofStrong} -> final_risk_score=${final_risk_score} ` +
      `risk_level=${risk_level} reason="${reason}"`
  );

  return { final_risk_score, risk_level, reason };
}
