import axios from 'axios';
import type { MLAnalyzeResponse } from '../types/contracts';
import { logger } from '../utils/logger';

const ML_API_URL = process.env.ML_API_URL || 'http://localhost:8000';

/**
 * Calls the ML service /analyze endpoint. On any failure (timeout, connection
 * refused, bad response) it returns a safe fallback so the pipeline degrades
 * gracefully instead of crashing the session.
 */
export async function callMLAnalyze(
  audioBase64: string,
  transcript: string
): Promise<MLAnalyzeResponse> {
  try {
    const { data } = await axios.post<MLAnalyzeResponse>(
      `${ML_API_URL}/analyze`,
      { audio_base64: audioBase64, transcript },
      { timeout: 8000 }
    );
    console.log(
      `[mlBridge] /analyze ok | risk_hint=${data.risk_hint} ` +
        `spoof_strong=${data.spoof_strong} ` +
        `transcript=${JSON.stringify(data.transcript ?? '')}`
    );
    return data;
  } catch (err) {
    logger.error(err, 'callMLAnalyze failed - returning safe fallback');
    console.log(
      '[mlBridge] /analyze FAILED - returning safe fallback ' +
        `(transcript falls back to client-supplied ${JSON.stringify(transcript)})`
    );
    return {
      speaker_similarity: 0,
      speaker_status: 'UNKNOWN',
      spoof_score: 0.5,
      spoof_label: 'GENUINE',
      spoof_strong: false,
      risk_hint: 'NEUTRAL',
      transcript,
    };
  }
}
