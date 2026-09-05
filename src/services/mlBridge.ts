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
    return data;
  } catch (err) {
    logger.error(err, 'callMLAnalyze failed - returning safe fallback');
    return {
      speaker_similarity: 0,
      speaker_status: 'UNKNOWN',
      spoof_score: 0.5,
      spoof_label: 'GENUINE',
      transcript,
    };
  }
}
