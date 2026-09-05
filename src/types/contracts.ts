export interface MLAnalyzeResponse {
  speaker_similarity: number;
  speaker_status: 'MATCH' | 'MISMATCH' | 'UNKNOWN';
  spoof_score: number;
  spoof_label: 'GENUINE' | 'SPOOF';
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
}
