import type { LLMRiskScore } from '../types/contracts';

interface Rule {
  keywords: string[];
  signal: string;
  weight: number;
}

const RULES: Rule[] = [
  { keywords: ['otp', 'one time password'], signal: 'otp_request', weight: 0.4 },
  {
    keywords: ['transfer', 'send money', 'deposit'],
    signal: 'money_request',
    weight: 0.35,
  },
  {
    keywords: ['urgent', 'immediately', 'right now'],
    signal: 'urgency',
    weight: 0.25,
  },
  {
    keywords: ['bank manager', 'bank official', 'rbi', 'police'],
    signal: 'impersonation',
    weight: 0.35,
  },
  {
    keywords: ['dont tell', "don't tell", 'keep secret'],
    signal: 'secrecy',
    weight: 0.3,
  },
  {
    keywords: ['blocked', 'suspended', 'arrested'],
    signal: 'threat',
    weight: 0.2,
  },
];

/**
 * Local rule-based fraud scorer. Scans the transcript for known voice-clone
 * scam signal keywords and accumulates a risk score. No external API call.
 */
export async function scoreTranscript(transcript: string): Promise<LLMRiskScore> {
  const text = transcript.toLowerCase();

  const detected_signals: string[] = [];
  let risk = 0;

  for (const rule of RULES) {
    if (rule.keywords.some((kw) => text.includes(kw))) {
      detected_signals.push(rule.signal);
      risk += rule.weight;
    }
  }

  const llm_risk_score = Math.min(risk, 1.0);

  return { llm_risk_score, detected_signals };
}
