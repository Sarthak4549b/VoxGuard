import type { LLMRiskScore } from '../types/contracts';

interface Rule {
  keywords: string[];
  signal: string;
  weight: number;
}

// All keywords are lower-case; matching lower-cases the transcript first, so
// detection is fully case-insensitive.
const RULES: Rule[] = [
  {
    keywords: [
      'otp',
      'one time password',
      'one-time password',
      'share code',
      'share the code',
      'pin number',
      'pin code',
      'verification code',
    ],
    signal: 'otp_request',
    weight: 0.4,
  },
  {
    keywords: [
      'transfer',
      'transfer money',
      'send money',
      'deposit',
      'wire transfer',
      'make a payment',
    ],
    signal: 'money_request',
    weight: 0.35,
  },
  {
    keywords: ['urgent', 'immediately', 'right now', 'as soon as possible', 'act now'],
    signal: 'urgency',
    weight: 0.25,
  },
  {
    keywords: [
      'bank manager',
      'bank official',
      'bank account',
      'verify account',
      'verify your account',
      'account verification',
      'rbi',
      'police',
    ],
    signal: 'impersonation',
    weight: 0.35,
  },
  {
    keywords: ['dont tell', "don't tell", 'keep secret', 'keep this secret', 'between us'],
    signal: 'secrecy',
    weight: 0.3,
  },
  {
    keywords: [
      'blocked',
      'suspended',
      'arrested',
      'account blocked',
      'account suspended',
      'account will be closed',
    ],
    signal: 'threat',
    weight: 0.2,
  },
];

/**
 * Local rule-based fraud scorer. Scans the transcript for known voice-clone
 * scam signal keywords (case-insensitive) and accumulates a risk score.
 * No external API call.
 *
 * Logs the transcript it receives and the score it returns so the live flow
 * can be traced in the backend terminal.
 */
export async function scoreTranscript(transcript: string): Promise<LLMRiskScore> {
  const text = (transcript ?? '').toLowerCase();

  const detected_signals: string[] = [];
  const matched_keywords: string[] = [];
  let risk = 0;

  for (const rule of RULES) {
    const hits = rule.keywords.filter((kw) => text.includes(kw));
    if (hits.length > 0) {
      detected_signals.push(rule.signal);
      matched_keywords.push(...hits);
      risk += rule.weight;
    }
  }

  const llm_risk_score = Math.min(risk, 1.0);

  console.log(
    `[claudeScorer] transcript=${JSON.stringify(text)} | ` +
      `matched=[${matched_keywords.join(', ')}] | ` +
      `signals=[${detected_signals.join(', ')}] | ` +
      `llm_risk_score=${llm_risk_score}`
  );

  return { llm_risk_score, detected_signals };
}
