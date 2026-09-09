import Anthropic from '@anthropic-ai/sdk';
import type { LLMRiskScore } from '../types/contracts';

/**
 * Model fallback chain, tried in order until one is accepted by the API.
 *
 * `claude-3-5-haiku-20241022` (the previous value here) returns 404 because it
 * was retired. `claude-haiku-4-5` is the current fast/cheap Haiku model and is
 * what actually works today. The two dated 3.x IDs below are kept only because
 * the fix request asked for them as fallbacks -- both are retired/deprecated and
 * will very likely also 404, so `claude-haiku-4-5` must stay first.
 */
const MODEL_CANDIDATES = [
  'claude-haiku-4-5',
  'claude-3-5-haiku-latest',
  'claude-3-haiku-20240307',
] as const;

const MAX_TOKENS = 200;
const API_TIMEOUT_MS = 5_000;

const SYSTEM_PROMPT =
  'You are a fraud detection assistant analyzing a live phone call transcript ' +
  'chunk for signs of voice-clone impersonation fraud. Look for: urgency ' +
  'language, OTP or verification code requests, money transfer requests, ' +
  'impersonation of bank officials or authorities, threats, unusual pressure ' +
  'tactics. Respond ONLY with a JSON object in this exact format, no extra ' +
  'text: { "risk_score": 0.0, "signals": [] } where risk_score is 0.0 to 1.0 ' +
  'and signals is an array of short signal names like otp_request, urgency, ' +
  'impersonation, money_request, threat, secrecy.';

const SAFE_FALLBACK: LLMRiskScore = { llm_risk_score: 0, detected_signals: [] };

// Reads ANTHROPIC_API_KEY from the environment (loaded via dotenv in src/index.ts).
const client = new Anthropic();

/** True when the error means "this model id is not valid" -> try the next candidate. */
function isModelNotFound(err: unknown): boolean {
  if (err instanceof Anthropic.NotFoundError) return true;
  if (err instanceof Anthropic.APIError && err.status === 404) return true;
  return false;
}

/** Turn Claude's JSON reply text into an LLMRiskScore. Throws on unparseable input. */
function parseRiskScore(raw: string): LLMRiskScore {
  // Be tolerant of a stray markdown fence or surrounding prose.
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  const jsonSlice = start !== -1 && end !== -1 ? raw.slice(start, end + 1) : raw;

  const parsed = JSON.parse(jsonSlice) as { risk_score?: unknown; signals?: unknown };

  const llm_risk_score =
    typeof parsed.risk_score === 'number' && Number.isFinite(parsed.risk_score)
      ? Math.min(Math.max(parsed.risk_score, 0), 1)
      : 0;
  const detected_signals = Array.isArray(parsed.signals)
    ? parsed.signals.filter((s): s is string => typeof s === 'string')
    : [];

  return { llm_risk_score, detected_signals };
}

/**
 * Real Claude-powered fraud scorer. Sends the transcript chunk to the Anthropic
 * API and asks Claude to rate voice-clone impersonation fraud risk and name the
 * signals it saw.
 *
 * Walks MODEL_CANDIDATES in order: a 404 (model not found) moves on to the next
 * id; any other failure (network / rate-limit / timeout / parsing) is logged and
 * the safe fallback ({ llm_risk_score: 0, detected_signals: [] }) is returned so
 * the real-time pipeline keeps running. Never throws.
 *
 * Logs the exact model name used on every request so we can verify which id the
 * API actually accepts.
 */
export async function scoreTranscript(transcript: string): Promise<LLMRiskScore> {
  const text = (transcript ?? '').trim();

  if (text.length < 5) {
    console.log(
      `[claudeScorer] (real Claude API) transcript too short, skipping API call | ` +
        `transcript=${JSON.stringify(text)} | llm_risk_score=0`
    );
    return { ...SAFE_FALLBACK };
  }

  for (const model of MODEL_CANDIDATES) {
    console.log(`[claudeScorer] (real Claude API) requesting with model=${model}`);

    try {
      const response = await client.messages.create(
        {
          model,
          max_tokens: MAX_TOKENS,
          system: SYSTEM_PROMPT,
          messages: [{ role: 'user', content: text }],
        },
        { timeout: API_TIMEOUT_MS }
      );

      const raw = response.content
        .filter((block): block is Anthropic.TextBlock => block.type === 'text')
        .map((block) => block.text)
        .join('')
        .trim();

      const { llm_risk_score, detected_signals } = parseRiskScore(raw);

      console.log(
        `[claudeScorer] (real Claude API) success model=${model} | ` +
          `transcript=${JSON.stringify(text)} | ` +
          `raw=${JSON.stringify(raw)} | ` +
          `signals=[${detected_signals.join(', ')}] | ` +
          `llm_risk_score=${llm_risk_score}`
      );

      return { llm_risk_score, detected_signals };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);

      if (isModelNotFound(err)) {
        console.error(
          `[claudeScorer] (real Claude API) model=${model} rejected as not found (404), ` +
            `trying next candidate | error=${message}`
        );
        continue;
      }

      console.error(
        `[claudeScorer] (real Claude API) call failed with model=${model}, returning safe fallback | ` +
          `transcript=${JSON.stringify(text)} | error=${message}`
      );
      return { ...SAFE_FALLBACK };
    }
  }

  console.error(
    `[claudeScorer] (real Claude API) all model candidates exhausted ` +
      `(${MODEL_CANDIDATES.join(', ')}), returning safe fallback`
  );
  return { ...SAFE_FALLBACK };
}
