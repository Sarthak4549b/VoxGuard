import { logger } from './logger';

/**
 * Tracks per-chunk processing latency for a single session and exposes
 * running summary stats.
 */
export class LatencyTracker {
  private readonly session_id: string;
  private readonly starts = new Map<number, number>();
  private readonly durations: number[] = [];

  constructor(session_id: string) {
    this.session_id = session_id;
  }

  start(chunk_id: number): void {
    this.starts.set(chunk_id, Date.now());
  }

  end(chunk_id: number): number {
    const started = this.starts.get(chunk_id);
    const elapsed = started === undefined ? 0 : Date.now() - started;
    this.starts.delete(chunk_id);
    this.durations.push(elapsed);
    logger.info(
      `Session ${this.session_id} chunk ${chunk_id} latency: ${elapsed}ms`
    );
    return elapsed;
  }

  getSummary(): { avg_ms: number; max_ms: number; count: number } {
    const count = this.durations.length;
    if (count === 0) {
      return { avg_ms: 0, max_ms: 0, count: 0 };
    }
    const sum = this.durations.reduce((a, b) => a + b, 0);
    return {
      avg_ms: sum / count,
      max_ms: Math.max(...this.durations),
      count,
    };
  }
}
