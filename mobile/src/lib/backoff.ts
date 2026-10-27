export const exponentialBackoff = (attempt: number, baseMs: number, maxMs: number): number =>
  Math.min(maxMs, baseMs * 2 ** Math.max(0, attempt));
