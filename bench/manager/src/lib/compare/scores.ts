export type Better = "lower" | "higher" | null

type Score = { label: string; better: Better; item?: string; note?: string }

/** The scores runs are ranked on, keyed like summary.json's metrics; `item` names the per-item distribution key. */
export const SCORES: Record<string, Score> = {
  wer: { label: "WER", better: "lower", item: "wer" },
  cer: { label: "CER", better: "lower", item: "cer" },
  avg_fsl_sec: { label: "FSL", better: "lower", item: "avg_fsl_sec" },
  token_emission_ms: { label: "Token emission", better: "lower", item: "token_emission_ms" },
  token_emission_ca_ms: { label: "Token emission CA", better: "lower", item: "token_emission_ca_ms" },
  bleu: { label: "BLEU", better: "higher", item: "sentence_bleu", note: "표는 corpus BLEU, 차트는 항목별 sentence BLEU예요" },
  comet: { label: "COMET", better: "higher", item: "comet", note: "차트의 점 하나가 문장 하나예요" },
  laal_ms: { label: "LAAL", better: "lower", item: "laal_ms" },
  laal_ca_ms: { label: "LAAL CA", better: "lower", item: "laal_ca_ms" },
  yaal_ms: { label: "YAAL", better: "lower", item: "yaal_ms" },
  yaal_ca_ms: { label: "YAAL CA", better: "lower", item: "yaal_ca_ms" },
  longyaal_ms: { label: "LongYAAL", better: "lower", item: "longyaal_ms" },
  longyaal_ca_ms: { label: "LongYAAL CA", better: "lower", item: "longyaal_ca_ms" },
  lang_detect_accuracy: { label: "Lang detect", better: "higher" },
  segments_per_item: { label: "Segments / item", better: null, item: "n_segments" },
}

export const BETTER_TEXT: Record<string, string> = { lower: "낮을수록 좋음", higher: "높을수록 좋음" }

export const unitOf = (key: string) => (/_(ms|sec)$/.test(key) ? "s" : key === "lang_detect_accuracy" ? "%" : "")

/** A per-item value in the unit the table shows (milliseconds read as seconds). */
export const displayScale = (key: string) => (/_ms$/.test(key) ? 0.001 : 1)

export const axisText = (key: string) => `${SCORES[key].label}${unitOf(key) ? ` (${unitOf(key)})` : ""}`

export function formatScore(key: string, v: number | null | undefined, { unit = true } = {}) {
  if (v == null || !Number.isFinite(v)) return "—"
  const u = unit ? unitOf(key) : ""
  if (/_ms$/.test(key)) return (v / 1000).toFixed(2) + u
  if (/_sec$/.test(key)) return v.toFixed(2) + u
  if (key === "lang_detect_accuracy") return (100 * v).toFixed(1) + u
  if (/bleu/.test(key)) return v.toFixed(1)
  if (key === "comet") return v.toFixed(3)
  if (/segment/.test(key)) return v.toFixed(1)
  return v.toFixed(3)
}

/** Negative when `a` is the better value, positive when `b` is, 0 when the score has no direction. */
export function compareScores(key: string, a: number | null, b: number | null) {
  const better = SCORES[key]?.better
  if (!better || a == null || b == null) return 0
  return better === "lower" ? a - b : b - a
}

export function metric(metrics: Record<string, unknown>, key: string): number | null {
  const v = metrics[key]
  return typeof v === "number" && Number.isFinite(v) ? v : null
}

export function niceTicks(lo: number, hi: number, count = 6) {
  if (!(hi > lo)) hi = lo + 1
  const raw = (hi - lo) / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => raw <= s) ?? 10 * mag
  const out: number[] = []
  for (let t = Math.floor(lo / step) * step; t <= hi + step * 1e-6; t += step) out.push(+t.toFixed(10))
  if (out[out.length - 1] < hi) out.push(+(out[out.length - 1] + step).toFixed(10))
  return out
}

export const tickText = (key: string, t: number) => (key === "comet" ? t.toFixed(2) : String(+t.toFixed(3)))

export const clip = (s: string, n: number) => (s.length > n ? `${s.slice(0, n - 1)}…` : s)
