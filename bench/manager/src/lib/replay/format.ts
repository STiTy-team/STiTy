const LANG_HUE: Record<string, string> = {
  en: "#6080C8",
  ko: "#7A9030",
  ja: "#C87060",
  zh: "#9060C8",
  es: "#C8A030",
  fr: "#308898",
  id: "#30A070",
  vi: "#B85050",
  th: "#5080C0",
  de: "#C09050",
  ar: "#C56BA8",
}
export const langHue = (lang: string | undefined) => LANG_HUE[lang ?? ""] ?? "#8B95A1"

export const REASON_COLOR: Record<string, string> = {
  seg: "#6080C8",
  vad: "#7A9030",
  dot: "#C8A030",
  always: "#9060C8",
  finish: "#8B95A1",
}
export const reasonColor = (reason: string | undefined) => REASON_COLOR[reason ?? ""] ?? REASON_COLOR.finish

export const LANE_COLORS = ["#4C9A5A", "#D08A2A", "#C0554F", "#7B60C0", "#3F8A9C", "#A0662A", "#6D7FA8", "#8A8F84"]

export function stampText(stamp: string) {
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/.exec(stamp)
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` : stamp
}

/** Run stamps (`20261005T143014`) are UTC. */
export function stampIso(stamp: string) {
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/.exec(stamp)
  return m ? `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}Z` : null
}

export const seconds = (v: number, digits = 2) => `${v.toFixed(digits)}s`

export const ITEM_METRICS: Record<string, { label: string; format: (v: number) => string }> = {
  wer: { label: "WER", format: (v) => v.toFixed(2) },
  cer: { label: "CER", format: (v) => v.toFixed(2) },
  sentence_bleu: { label: "BLEU", format: (v) => v.toFixed(1) },
  avg_fsl_sec: { label: "FSL", format: (v) => seconds(v) },
  laal_ms: { label: "LAAL", format: (v) => seconds(v / 1000) },
  yaal_ms: { label: "YAAL", format: (v) => seconds(v / 1000) },
  longyaal_ms: { label: "LongYAAL", format: (v) => seconds(v / 1000) },
  token_emission_ms: { label: "Token emission", format: (v) => seconds(v / 1000) },
  n_segments: { label: "Segments", format: (v) => String(v) },
}

const OTHER_GROUP = "Other"
const METRIC_GROUPS: [string, (key: string) => boolean][] = [
  ["Transcription accuracy", (k) => /^(wer|cer)/.test(k)],
  ["Transcription latency", (k) => /(fsl|token_emission)/.test(k)],
  ["Translation accuracy", (k) => /^(bleu|sentence_bleu|comet)/.test(k)],
  ["Translation latency", (k) => /(laal|yaal)/.test(k)],
  ["Language detection", (k) => /^(lang_detect|confusion)/.test(k)],
  ["Segmentation", (k) => /(segment|commit_reasons)/.test(k)],
]

export const groupOf = (key: string) => METRIC_GROUPS.find(([, test]) => test(key))?.[0] ?? OTHER_GROUP

export function byGroup(keys: string[]): [string, string[]][] {
  const buckets = new Map<string, string[]>([...METRIC_GROUPS.map(([title]) => title), OTHER_GROUP].map((t) => [t, []]))
  for (const key of keys) buckets.get(groupOf(key))!.push(key)
  return [...buckets].filter(([, ks]) => ks.length)
}

const ACRONYMS: Record<string, string> = {
  wer: "WER",
  cer: "CER",
  bleu: "BLEU",
  comet: "COMET",
  fsl: "FSL",
  laal: "LAAL",
  yaal: "YAAL",
  longyaal: "LongYAAL",
  ca: "CA",
  vad: "VAD",
  gpu: "GPU",
  url: "URL",
  p50: "p50",
  p90: "p90",
}
const METRIC_LABELS: Record<string, string> = {
  avg_fsl_sec: "Avg FSL",
  token_emission: "Token emission",
  lang_detect_accuracy: "Lang detect accuracy",
  segments_per_item: "Segments / item",
  commit_reasons: "Commit reasons",
  confusion: "Language confusion",
  wer_by_lang: "WER by language",
  cer_by_lang: "CER by language",
  bleu_by_pair: "BLEU by pair",
  comet_by_pair: "COMET by pair",
  wer_scored_only: "WER (scored only)",
  comet_model: "COMET model",
}
const PERCENT_KEYS = new Set(["lang_detect_accuracy", "commit_reasons"])

export function labelOf(key: string): string | null {
  const bare = key.replace(/_(ms|sec)$/, "")
  const named = METRIC_LABELS[key] ?? METRIC_LABELS[bare]
  if (named) return named
  const words = bare.split("_")
  if (words[0] === "token" && words[1] === "emission")
    return ["Token emission", ...words.slice(2).map((w) => ACRONYMS[w] ?? w)].join(" ")
  return words.every((w) => ACRONYMS[w]) ? words.map((w) => ACRONYMS[w]).join(" ") : null
}

export function formatValue(key: string, v: unknown, parent?: string): string {
  if (v == null || v === "") return "—"
  if (Array.isArray(v)) return v.map((x) => formatValue(key, x, parent)).join(", ")
  if (typeof v === "boolean") return v ? "yes" : "no"
  if (typeof v !== "number") return String(v)
  if (/_ms$/.test(key)) return seconds(v / 1000)
  if (/_sec$/.test(key)) return seconds(v)
  if (PERCENT_KEYS.has(key) || (parent && PERCENT_KEYS.has(parent))) return `${(100 * v).toFixed(1)}%`
  if (/^bleu/.test(key) || /^bleu/.test(parent ?? "")) return v.toFixed(1)
  if (Number.isInteger(v)) return v.toLocaleString("en-US")
  return v.toFixed(3)
}

export const isRecord = (v: unknown): v is Record<string, unknown> =>
  v != null && typeof v === "object" && !Array.isArray(v)

export function leaves(obj: Record<string, unknown>, prefix = ""): [string, unknown][] {
  return Object.entries(obj).flatMap(([k, v]): [string, unknown][] => {
    const key = prefix ? `${prefix}.${k}` : k
    return isRecord(v) ? leaves(v, key) : [[key, v]]
  })
}
