// Shared by the dashboard and the session replay: DOM and HTML helpers, the
// language colours, and how a score is grouped, labelled and formatted.

// HomeScreen.tsx LANG_COLORS, verbatim -- the phone panel shows the app's colours.
const LANG = {
  en: {bubble:"#6080C8", avatar:"#9BB4D4"}, ko: {bubble:"#7A9030", avatar:"#A8B870"},
  ja: {bubble:"#C87060", avatar:"#E0A898"}, zh: {bubble:"#9060C8", avatar:"#B898E0"},
  es: {bubble:"#C8A030", avatar:"#E0C878"}, fr: {bubble:"#308898", avatar:"#78B8C8"},
  id: {bubble:"#30A070", avatar:"#70C0A0"}, vi: {bubble:"#B85050", avatar:"#E09080"},
  th: {bubble:"#5080C0", avatar:"#88B0E0"}, de: {bubble:"#C09050", avatar:"#E0C080"},
};
const langColor = c => LANG[c] || {bubble:"#909090", avatar:"#B8B8B8"};

const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

const $ = id => document.getElementById(id);

// Every score belongs to one of these, by its key (bench/README.md's metric table).
const METRIC_GROUPS = [
  ["Transcription accuracy", k => /^(wer|cer)/.test(k)],
  ["Transcription latency", k => /(fsl|token_emission)/.test(k)],
  ["Translation accuracy", k => /^(bleu|sentence_bleu|comet)/.test(k)],
  ["Translation latency", k => /(laal|yaal)/.test(k)],
  ["Language detection", k => /^(lang_detect|confusion)/.test(k)],
  ["Segmentation", k => /(segment|commit_reasons)/.test(k)],
];
function groupOf(key) {
  const hit = METRIC_GROUPS.find(([, test]) => test(key));
  return hit ? hit[0] : "Other";
}
// [[title, [key, ...]], ...] in METRIC_GROUPS order, empty groups left out.
function byGroup(keys) {
  const order = METRIC_GROUPS.map(([t]) => t).concat("Other");
  const buckets = new Map(order.map(t => [t, []]));
  for (const k of keys) buckets.get(groupOf(k)).push(k);
  return [...buckets].filter(([, ks]) => ks.length);
}


// Run-level numbers: how to label and format a key, falling back to its own name.
const ACRONYMS = {wer: "WER", cer: "CER", bleu: "BLEU", comet: "COMET", fsl: "FSL",
                  laal: "LAAL", yaal: "YAAL", longyaal: "LongYAAL", ca: "CA", vad: "VAD",
                  gpu: "GPU", url: "URL", p50: "p50", p90: "p90"};
const PERCENT_KEYS = new Set(["lang_detect_accuracy", "commit_reasons"]);
function labelOf(key) {
  return key.replace(/_(ms|sec)$/, "").split("_")
    .map((w, i) => ACRONYMS[w] || (i === 0 ? w[0].toUpperCase() + w.slice(1) : w)).join(" ");
}
function formatValue(key, v, parent) {
  if (v == null || v === "") return "—";
  if (Array.isArray(v)) return v.map(x => formatValue(key, x, parent)).join(", ");
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v !== "number") return String(v);
  if (/_ms$/.test(key)) return (v / 1000).toFixed(2) + "s";
  if (/_sec$/.test(key)) return v.toFixed(2) + "s";
  if (PERCENT_KEYS.has(key) || PERCENT_KEYS.has(parent)) return (100 * v).toFixed(1) + "%";
  if (/^bleu/.test(key) || /^bleu/.test(parent || "")) return v.toFixed(1);
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(3);
}
// Nested keys below a group's own level become dotted labels, so every leaf is a tile.
function leaves(obj, prefix = "") {
  const out = [];
  for (const [k, v] of Object.entries(obj || {})) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) out.push(...leaves(v, key));
    else out.push([key, v]);
  }
  return out;
}
function tilesHtml(pairs, {parent = null, labels = true} = {}) {
  return `<div class="stat-tiles small">` + pairs.map(([key, v]) => {
    const last = key.split(".").at(-1);
    const shown = formatValue(last, v, parent);
    const text = typeof v !== "number";
    return `<div class="stat-tile"><span class="eyebrow">${esc(labels ? labelOf(last) : key)}</span>` +
      `<span class="stat-value${text ? " text" : ""}" title="${esc(key)}">${esc(shown)}</span></div>`;
  }).join("") + `</div>`;
}
function groupHtml(title, sub, body) {
  return `<div class="kv-group"><div class="kv-group-title">${esc(title)}` +
    (sub ? ` <span class="dim">${esc(sub)}</span>` : "") + `</div>${body}</div>`;
}


// The scores the dashboard ranks runs on, keyed like summary.json's metrics.
// `better` says which way is good; `item` names the per-item key whose
// distribution the charts draw (items.jsonl, or comet_scores.jsonl for COMET).
const SCORES = {
  wer: {label: "WER", better: "lower", item: "wer"},
  cer: {label: "CER", better: "lower", item: "cer"},
  avg_fsl_sec: {label: "FSL", better: "lower", item: "avg_fsl_sec"},
  token_emission_ms: {label: "Token emission", better: "lower", item: "token_emission_ms"},
  token_emission_ca_ms: {label: "Token emission CA", better: "lower", item: "token_emission_ca_ms"},
  bleu: {label: "BLEU", better: "higher", item: "sentence_bleu",
         note: "table: corpus BLEU · chart: sentence BLEU per item"},
  comet: {label: "COMET", better: "higher", item: "comet", note: "chart: one point per sentence"},
  laal_ms: {label: "LAAL", better: "lower", item: "laal_ms"},
  laal_ca_ms: {label: "LAAL CA", better: "lower", item: "laal_ca_ms"},
  yaal_ms: {label: "YAAL", better: "lower", item: "yaal_ms"},
  yaal_ca_ms: {label: "YAAL CA", better: "lower", item: "yaal_ca_ms"},
  longyaal_ms: {label: "LongYAAL", better: "lower", item: "longyaal_ms"},
  longyaal_ca_ms: {label: "LongYAAL CA", better: "lower", item: "longyaal_ca_ms"},
  lang_detect_accuracy: {label: "Lang detect", better: "higher"},
  segments_per_item: {label: "Segments / item", better: null, item: "n_segments"},
};
// A per-item value in the unit the table shows (milliseconds read as seconds).
const displayScale = key => /_ms$/.test(key) ? 0.001 : 1;
function formatScore(key, v) {
  if (v == null || !Number.isFinite(v)) return "—";
  if (/_ms$/.test(key)) return (v / 1000).toFixed(2) + "s";
  if (/_sec$/.test(key)) return v.toFixed(2) + "s";
  if (key === "lang_detect_accuracy") return (100 * v).toFixed(1) + "%";
  if (/bleu/.test(key)) return v.toFixed(1);
  if (key === "comet") return v.toFixed(3);
  if (/segment/.test(key)) return v.toFixed(1);
  return v.toFixed(3);
}
// -1 when `a` is the better value, 1 when `b` is, 0 when the score has no direction.
function compareScores(key, a, b) {
  const better = (SCORES[key] || {}).better;
  if (!better || a == null || b == null) return 0;
  return better === "lower" ? a - b : b - a;
}
