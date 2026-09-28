// Shared by the dashboard and the session replay: DOM and HTML helpers, the
// language hues, and how a score is grouped, labelled and formatted.
// UI copy is Korean 해요체 around English domain terms (run, item, commit, metric
// names...), so the page reads with the same words as the code and the docs.

// DESIGN.md lang-* hues: marks only (dots, bands), never a fill behind text.
const LANG_HUE = {
  en: "#6080C8", ko: "#7A9030", ja: "#C87060", zh: "#9060C8", es: "#C8A030", fr: "#308898",
  id: "#30A070", vi: "#B85050", th: "#5080C0", de: "#C09050", ar: "#C56BA8",
};
const langHue = c => LANG_HUE[c] || "#8B95A1";

const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

const $ = id => document.getElementById(id);

// `20260927T194407` → `2026. 9. 27. 19:44`
function stampText(stamp) {
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/.exec(stamp || "");
  return m ? `${m[1]}. ${+m[2]}. ${+m[3]}. ${m[4]}:${m[5]}` : (stamp || "");
}

// A config's `meta` block, as bench/replay/runs.py::run_meta sends it.
const tagChips = meta => (meta && meta.tags || []).map(t => `<span class="tag">${esc(t)}</span>`).join("");
function metaLine(kind, meta) {
  if (!meta) return "";
  return `<div class="meta-line"><span class="kind">${esc(kind)}</span> ` +
    `<span class="ref" title="config hash ${esc(meta.hash || "기록 없음")}">${esc(meta.ref)}</span> ` +
    tagChips(meta) +
    (meta.description ? ` <span class="desc">${esc(meta.description)}</span>` : "") + `</div>`;
}

// Every score belongs to one of these, by its key (bench/README.md's metric table).
const OTHER_GROUP = "기타";
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
  return hit ? hit[0] : OTHER_GROUP;
}
// [[title, [key, ...]], ...] in METRIC_GROUPS order, empty groups left out.
function byGroup(keys) {
  const order = METRIC_GROUPS.map(([t]) => t).concat(OTHER_GROUP);
  const buckets = new Map(order.map(t => [t, []]));
  for (const k of keys) buckets.get(groupOf(k)).push(k);
  return [...buckets].filter(([, ks]) => ks.length);
}


// Run-level numbers: how to label and format a key, falling back to its own name.
const ACRONYMS = {wer: "WER", cer: "CER", bleu: "BLEU", comet: "COMET", fsl: "FSL",
                  laal: "LAAL", yaal: "YAAL", longyaal: "LongYAAL", ca: "CA", vad: "VAD",
                  gpu: "GPU", url: "URL", p50: "p50", p90: "p90"};
const METRIC_LABELS = {
  avg_fsl_sec: "Avg FSL", token_emission: "Token emission", lang_detect_accuracy: "Lang detect accuracy",
  segments_per_item: "Segments / item", commit_reasons: "Commit reasons", confusion: "Language confusion",
  wer_by_lang: "WER by language", cer_by_lang: "CER by language", bleu_by_pair: "BLEU by pair",
  comet_by_pair: "COMET by pair", wer_scored_only: "WER (scored only)", comet_model: "COMET model",
};
const PERCENT_KEYS = new Set(["lang_detect_accuracy", "commit_reasons"]);
function labelOf(key) {
  const bare = key.replace(/_(ms|sec)$/, "");
  if (METRIC_LABELS[key] || METRIC_LABELS[bare]) return METRIC_LABELS[key] || METRIC_LABELS[bare];
  const words = bare.split("_");
  if (words[0] === "token" && words[1] === "emission")
    return ["Token emission", ...words.slice(2).map(w => ACRONYMS[w] || w)].join(" ");
  return words.every(w => ACRONYMS[w]) ? words.map(w => ACRONYMS[w]).join(" ") : null;
}
function formatValue(key, v, parent) {
  if (v == null || v === "") return "—";
  if (Array.isArray(v)) return v.map(x => formatValue(key, x, parent)).join(", ");
  if (typeof v === "boolean") return v ? "예" : "아니요";
  if (typeof v !== "number") return String(v);
  if (/_ms$/.test(key)) return (v / 1000).toFixed(2) + "초";
  if (/_sec$/.test(key)) return v.toFixed(2) + "초";
  if (PERCENT_KEYS.has(key) || PERCENT_KEYS.has(parent)) return (100 * v).toFixed(1) + "%";
  if (/^bleu/.test(key) || /^bleu/.test(parent || "")) return v.toFixed(1);
  if (Number.isInteger(v)) return v.toLocaleString("ko-KR");
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
// `labels`: true names a metric in words where it has a name, else shows its key as code.
function tilesHtml(pairs, {parent = null, labels = true} = {}) {
  return `<div class="stat-tiles">` + pairs.map(([key, v]) => {
    const last = key.split(".").at(-1);
    const shown = formatValue(last, v, parent);
    const text = typeof v !== "number";
    const name = labels ? labelOf(last) : null;
    const label = name ? `<span class="label">${esc(name)}</span>` : `<span class="label code">${esc(labels ? last : key)}</span>`;
    return `<div class="stat-tile">${label}` +
      `<span class="stat-value${text ? " text" : ""}" title="${esc(key)}">${esc(shown)}</span></div>`;
  }).join("") + `</div>`;
}
function groupHtml(title, sub, body) {
  return `<div class="kv-group"><div class="kv-group-title">${esc(title)}` +
    (sub ? `<span class="dim">${esc(sub)}</span>` : "") + `</div>${body}</div>`;
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
         note: "표는 corpus BLEU, 그림은 item 별 sentence BLEU예요"},
  comet: {label: "COMET", better: "higher", item: "comet", note: "그림의 점 하나가 sentence 하나예요"},
  laal_ms: {label: "LAAL", better: "lower", item: "laal_ms"},
  laal_ca_ms: {label: "LAAL CA", better: "lower", item: "laal_ca_ms"},
  yaal_ms: {label: "YAAL", better: "lower", item: "yaal_ms"},
  yaal_ca_ms: {label: "YAAL CA", better: "lower", item: "yaal_ca_ms"},
  longyaal_ms: {label: "LongYAAL", better: "lower", item: "longyaal_ms"},
  longyaal_ca_ms: {label: "LongYAAL CA", better: "lower", item: "longyaal_ca_ms"},
  lang_detect_accuracy: {label: "Lang detect", better: "higher"},
  segments_per_item: {label: "Segments / item", better: null, item: "n_segments"},
};
const BETTER_TEXT = {lower: "낮을수록 좋아요", higher: "높을수록 좋아요"};
const unitOf = key => /_(ms|sec)$/.test(key) ? "초" : key === "lang_detect_accuracy" ? "%" : "";
// A per-item value in the unit the table shows (milliseconds read as seconds).
const displayScale = key => /_ms$/.test(key) ? 0.001 : 1;
function formatScore(key, v, {unit = true} = {}) {
  if (v == null || !Number.isFinite(v)) return "—";
  const u = unit ? unitOf(key) : "";
  if (/_ms$/.test(key)) return (v / 1000).toFixed(2) + u;
  if (/_sec$/.test(key)) return v.toFixed(2) + u;
  if (key === "lang_detect_accuracy") return (100 * v).toFixed(1) + u;
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
