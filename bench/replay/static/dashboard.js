// The bench dashboard. Runs come from /api/overview; a dataset's per-item
// distributions from /api/distributions/<run>, fetched when the dataset is shown.
// State that is worth a link (dataset, metric, sort) lives in the URL hash.

const PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
                 "#b279a2", "#eeca3b", "#9d755d", "#ff9da6", "#bab0ac"];
const SVG_NS = 'xmlns="http://www.w3.org/2000/svg"';

let RUNS = [];                        // every run, as /api/overview returns it
const dists = new Map();              // run name -> {item key -> stats}
const state = {corpus: null, dataset: null, metric: null, sort: null, desc: false,
               x: null, y: null, points: true, hidden: new Set()};

// ── URL hash: dataset, metric and sort survive a reload and can be shared ──
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  for (const k of ["dataset", "metric", "sort", "x", "y"]) if (h.get(k)) state[k] = h.get(k);
  if (h.get("desc")) state.desc = h.get("desc") === "1";
  if (h.get("hide")) state.hidden = new Set(h.get("hide").split(","));
}
function writeHash() {
  const h = new URLSearchParams();
  for (const k of ["dataset", "metric", "sort", "x", "y"]) if (state[k]) h.set(k, state[k]);
  if (state.desc) h.set("desc", "1");
  const hidden = runsOf(state.dataset).filter(r => state.hidden.has(r.name)).map(r => r.name);
  if (hidden.length) h.set("hide", hidden.join(","));
  history.replaceState(null, "", "#" + h.toString());
}

// ── tooltip ────────────────────────────────────────────────────────────────
const tip = $("tip");
function showTip(ev, text) {
  tip.textContent = text; tip.style.display = "block";
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(innerWidth - w - 8, ev.clientX + pad) + "px";
  tip.style.top = Math.min(innerHeight - h - 8, ev.clientY + pad) + "px";
}
const hideTip = () => { tip.style.display = "none"; };
function wireTips(root) {
  root.querySelectorAll("[data-tip]").forEach(el => {
    el.addEventListener("mousemove", ev => showTip(ev, el.dataset.tip));
    el.addEventListener("mouseleave", hideTip);
  });
}

// ── datasets ───────────────────────────────────────────────────────────────
const runsOf = key => RUNS.filter(r => r.dataset_key === key)
  .sort((a, b) => a.pipeline.localeCompare(b.pipeline));
// The runs the charts and table draw: the dataset's runs minus the ones unticked
// in the model menu. Colours are given over all of them, so unticking one does
// not repaint the rest.
const shownRuns = () => runsOf(state.dataset).filter(r => !state.hidden.has(r.name));
const colorOf = new Map();
function datasetsOf(corpus) {
  const keys = [...new Set(RUNS.filter(r => r.corpus === corpus).map(r => r.dataset_key))];
  return keys.sort();
}
function describeDataset(key) {
  const runs = runsOf(key);
  const r = runs[0];
  if (!r) return "";
  const shas = new Set(runs.map(x => x.dataset_sha).filter(Boolean));
  const items = new Set(runs.map(x => x.counts.items).filter(n => n != null));
  return [
    `${r.dataset_name || "?"} → ${r.target || "?"}`,
    r.limit ? `first ${r.limit} items` : "",
    items.size ? `${[...items].join(" / ")} items` : "",
    shas.size === 1 ? `manifest ${[...shas][0].slice(0, 12)}` : shas.size > 1 ? "⚠ runs used different manifests" : "",
    `${runs.length} run${runs.length === 1 ? "" : "s"}`,
  ].filter(Boolean).join("  ·  ");
}

function renderHeader() {
  const corpora = [...new Set(RUNS.map(r => r.corpus))].sort();
  const corpus = $("corpus");
  corpus.innerHTML = corpora.map(c =>
    `<option value="${esc(c)}"${c === state.corpus ? " selected" : ""}>${esc(c)}</option>`).join("");
  $("datasets").innerHTML = datasetsOf(state.corpus).map(k =>
    `<button data-k="${esc(k)}" aria-pressed="${k === state.dataset}">${esc(k)}` +
    `<span class="n">${runsOf(k).length}</span></button>`).join("");
  $("dataset-meta").textContent = describeDataset(state.dataset);
  renderModelMenu();
  const first = runsOf(state.dataset)[0];
  $("replay-link").href = first ? `/replay?run=${encodeURIComponent(first.name)}` : "/replay";
}
function renderModelMenu() {
  const runs = runsOf(state.dataset), shown = shownRuns();
  $("models-label").textContent = shown.length === runs.length
    ? `Models: all ${runs.length}` : `Models: ${shown.length} of ${runs.length}`;
  $("models-menu").innerHTML =
    `<div class="all"><button data-all="1">Select all</button><button data-all="0">Clear</button></div>` +
    runs.map(r => `<label><input type="checkbox" data-run="${esc(r.name)}"${state.hidden.has(r.name) ? "" : " checked"}>` +
      `<span class="swatch" style="background:${colorOf.get(r.name)}"></span>` +
      `<span>${esc(r.pipeline)}${r.status && r.status !== "ok" ? ` <span class="sub">${esc(r.status)}</span>` : ""}` +
      `<div class="sub">${esc(modelLine(r))}</div></span></label>`).join("");
}
$("models-menu").addEventListener("change", ev => {
  const box = ev.target.closest("input[data-run]"); if (!box) return;
  box.checked ? state.hidden.delete(box.dataset.run) : state.hidden.add(box.dataset.run);
  modelsChanged();
});
$("models-menu").addEventListener("click", ev => {
  const b = ev.target.closest("button[data-all]"); if (!b) return;
  for (const r of runsOf(state.dataset)) b.dataset.all === "1" ? state.hidden.delete(r.name) : state.hidden.add(r.name);
  modelsChanged();
});
function modelsChanged() {
  writeHash(); renderModelMenu(); renderBoard(); renderDistributions(); renderScatter();
}
// The menu closes on a click anywhere outside it.
addEventListener("click", ev => {
  const menu = $("models");
  if (menu.open && !menu.contains(ev.target)) menu.open = false;
});

$("corpus").addEventListener("change", ev => {
  state.corpus = ev.target.value;
  selectDataset(datasetsOf(state.corpus)[0]);
});
$("datasets").addEventListener("click", ev => {
  const b = ev.target.closest("button"); if (b) selectDataset(b.dataset.k);
});

async function selectDataset(key) {
  state.dataset = key;
  const runs = runsOf(key);
  colorOf.clear();
  runs.forEach((r, i) => colorOf.set(r.name, PALETTE[i % PALETTE.length]));
  writeHash();
  renderHeader();
  renderBoard();
  renderScatter();
  $("dist-chart").innerHTML = `<div class="empty">loading distributions…</div>`;
  await Promise.all(runs.filter(r => !dists.has(r.name)).map(async r => {
    try {
      const res = await fetch(`/api/distributions/${encodeURIComponent(r.name)}`);
      dists.set(r.name, res.ok ? await res.json() : {});
    } catch { dists.set(r.name, {}); }
  }));
  if (state.dataset === key) renderDistributions();
}

// ── leaderboard ──────────────────────────────────────────────────────────
function scoreKeys(runs) {
  return Object.keys(SCORES).filter(k => runs.some(r => Number.isFinite(r.summary_metrics[k])));
}
function modelLine(r) {
  const m = r.models || {};
  return [m.transcription, m.correction, m.translation].filter(Boolean).join(" → ");
}
function renderBoard() {
  const runs = shownRuns();
  const board = $("board");
  if (!runs.length) { board.innerHTML = `<tr><td class="empty">no model selected</td></tr>`; return; }
  const keys = scoreKeys(runs);
  const groups = byGroup(keys);
  const ordered = groups.flatMap(([, ks]) => ks);
  if (!state.sort || !ordered.includes(state.sort)) {
    state.sort = ordered.find(k => SCORES[k].better) || null;
    state.desc = false;
  }
  const value = (r, k) => Number.isFinite(r.summary_metrics[k]) ? r.summary_metrics[k] : null;
  const sorted = [...runs].sort((a, b) => {
    if (!state.sort) return 0;
    const va = value(a, state.sort), vb = value(b, state.sort);
    if (va == null) return 1; if (vb == null) return -1;
    const c = compareScores(state.sort, va, vb) || (va - vb);
    return state.desc ? -c : c;
  });
  const best = {}, span = {};
  for (const k of ordered) {
    const vs = runs.map(r => value(r, k)).filter(v => v != null);
    span[k] = [Math.min(...vs), Math.max(...vs)];
    if (SCORES[k].better && vs.length > 1)
      best[k] = vs.reduce((a, b) => compareScores(k, a, b) <= 0 ? a : b);
  }
  const arrow = k => SCORES[k].better === "lower" ? "↓" : SCORES[k].better === "higher" ? "↑" : "";
  board.innerHTML =
    `<thead><tr class="groups"><th class="run"></th><th></th>` +
    groups.map(([title, ks]) => `<th class="g" colspan="${ks.length}">${esc(title)}</th>`).join("") +
    `</tr><tr><th class="run">run</th><th>items</th>` +
    groups.map(([, ks]) => ks.map((k, i) =>
      `<th class="sortable${i === 0 ? " g" : ""}${k === state.sort ? " sorted" : ""}" data-k="${k}"` +
      ` title="${esc(SCORES[k].note || "")}${SCORES[k].better ? ` (${SCORES[k].better} is better)` : ""}">` +
      `${esc(SCORES[k].label)} <span class="dir">${arrow(k)}</span>` +
      `${k === state.sort ? (state.desc ? " ▴" : " ▾") : ""}</th>`).join("")).join("") +
    `</tr></thead><tbody>` +
    sorted.map(r => {
      const status = r.status && r.status !== "ok"
        ? ` <span class="chip ${r.status === "running" ? "" : "bad"}">${esc(r.status)}</span>` : "";
      return `<tr><td class="run"><span class="swatch" style="background:${colorOf.get(r.name)}"></span>` +
        `<a href="/replay?run=${encodeURIComponent(r.name)}">${esc(r.pipeline)}</a>${status}` +
        `<div class="sub">${esc(modelLine(r))}</div></td>` +
        `<td class="num-cell">${r.counts.items ?? "—"}${r.counts.of ? ` / ${r.counts.of}` : ""}</td>` +
        groups.map(([, ks]) => ks.map((k, i) => {
          const v = value(r, k);
          const [lo, hi] = span[k];
          const frac = v == null || hi <= 0 ? 0 : Math.max(0.02, v / Math.max(hi, Math.abs(lo)));
          const isBest = v != null && best[k] != null && v === best[k];
          return `<td class="num-cell${i === 0 ? " g" : ""}${isBest ? " best" : ""}">${formatScore(k, v)}` +
            (v == null ? "" : `<span class="bar" style="width:calc((100% - 12px) * ${frac.toFixed(3)})"></span>`) +
            `</td>`;
        }).join("")).join("") + `</tr>`;
    }).join("") + `</tbody>`;
  board.querySelectorAll("th.sortable").forEach(th => th.addEventListener("click", () => {
    if (state.sort === th.dataset.k) state.desc = !state.desc;
    else { state.sort = th.dataset.k; state.desc = false; }
    writeHash(); renderBoard();
  }));
}

// ── axis helpers ─────────────────────────────────────────────────────────
function niceTicks(lo, hi, count = 6) {
  if (!(hi > lo)) { hi = lo + 1; }
  const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => raw <= s) || 10 * mag;
  const start = Math.floor(lo / step) * step, out = [];
  for (let t = start; t <= hi + step * 1e-6; t += step) out.push(+t.toFixed(10));
  if (out[out.length - 1] < hi) out.push(+(out[out.length - 1] + step).toFixed(10));
  return out;
}
const tickText = (key, t) => /_(ms|sec)$/.test(key) ? `${+t.toFixed(3)}s`
  : key === "comet" ? t.toFixed(2) : String(+t.toFixed(3));

// ── per-item distribution: one column per run, best on the left ────────────
function distKeys(runs) {
  return Object.keys(SCORES).filter(k => SCORES[k].item &&
    runs.some(r => (dists.get(r.name) || {})[SCORES[k].item]));
}
function renderDistributions() {
  const runs = shownRuns();
  const keys = distKeys(runs);
  const picker = $("dist-picker"), chart = $("dist-chart");
  if (!keys.length) {
    picker.innerHTML = ""; $("dist-legend").innerHTML = "";
    chart.innerHTML = `<div class="empty">${runs.length ? "no per-item scores in these runs" : "no model selected"}</div>`;
    return;
  }
  if (!keys.includes(state.metric)) state.metric = keys.find(k => SCORES[k].better) || keys[0];
  picker.innerHTML = byGroup(keys).map(([title, ks]) =>
    `<div class="grp"><span class="grp-name">${esc(title)}</span>` + ks.map(k =>
      `<button data-k="${k}" aria-pressed="${k === state.metric}">${esc(SCORES[k].label)}</button>`).join("") +
    `</div>`).join("");
  picker.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    state.metric = b.dataset.k; writeHash(); renderDistributions();
  }));

  const key = state.metric, spec = SCORES[key], scale = displayScale(key);
  $("dist-note").textContent = "every item's score, not just the average — best median on the left" +
    (spec.note ? ` · ${spec.note}` : "");
  $("dist-legend").innerHTML = legendHtml(spec);
  $("dist-points").addEventListener("change", ev => { state.points = ev.target.checked; renderDistributions(); });

  const cols = runs.map(r => ({run: r, s: (dists.get(r.name) || {})[spec.item]})).filter(x => x.s);
  cols.sort((a, b) => compareScores(key, a.s.median, b.s.median) || a.s.median - b.s.median);
  const lo = Math.min(...cols.map(x => x.s.min)) * scale, hi = Math.max(...cols.map(x => x.s.max)) * scale;
  const ticks = niceTicks(lo, hi, 6);
  // Columns keep a readable width; many runs make the chart wider and it scrolls.
  const padL = 64, padR = 16, top = 26, avail = Math.max(560, chart.clientWidth || 900) - padL - padR;
  const colW = Math.max(64, Math.min(220, avail / Math.max(cols.length, 1)));
  const tilt = colW < 118;
  const labelH = tilt ? 96 : 48, plotH = 320;
  const W = padL + padR + colW * cols.length, H = top + plotH + labelH;
  const y = v => top + plotH * (1 - (v - ticks[0]) / (ticks[ticks.length - 1] - ticks[0]));
  const out = [`<svg ${SVG_NS} viewBox="0 0 ${Math.max(W, padL + padR + avail)} ${H}" width="${Math.max(W, padL + padR + avail)}" height="${H}">`];
  for (const t of ticks) out.push(
    `<line class="c-grid" x1="${padL}" x2="${W - padR}" y1="${y(t)}" y2="${y(t)}"/>`,
    `<text class="c-axis" x="${padL - 8}" y="${y(t) + 3}" text-anchor="end">${tickText(key, t)}</text>`);
  const dir = spec.better === "lower" ? " · lower is better" : spec.better === "higher" ? " · higher is better" : "";
  out.push(`<text class="c-axis" x="14" y="${top + plotH / 2}" text-anchor="middle" transform="rotate(-90 14 ${top + plotH / 2})">${esc(spec.label)}${dir}</text>`);
  cols.forEach(({run, s}, i) => {
    const cx = padL + colW * (i + 0.5), color = colorOf.get(run.name), half = Math.min(26, colW * 0.28);
    const v = k => y(s[k] * scale);
    if (state.points) s.points.forEach((p, j) => {
      const jitter = (((j * 0.6180339887) % 1) - 0.5) * colW * 0.62;
      out.push(`<circle cx="${(cx + jitter).toFixed(1)}" cy="${y(p * scale).toFixed(1)}" r="1.8" fill="${color}" opacity="0.28"/>`);
    });
    out.push(
      `<line class="c-whisker" x1="${cx}" x2="${cx}" y1="${v("max")}" y2="${v("q3")}"/>`,
      `<line class="c-whisker" x1="${cx}" x2="${cx}" y1="${v("q1")}" y2="${v("min")}"/>`,
      `<line class="c-whisker" x1="${cx - half / 2}" x2="${cx + half / 2}" y1="${v("max")}" y2="${v("max")}"/>`,
      `<line class="c-whisker" x1="${cx - half / 2}" x2="${cx + half / 2}" y1="${v("min")}" y2="${v("min")}"/>`,
      `<rect x="${cx - half}" y="${v("q3")}" width="${half * 2}" height="${Math.max(1, v("q1") - v("q3"))}" rx="2"` +
      ` fill="${color}" fill-opacity="0.28" stroke="${color}" stroke-width="1.4"/>`,
      `<line class="c-median" x1="${cx - half - 2}" x2="${cx + half + 2}" y1="${v("median")}" y2="${v("median")}"/>`,
      `<path class="c-mean" d="M ${cx} ${v("mean") - 5} l 5 5 l -5 5 l -5 -5 z"/>`,
      `<text class="c-value" x="${cx}" y="${v("max") - 7}" text-anchor="middle">${formatScore(key, s.median)}</text>`);
    const ly = top + plotH + 16;
    out.push(tilt
      ? `<text class="c-label" x="${cx}" y="${ly}" text-anchor="end" transform="rotate(-35 ${cx} ${ly})">${esc(clip(run.pipeline, 26))}</text>`
      : `<text class="c-label" x="${cx}" y="${ly}" text-anchor="middle">${esc(clip(run.pipeline, Math.floor(colW / 7)))}</text>` +
        `<text class="c-sub" x="${cx}" y="${ly + 14}" text-anchor="middle">n=${s.n}</text>`);
    out.push(`<rect x="${cx - colW / 2}" y="${top}" width="${colW}" height="${plotH + labelH}" fill="transparent" data-tip="${esc(
        `${run.pipeline}\n` +
        `n       ${s.n}\nmax     ${formatScore(key, s.max)}\nq3      ${formatScore(key, s.q3)}\n` +
        `median  ${formatScore(key, s.median)}\nmean    ${formatScore(key, s.mean)}\n` +
        `q1      ${formatScore(key, s.q1)}\nmin     ${formatScore(key, s.min)}`)}"/>`);
  });
  out.push("</svg>");
  chart.innerHTML = out.join("");
  wireTips(chart);
}
function legendHtml(spec) {
  const dir = spec.better === "lower" ? "lower is better" : spec.better === "higher" ? "higher is better" : "no better direction";
  return `<span><svg ${SVG_NS} width="20" height="40"><line class="c-whisker" x1="10" x2="10" y1="2" y2="38"/>` +
    `<line class="c-whisker" x1="5" x2="15" y1="2" y2="2"/><line class="c-whisker" x1="5" x2="15" y1="38" y2="38"/>` +
    `<rect x="4" y="11" width="12" height="16" fill="var(--ink-faint)" fill-opacity="0.25" stroke="var(--ink-soft)"/>` +
    `<line class="c-median" x1="2" x2="18" y1="17" y2="17"/></svg> max · 75% · median · 25% · min</span>` +
    `<span><svg ${SVG_NS} width="12" height="12"><path class="c-mean" d="M 6 1 l 5 5 l -5 5 l -5 -5 z"/></svg> mean</span>` +
    `<span>number above: median · ${dir}</span>` +
    `<label><input type="checkbox" id="dist-points"${state.points ? " checked" : ""}> show items</label>`;
}
const clip = (s, n) => s.length > n ? s.slice(0, n - 1) + "…" : s;

// ── accuracy vs latency ───────────────────────────────────────────────────
const ACCURACY_GROUPS = new Set(["Transcription accuracy", "Translation accuracy", "Language detection"]);
const LATENCY_GROUPS = new Set(["Transcription latency", "Translation latency"]);
function renderScatter() {
  const runs = shownRuns();
  const keys = scoreKeys(runs).filter(k => SCORES[k].better);
  const yKeys = keys.filter(k => ACCURACY_GROUPS.has(groupOf(k)));
  const xKeys = keys.filter(k => LATENCY_GROUPS.has(groupOf(k)));
  const chart = $("scatter");
  const fill = (sel, ks, cur) => {
    sel.innerHTML = ks.map(k => `<option value="${k}"${k === cur ? " selected" : ""}>${esc(SCORES[k].label)}</option>`).join("");
  };
  if (!yKeys.includes(state.y)) state.y = ["comet", "bleu", "wer"].find(k => yKeys.includes(k)) || yKeys[0];
  if (!xKeys.includes(state.x))
    state.x = ["laal_ms", "yaal_ms", "longyaal_ms", "avg_fsl_sec"].find(k => xKeys.includes(k)) || xKeys[0];
  fill($("scatter-y"), yKeys, state.y); fill($("scatter-x"), xKeys, state.x);
  if (!state.x || !state.y) { $("scatter-note").textContent = ""; chart.innerHTML = `<div class="empty">these runs have no accuracy and latency scores to plot</div>`; return; }

  const all = runs.map(r => ({run: r, x: r.summary_metrics[state.x], y: r.summary_metrics[state.y]}));
  const pts = all.filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
  // Say who is missing and why, rather than drawing fewer dots without a word.
  const left = all.filter(p => !pts.includes(p)).map(p =>
    `${p.run.pipeline} (${p.run.status === "running" ? "still running — no run summary yet"
      : `no ${[!Number.isFinite(p.y) && SCORES[state.y].label, !Number.isFinite(p.x) && SCORES[state.x].label].filter(Boolean).join(" or ")}`})`);
  $("scatter-note").textContent = left.length ? `not plotted: ${left.join(" · ")}` : "";
  if (!pts.length) { chart.innerHTML = `<div class="empty">no run has both scores</div>`; return; }
  const sx = displayScale(state.x), sy = displayScale(state.y);
  const xs = pts.map(p => p.x * sx), ys = pts.map(p => p.y * sy);
  const pad = (lo, hi) => hi > lo ? (hi - lo) * 0.12 : Math.abs(hi || 1) * 0.1;
  const xt = niceTicks(Math.min(...xs) - pad(Math.min(...xs), Math.max(...xs)), Math.max(...xs) + pad(Math.min(...xs), Math.max(...xs)), 6);
  const yt = niceTicks(Math.min(...ys) - pad(Math.min(...ys), Math.max(...ys)), Math.max(...ys) + pad(Math.min(...ys), Math.max(...ys)), 5);
  const W = Math.max(560, chart.clientWidth || 900), H = 380, L = 64, R = 24, T = 16, B = 46;
  // Better is always up and to the right: a lower-is-better axis runs backwards.
  const flipX = SCORES[state.x].better === "lower", flipY = SCORES[state.y].better === "lower";
  const fx = v => { const f = (v - xt[0]) / (xt[xt.length - 1] - xt[0]); return L + (W - L - R) * (flipX ? 1 - f : f); };
  const fy = v => { const f = (v - yt[0]) / (yt[yt.length - 1] - yt[0]); return T + (H - T - B) * (flipY ? f : 1 - f); };
  const out = [`<svg ${SVG_NS} viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`];
  for (const t of xt) out.push(`<line class="c-grid" x1="${fx(t)}" x2="${fx(t)}" y1="${T}" y2="${H - B}"/>`,
    `<text class="c-axis" x="${fx(t)}" y="${H - B + 16}" text-anchor="middle">${tickText(state.x, t)}</text>`);
  for (const t of yt) out.push(`<line class="c-grid" x1="${L}" x2="${W - R}" y1="${fy(t)}" y2="${fy(t)}"/>`,
    `<text class="c-axis" x="${L - 8}" y="${fy(t) + 3}" text-anchor="end">${tickText(state.y, t)}</text>`);
  out.push(`<text class="c-axis" x="${(L + W - R) / 2}" y="${H - 8}" text-anchor="middle">${esc(SCORES[state.x].label)} (${SCORES[state.x].better} is better) →  better</text>`,
    `<text class="c-axis" x="14" y="${(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 14 ${(T + H - B) / 2})">${esc(SCORES[state.y].label)} (${SCORES[state.y].better} is better) →  better</text>`);

  const beats = (a, b) => {
    const cx = compareScores(state.x, a.x, b.x), cy = compareScores(state.y, a.y, b.y);
    return cx <= 0 && cy <= 0 && (cx < 0 || cy < 0);
  };
  const front = pts.filter(p => !pts.some(q => q !== p && beats(q, p)))
    .sort((a, b) => fx(a.x * sx) - fx(b.x * sx));
  if (front.length > 1) out.push(`<polyline class="c-frontier" points="${front.map(p => `${fx(p.x * sx)},${fy(p.y * sy)}`).join(" ")}"/>`);
  for (const p of pts) {
    const cx = fx(p.x * sx), cy = fy(p.y * sy), color = colorOf.get(p.run.name);
    out.push(`<circle cx="${cx}" cy="${cy}" r="6" fill="${color}" stroke="var(--panel)" stroke-width="1.5"` +
      ` data-tip="${esc(`${p.run.pipeline}\n${SCORES[state.y].label}  ${formatScore(state.y, p.y)}\n${SCORES[state.x].label}  ${formatScore(state.x, p.x)}${front.includes(p) ? "\non the frontier" : ""}`)}"/>`,
      `<text class="c-dot-label" x="${cx + 9}" y="${cy + 4}">${esc(clip(p.run.pipeline, 34))}</text>`);
  }
  out.push("</svg>");
  chart.innerHTML = out.join("");
  wireTips(chart);
}
$("scatter-x").addEventListener("change", ev => { state.x = ev.target.value; writeHash(); renderScatter(); });
$("scatter-y").addEventListener("change", ev => { state.y = ev.target.value; writeHash(); renderScatter(); });

// ── start ─────────────────────────────────────────────────────────────────
let resizeTimer = null;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { renderDistributions(); renderScatter(); }, 150);
});

(async () => {
  readHash();
  const res = await fetch("/api/overview");
  RUNS = (await res.json()).runs;
  if (!RUNS.length) {
    document.querySelector(".wrap").innerHTML = `<div class="card empty">no runs in bench/runs yet — make bench CONFIG=… DATASET=…</div>`;
    return;
  }
  const start = RUNS.find(r => r.dataset_key === state.dataset) || RUNS[0];
  state.corpus = start.corpus;
  await selectDataset(start.dataset_key);
})();
