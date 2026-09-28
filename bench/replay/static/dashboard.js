// The bench dashboard. Runs come from /api/overview; a dataset's per-item
// distributions from /api/distributions/<run>, fetched when the dataset is shown.
// State that is worth a link (dataset, metric, sort) lives in the URL hash.

// DESIGN.md data-lane-*: every run on the dashboard matters equally.
const PALETTE = ["#4C9A5A", "#D08A2A", "#C0554F", "#7B60C0", "#3F8A9C", "#A0662A", "#6D7FA8", "#8A8F84"];
const MAX_COLS = 7;
const SVG_NS = 'xmlns="http://www.w3.org/2000/svg"';

let RUNS = [];                        // every run, as /api/overview returns it
const dists = new Map();              // run name -> {item key -> stats}
const state = {dataset: null, metric: null, sort: null, desc: false, base: null, more: false,
               x: null, y: null, points: true, hidden: new Set()};

// ── URL hash: dataset, metric and sort survive a reload and can be shared ──
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  for (const k of ["dataset", "metric", "sort", "x", "y", "base"]) if (h.get(k)) state[k] = h.get(k);
  if (h.get("desc")) state.desc = h.get("desc") === "1";
  if (h.get("more")) state.more = h.get("more") === "1";
  if (h.get("hide")) state.hidden = new Set(h.get("hide").split(","));
}
function writeHash() {
  const h = new URLSearchParams();
  for (const k of ["dataset", "metric", "sort", "x", "y", "base"]) if (state[k]) h.set(k, state[k]);
  if (state.desc) h.set("desc", "1");
  if (state.more) h.set("more", "1");
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
const datasetsOf = corpus =>
  [...new Set(RUNS.filter(r => r.corpus === corpus).map(r => r.dataset_key))].sort();
function describeDataset(key) {
  const runs = runsOf(key);
  const r = runs[0];
  if (!r) return "";
  const shas = new Set(runs.map(x => x.dataset_sha).filter(Boolean));
  const hashes = new Set(runs.map(x => x.meta.dataset.hash).filter(Boolean));
  const items = new Set(runs.map(x => x.counts.items).filter(n => n != null));
  const latest = runs.map(x => x.stamp).filter(Boolean).sort().at(-1);
  const warn = t => `<span class="warn">${esc(t)}</span>`;
  return [
    `<span><code>${esc(r.dataset_name || "?")}</code> → ${esc(r.target || "?")}</span>`,
    latest ? `<span>last run <span class="num">${esc(stampText(latest))}</span></span>` : "",
    `<span>run ${runs.length}개</span>`,
    items.size ? `<span>item <span class="num">${[...items].join(" / ")}</span>개</span>` : "",
    r.limit ? `<span>${r.pick === "longest" ? "가장 긴" : "앞의"} item ${r.limit}개만</span>` : "",
    shas.size === 1 ? `<span>manifest <code>${esc([...shas][0].slice(0, 12))}</code></span>` : "",
    hashes.size > 1 ? warn("run 마다 dataset config 가 달라요") : "",
    shas.size > 1 ? warn("run 마다 manifest 가 달라요") : "",
  ].filter(Boolean).join("");
}

function renderHeader() {
  // One dropdown: every dataset config, grouped by corpus, with its run count.
  const corpora = [...new Set(RUNS.map(r => r.corpus))].sort();
  $("dataset").innerHTML = corpora.map(c => `<optgroup label="${esc(c)}">` +
    datasetsOf(c).map(k => {
      const n = runsOf(k).length;
      return `<option value="${esc(k)}"${k === state.dataset ? " selected" : ""}>` +
        `${esc(k)} (run ${n}개)</option>`;
    }).join("") + `</optgroup>`).join("");
  const first = runsOf(state.dataset)[0];
  $("dataset-about").innerHTML = first ? metaLine("dataset", first.meta.dataset) : "";
  $("dataset-meta").innerHTML = describeDataset(state.dataset);
  renderBasePicker();
  renderModelMenu();
  $("replay-link").href = first ? `/replay?run=${encodeURIComponent(first.name)}` : "/replay";
}
function renderModelMenu() {
  const runs = runsOf(state.dataset), shown = shownRuns();
  $("models-label").textContent = shown.length === runs.length
    ? `Models: ${runs.length}개 모두` : `Models: ${runs.length}개 중 ${shown.length}개`;
  $("models-menu").innerHTML =
    `<div class="all"><button class="btn" data-all="1">모두 선택</button><button class="btn" data-all="0">모두 해제</button></div>` +
    runs.map(r => `<label><input type="checkbox" data-run="${esc(r.name)}"${state.hidden.has(r.name) ? "" : " checked"}>` +
      `<span class="dot" style="background:${colorOf.get(r.name)}"></span>` +
      `<span><span class="name">${esc(r.pipeline)}</span>${r.status && r.status !== "ok" ? ` <span class="sub">${esc(r.status)}</span>` : ""}` +
      `<div class="sub">${esc(modelLine(r))}</div></span></label>`).join("");
}
// The run every other row's change is measured from; defaults to the first run.
const baseRun = () => {
  const runs = shownRuns();
  return runs.find(r => r.name === state.base) || runs[0] || null;
};
function renderBasePicker() {
  const base = baseRun();
  $("base").innerHTML = shownRuns().map(r =>
    `<option value="${esc(r.name)}"${base && r.name === base.name ? " selected" : ""}>${esc(r.pipeline)}</option>`).join("");
}
$("base").addEventListener("change", ev => { state.base = ev.target.value; writeHash(); renderBoard(); });
$("more").addEventListener("click", () => { state.more = !state.more; writeHash(); renderBoard(); });
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
  writeHash(); renderModelMenu(); renderBasePicker(); renderBoard(); renderDistributions(); renderScatter();
}
// The menu closes on a click anywhere outside it.
addEventListener("click", ev => {
  const menu = $("models");
  if (menu.open && !menu.contains(ev.target)) menu.open = false;
});

$("dataset").addEventListener("change", ev => selectDataset(ev.target.value));

async function selectDataset(key) {
  state.dataset = key;
  const runs = runsOf(key);
  colorOf.clear();
  runs.forEach((r, i) => colorOf.set(r.name, PALETTE[i % PALETTE.length]));
  writeHash();
  renderHeader();
  renderBoard();
  renderScatter();
  $("dist-chart").innerHTML = `<div class="empty">distribution 을 불러오는 중이에요</div>`;
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
  if (!runs.length) {
    board.innerHTML = `<tr><td class="empty">고른 model 이 없어요. Models 메뉴에서 하나 이상 골라 주세요.</td></tr>`;
    $("more").hidden = true;
    return;
  }
  const all = byGroup(scoreKeys(runs)).flatMap(([, ks]) => ks);
  if (!state.sort || !all.includes(state.sort)) {
    state.sort = all.find(k => SCORES[k].better) || null;
    state.desc = false;
  }
  // People compare a handful of numbers at once; the rest wait behind a toggle.
  const visible = state.more ? all : all.slice(0, MAX_COLS);
  if (!visible.includes(state.sort) && state.sort) visible.push(state.sort);
  const groups = byGroup(visible);
  const ordered = groups.flatMap(([, ks]) => ks);
  $("more").hidden = all.length <= MAX_COLS;
  $("more").setAttribute("aria-pressed", String(state.more));
  $("more").textContent = state.more ? "Metric 줄여 보기" : `Metric 더 보기 (${all.length - MAX_COLS}개)`;

  const value = (r, k) => Number.isFinite(r.summary_metrics[k]) ? r.summary_metrics[k] : null;
  const sorted = [...runs].sort((a, b) => {
    if (!state.sort) return 0;
    const va = value(a, state.sort), vb = value(b, state.sort);
    if (va == null) return 1; if (vb == null) return -1;
    const c = compareScores(state.sort, va, vb) || (va - vb);
    return state.desc ? -c : c;
  });
  const best = {};
  for (const k of ordered) {
    const vs = runs.map(r => value(r, k)).filter(v => v != null);
    if (SCORES[k].better && vs.length > 1)
      best[k] = vs.reduce((a, b) => compareScores(k, a, b) <= 0 ? a : b);
  }
  const base = baseRun();
  const delta = (r, k, v) => {
    if (r === base) return `<span class="delta">baseline</span>`;
    const b = base && value(base, k);
    if (v == null || b == null) return `<span class="delta">—</span>`;
    const d = v - b, shown = formatScore(k, Math.abs(d), {unit: false});
    if (!/[1-9]/.test(shown)) return `<span class="delta">0</span>`;
    const c = compareScores(k, v, b);
    return `<span class="delta${c < 0 ? " up" : c > 0 ? " down" : ""}">${d > 0 ? "+" : "−"}${shown}</span>`;
  };
  const arrow = k => SCORES[k].better === "lower" ? " ↓" : SCORES[k].better === "higher" ? " ↑" : "";
  board.innerHTML =
    `<thead><tr class="groups"><th class="run"></th><th></th>` +
    groups.map(([title, ks]) => `<th class="g" colspan="${ks.length}">${esc(title)}</th>`).join("") +
    `</tr><tr><th class="run">Run</th><th>Items</th>` +
    groups.map(([, ks]) => ks.map((k, i) => {
      const unit = unitOf(k);
      return `<th class="sortable${i === 0 ? " g" : ""}${k === state.sort ? " sorted" : ""}" data-k="${k}"` +
        ` title="${esc([SCORES[k].note, BETTER_TEXT[SCORES[k].better]].filter(Boolean).join(" · "))}">` +
        `${esc(SCORES[k].label)}${unit ? ` <span class="unit">(${unit})</span>` : ""}${arrow(k)}` +
        `${k === state.sort ? (state.desc ? " ▴" : " ▾") : ""}</th>`;
    }).join("")).join("") +
    `</tr></thead><tbody>` +
    sorted.map(r => {
      const status = r.status && r.status !== "ok"
        ? ` <span class="chip${r.status === "running" ? "" : " bad"}">${esc(r.status)}</span>` : "";
      return `<tr${r === base ? ` class="base"` : ""}><td class="run"><div class="line">` +
        `<span class="dot" style="background:${colorOf.get(r.name)}"></span>` +
        `<a href="/replay?run=${encodeURIComponent(r.name)}" title="config hash ${esc(r.meta.pipeline.hash || "기록 없음")}">` +
        `${esc(r.pipeline)}</a>${status}${tagChips(r.meta.pipeline)}</div>` +
        (r.meta.pipeline.description ? `<div class="sub">${esc(r.meta.pipeline.description)}</div>` : "") +
        `<div class="sub code">${esc(modelLine(r))}</div></td>` +
        `<td class="n">${r.counts.items ?? "—"}${r.counts.of ? ` / ${r.counts.of}` : ""}</td>` +
        groups.map(([, ks]) => ks.map((k, i) => {
          const v = value(r, k);
          const isBest = v != null && best[k] != null && v === best[k];
          return `<td class="n${i === 0 ? " g" : ""}${isBest ? " best" : ""}"${isBest ? ` title="이 열에서 가장 좋아요"` : ""}>` +
            `${formatScore(k, v, {unit: false})}${delta(r, k, v)}</td>`;
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
const tickText = (key, t) => key === "comet" ? t.toFixed(2) : String(+t.toFixed(3));
const axisText = key => `${SCORES[key].label}${unitOf(key) ? ` (${unitOf(key)})` : ""}`;

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
    chart.innerHTML = `<div class="empty">${runs.length ? "이 run 들에는 per-item score 가 없어요." : "고른 model 이 없어요. Models 메뉴에서 하나 이상 골라 주세요."}</div>`;
    return;
  }
  if (!keys.includes(state.metric)) state.metric = keys.find(k => SCORES[k].better) || keys[0];
  picker.innerHTML = byGroup(keys).map(([title, ks]) =>
    `<div class="grp"><span class="label">${esc(title)}</span><div class="seg-group">` + ks.map(k =>
      `<button data-k="${k}" aria-pressed="${k === state.metric}">${esc(SCORES[k].label)}</button>`).join("") +
    `</div></div>`).join("");
  picker.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    state.metric = b.dataset.k; writeHash(); renderDistributions();
  }));

  const key = state.metric, spec = SCORES[key], scale = displayScale(key);
  $("dist-note").textContent = "Mean 하나가 아니라 item 마다의 score 예요. Median 이 좋은 run 이 왼쪽에 와요." +
    (spec.note ? ` ${spec.note}.` : "");
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
  const dir = BETTER_TEXT[spec.better] ? ` · ${BETTER_TEXT[spec.better]}` : "";
  out.push(`<text class="c-axis" x="14" y="${top + plotH / 2}" text-anchor="middle" transform="rotate(-90 14 ${top + plotH / 2})">${esc(axisText(key))}${dir}</text>`);
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
        `<text class="c-sub" x="${cx}" y="${ly + 16}" text-anchor="middle">n=${s.n}</text>`);
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
  const dir = BETTER_TEXT[spec.better] || "좋은 방향이 없어요";
  return `<span><svg ${SVG_NS} width="20" height="40"><line class="c-whisker" x1="10" x2="10" y1="2" y2="38"/>` +
    `<line class="c-whisker" x1="5" x2="15" y1="2" y2="2"/><line class="c-whisker" x1="5" x2="15" y1="38" y2="38"/>` +
    `<rect x="4" y="11" width="12" height="16" fill="var(--text-3)" fill-opacity="0.25" stroke="var(--text-3)"/>` +
    `<line class="c-median" x1="2" x2="18" y1="17" y2="17"/></svg>위에서부터 max · q3 · median · q1 · min</span>` +
    `<span><svg ${SVG_NS} width="12" height="12"><path class="c-mean" d="M 6 1 l 5 5 l -5 5 l -5 -5 z"/></svg>mean</span>` +
    `<span>기둥 위 숫자는 median · ${dir}</span>` +
    `<label class="check"><input type="checkbox" id="dist-points"${state.points ? " checked" : ""}>item 점 보기</label>`;
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
  if (!state.x || !state.y) { $("scatter-note").textContent = ""; chart.innerHTML = `<div class="empty">이 run 들에는 찍을 accuracy 와 latency score 가 없어요.</div>`; return; }

  const all = runs.map(r => ({run: r, x: r.summary_metrics[state.x], y: r.summary_metrics[state.y]}));
  const pts = all.filter(p => Number.isFinite(p.x) && Number.isFinite(p.y));
  // Say who is missing and why, rather than drawing fewer dots without a word.
  const left = all.filter(p => !pts.includes(p)).map(p =>
    `${p.run.pipeline} (${p.run.status === "running" ? "아직 도는 중이라 run summary 가 없어요"
      : `${[!Number.isFinite(p.y) && SCORES[state.y].label, !Number.isFinite(p.x) && SCORES[state.x].label].filter(Boolean).join(", ")} 값이 없어요`})`);
  $("scatter-note").textContent = left.length ? `빠진 run: ${left.join(" · ")}` : "";
  if (!pts.length) { chart.innerHTML = `<div class="empty">두 score 를 모두 가진 run 이 없어요.</div>`; return; }
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
  out.push(`<text class="c-axis" x="${(L + W - R) / 2}" y="${H - 8}" text-anchor="middle">${esc(axisText(state.x))} · ${BETTER_TEXT[SCORES[state.x].better]} → 좋은 쪽</text>`,
    `<text class="c-axis" x="14" y="${(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 14 ${(T + H - B) / 2})">${esc(axisText(state.y))} · ${BETTER_TEXT[SCORES[state.y].better]} → 좋은 쪽</text>`);

  const beats = (a, b) => {
    const cx = compareScores(state.x, a.x, b.x), cy = compareScores(state.y, a.y, b.y);
    return cx <= 0 && cy <= 0 && (cx < 0 || cy < 0);
  };
  const front = pts.filter(p => !pts.some(q => q !== p && beats(q, p)))
    .sort((a, b) => fx(a.x * sx) - fx(b.x * sx));
  if (front.length > 1) {
    const [a, b] = front.slice(-2).map(p => [fx(p.x * sx), fy(p.y * sy)]);
    out.push(`<polyline class="c-frontier" points="${front.map(p => `${fx(p.x * sx)},${fy(p.y * sy)}`).join(" ")}"/>`,
      `<text class="c-axis" x="${(a[0] + b[0]) / 2}" y="${(a[1] + b[1]) / 2 - 8}" text-anchor="middle">Pareto frontier</text>`);
  }
  for (const p of pts) {
    const cx = fx(p.x * sx), cy = fy(p.y * sy), color = colorOf.get(p.run.name);
    out.push(`<circle cx="${cx}" cy="${cy}" r="6" fill="${color}" stroke="var(--panel)" stroke-width="1.5"` +
      ` data-tip="${esc(`${p.run.pipeline}\n${SCORES[state.y].label}  ${formatScore(state.y, p.y)}\n${SCORES[state.x].label}  ${formatScore(state.x, p.x)}${front.includes(p) ? "\nPareto frontier 위에 있어요" : ""}`)}"/>`,
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
    document.querySelector(".wrap").innerHTML = `<div class="panel empty">bench/runs 에 run 이 아직 없어요. <code>make bench CONFIG=… DATASET=…</code> 로 하나 돌려 주세요.</div>`;
    return;
  }
  const start = RUNS.find(r => r.dataset_key === state.dataset) || RUNS[0];
  await selectDataset(start.dataset_key);
})();
