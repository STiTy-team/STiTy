import type { CatalogRun, ReplayEvent, ReplayItem, RunInfo } from "@/lib/replay/types"

export type PlayedEvent = ReplayEvent & { audio: number; at: number }

export type PlayedItem = ReplayItem & {
  span: number
  played: PlayedEvent[]
  pending: [number, number][]
}

export const laneOf = (e: ReplayEvent) => e.tag || e.type
export const isTimed = (e: ReplayEvent): e is ReplayEvent & { t: number; dur: number } =>
  typeof e.dur === "number" && e.t != null

export function prepareItem(item: ReplayItem): PlayedItem {
  const span = Math.max(item.duration || 0, ...item.events.map((e) => e.audio || 0), 0.001)
  const played = item.events
    .map((e) => {
      const audio = e.audio ?? (e.type === "item_close" ? span : 0)
      return { ...e, audio, at: audio }
    })
    .sort((a, b) => a.at - b.at || (a.t == null || b.t == null ? 0 : a.t - b.t))
  const pending: [number, number][] = []
  played.forEach((e, i) => {
    if (e.type !== "transcribed") return
    const final = played.slice(i + 1).find((x) => x.type === "translated")
    if (final) pending.push([e.at, final.at])
  })
  return { ...item, span, played, pending }
}

export function pastEdge(events: PlayedEvent[], now: number) {
  const upto = events.findIndex((e) => e.at > now)
  return upto === -1 ? events.length : upto
}

export function laneOrder(item: ReplayItem) {
  const order: string[] = []
  for (const e of item.events) if (isTimed(e) && !order.includes(laneOf(e))) order.push(laneOf(e))
  return order
}

type Chunk = ReplayEvent & { t: number; from: number; read: number }

export type TimingRow = {
  commit: (ReplayEvent & { t: number; audio: number }) | null
  final: ReplayEvent | null
  start: number
  spans: (ReplayEvent & { t: number; dur: number })[]
  chunks: Chunk[]
  vad: ReplayEvent[]
}

export type TimingModel = {
  rows: TimingRow[]
  end: number
  wallAt: (audio: number) => number
  feedLag: number
}

const newRow = (commit: TimingRow["commit"], final: ReplayEvent | null): TimingRow => ({
  commit,
  final,
  start: Infinity,
  spans: [],
  chunks: [],
  vad: [],
})

/**
 * One row per commit. Audio goes to the first commit that had heard it; timed work goes
 * to the first commit delivered after the work ended, so a translation lands in its own
 * segment's row even when two commits came out of one decode.
 */
export function timingModel(item: ReplayItem): TimingModel {
  const clocked = item.events.filter((e): e is ReplayEvent & { t: number } => e.t != null)
  const chunks = clocked.filter((e) => e.type === "chunk")
  const finals = clocked.filter((e) => e.type === "translated")
  const end = Math.max(0.001, ...clocked.map((e) => (isTimed(e) ? e.t + e.dur : e.t)))
  const rows = clocked
    .filter((e) => e.type === "transcribed")
    .sort((a, b) => a.t - b.t)
    .map((commit, k) => newRow({ ...commit, audio: commit.audio ?? 0 }, finals[k] ?? null))
  const tail = newRow(null, null)
  const translatedAt = (row: TimingRow) =>
    row.final ? (row.final.translated_elapsed_sec ?? row.final.t ?? 0) : row.commit!.t
  const byAudio = (heard: number) => rows.find((row) => heard <= row.commit!.audio + 1e-6) ?? tail
  const byDelivery = (t: number) => rows.find((row) => t <= translatedAt(row) + 1e-6) ?? tail

  const reads = clocked.filter((e) => isTimed(e) && e.tag === "decode").map((e) => e.t)
  let from = 0
  for (const chunk of chunks) {
    const row = byAudio(chunk.audio ?? 0)
    row.chunks.push({ ...chunk, from, read: reads.filter((t) => t < from).length })
    row.start = Math.min(row.start, from)
    from = chunk.t
  }
  for (const e of clocked) {
    if (isTimed(e)) {
      const row = byDelivery(e.t + e.dur)
      row.spans.push(e)
      row.start = Math.min(row.start, e.t)
    } else if (e.type === "speech") byAudio(e.audio ?? 0).vad.push(e)
  }
  if (tail.spans.length || tail.chunks.length) rows.push(tail)
  for (const row of rows) if (row.start === Infinity) row.start = row.commit ? row.commit.t : 0

  const wallAt = (audio: number) => {
    let before: { heard: number; wall: number } | null = null
    let after: { heard: number; wall: number } | null = null
    for (const chunk of chunks) {
      const heard = chunk.audio ?? 0
      if (heard <= audio) before = { heard, wall: chunk.t }
      else {
        after = { heard, wall: chunk.t }
        break
      }
    }
    if (!before && !after) return audio
    if (!before) return Math.max(0, after!.wall - (after!.heard - audio))
    if (!after) return before.wall + (audio - before.heard)
    const heardSpan = after.heard - before.heard
    if (heardSpan <= 0) return before.wall
    return before.wall + ((after.wall - before.wall) * (audio - before.heard)) / heardSpan
  }
  // The playhead sweeps the wall axis at playback speed, offset by how late audio usually
  // reached the pipeline. Following wallAt instead makes it stall on every decode and jump
  // on the catch-up chunk after it (bench waits for listen() before feeding the next chunk).
  const lags = chunks
    .filter((c) => c.audio != null)
    .map((c) => c.t - c.audio!)
    .sort((a, b) => a - b)
  const feedLag = lags.length ? lags[Math.floor(lags.length / 2)] : 0
  return { rows, end, wallAt, feedLag }
}

/**
 * A compare target is another model on the same datapoint: fed the same data (same
 * dataset, target and augmentation -- clean and cafe share a manifest but not the
 * sound), a different pipeline, and a row for the item on screen.
 */
export function compareCandidates(runs: CatalogRun[], run: RunInfo, itemId: string) {
  const self = runs.find((r) => r.name === run.dir)
  const pipeline = JSON.stringify(run.pipeline_config)
  return runs.filter(
    (r) =>
      r.name !== run.dir &&
      r.data_key &&
      r.data_key === self?.data_key &&
      (!r.dataset_sha || !run.dataset_sha || r.dataset_sha === run.dataset_sha) &&
      JSON.stringify(r.pipeline_config) !== pipeline &&
      r.item_ids.includes(itemId),
  )
}
