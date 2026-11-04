import type { WeeklyBlock } from "@/lib/api"

const MINUTES_PER_DAY = 24 * 60
const MINUTE_MS = 60_000
const DAY_MS = MINUTES_PER_DAY * MINUTE_MS
const REFERENCE_MONDAY = Date.UTC(2024, 0, 1)

export const REFERENCE_WEEK_START = new Date(REFERENCE_MONDAY).toISOString()

export function toMinutes(clock: string): number {
  const [hours, minutes] = clock.split(":").map(Number)
  return hours * 60 + minutes
}

export function toClock(minutes: number): string {
  const pad = (value: number) => String(value).padStart(2, "0")
  return `${pad(Math.floor(minutes / 60))}:${pad(minutes % 60)}`
}

export function blockRange(block: WeeklyBlock): { start: Date; end: Date } {
  const day = REFERENCE_MONDAY + block.day * DAY_MS
  return {
    start: new Date(day + toMinutes(block.start) * MINUTE_MS),
    end: new Date(day + toMinutes(block.end) * MINUTE_MS),
  }
}

export function rangeToBlocks(start: Date, end: Date): WeeklyBlock[] {
  const from = Math.max(start.getTime(), REFERENCE_MONDAY) - REFERENCE_MONDAY
  const to = Math.min(end.getTime(), REFERENCE_MONDAY + 7 * DAY_MS) - REFERENCE_MONDAY
  const blocks: WeeklyBlock[] = []
  for (let day = Math.floor(from / DAY_MS); day * DAY_MS < to; day++) {
    const dayStart = Math.max(from, day * DAY_MS) - day * DAY_MS
    const dayEnd = Math.min(to, (day + 1) * DAY_MS) - day * DAY_MS
    if (dayEnd > dayStart) {
      blocks.push({ day, start: toClock(dayStart / MINUTE_MS), end: toClock(dayEnd / MINUTE_MS) })
    }
  }
  return blocks
}

export function normalizeBlocks(blocks: WeeklyBlock[]): WeeklyBlock[] {
  const sorted = [...blocks].sort((a, b) => a.day - b.day || toMinutes(a.start) - toMinutes(b.start))
  const merged: WeeklyBlock[] = []
  for (const block of sorted) {
    const last = merged.at(-1)
    if (last && last.day === block.day && toMinutes(block.start) <= toMinutes(last.end)) {
      if (toMinutes(block.end) > toMinutes(last.end)) last.end = block.end
    } else {
      merged.push({ ...block })
    }
  }
  return merged
}

export function sameBlocks(a: WeeklyBlock[], b: WeeklyBlock[]): boolean {
  return JSON.stringify(normalizeBlocks(a)) === JSON.stringify(normalizeBlocks(b))
}
