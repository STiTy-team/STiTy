const dateTime = new Intl.DateTimeFormat("ko-KR", {
  month: "long",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
})

export function formatDateTime(iso: string | null): string {
  return iso ? dateTime.format(new Date(iso)) : "—"
}

const MINUTE = 60
const HOUR = 60 * MINUTE
const DAY = 24 * HOUR

export function formatDuration(totalSec: number | null): string {
  if (totalSec == null || !Number.isFinite(totalSec)) return "—"
  const sec = Math.max(0, Math.round(totalSec))
  if (sec < MINUTE) return `${sec}초`
  if (sec < HOUR) return `${Math.floor(sec / MINUTE)}분 ${sec % MINUTE}초`
  if (sec < DAY) return `${Math.floor(sec / HOUR)}시간 ${Math.floor((sec % HOUR) / MINUTE)}분`
  return `${Math.floor(sec / DAY)}일 ${Math.floor((sec % DAY) / HOUR)}시간`
}

/** mm:ss (h:mm:ss past an hour), for a running clock and tabular item durations
 * where the unit is obvious from context and doesn't need spelling out. */
export function formatClock(totalSec: number | null): string {
  if (totalSec == null || !Number.isFinite(totalSec)) return "—"
  const sec = Math.round(totalSec)
  const h = Math.floor(sec / HOUR)
  const m = Math.floor((sec % HOUR) / MINUTE)
  const s = String(sec % MINUTE).padStart(2, "0")
  return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`
}

export function secondsBetween(fromIso: string | null, toIso: string | null = null): number | null {
  if (!fromIso) return null
  const to = toIso ? new Date(toIso).getTime() : Date.now()
  return (to - new Date(fromIso).getTime()) / 1000
}

export function timeAgo(iso: string | null): string {
  const sec = secondsBetween(iso)
  if (sec == null) return "—"
  if (sec < MINUTE) return "방금"
  if (sec < HOUR) return `${Math.floor(sec / MINUTE)}분 전`
  if (sec < DAY) return `${Math.floor(sec / HOUR)}시간 전`
  if (sec < 2 * DAY) return "어제"
  if (sec < 30 * DAY) return `${Math.floor(sec / DAY)}일 전`
  return formatDateTime(iso)
}

export const shortSha = (sha: string | null) => (sha ?? "").slice(0, 10)

const MAX_REASON_CHARS = 120

export function failureReason(error: string): string {
  const lines = error.split("\n").map((line) => line.trim()).filter(Boolean)
  const found = [...lines]
    .reverse()
    .find((line) => /error|exception|failed|refused|not on/i.test(line) && !/^make(\[\d+\])?:/.test(line))
  const line = found ?? lines[0] ?? ""
  return line.length > MAX_REASON_CHARS ? `${line.slice(0, MAX_REASON_CHARS)}…` : line
}
