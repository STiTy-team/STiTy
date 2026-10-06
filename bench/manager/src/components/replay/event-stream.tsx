import { ChevronDownIcon } from "lucide-react"
import { memo, useMemo, useRef, useState, type ReactNode } from "react"

import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { scrollWithin, useFollow, type Playback } from "@/hooks/use-playback"
import { reasonColor } from "@/lib/replay/format"
import { laneOf, pastEdge, type PlayedEvent } from "@/lib/replay/model"
import { cn } from "@/lib/utils"

const NOISY = ["chunk", "partial"]
const LEAD = ["original", "translation", "commit_reason", "detail", "error", "msg"]
const SKIP = new Set(["t", "type", "tag", "item", "session", "audio", "at", ...LEAD])
const EXTRA_COLUMNS = { lag: "지연 (lag)", item: "항목 (item)" } as const
type ExtraColumn = keyof typeof EXTRA_COLUMNS

const fmt = (n: number) => (Number.isInteger(n) ? String(n) : n.toFixed(3))
const fieldText = (v: unknown) => (typeof v === "number" ? fmt(v) : typeof v === "object" ? JSON.stringify(v) : String(v))

/** Fields every event carries with the same value; shown once above the table instead of on every row. */
function sharedFields(events: PlayedEvent[]): [string, string][] {
  if (events.length < 2) return []
  const out: [string, string][] = []
  for (const [k, v] of Object.entries(events[0])) {
    if (SKIP.has(k) || v == null || v === "") continue
    const text = fieldText(v)
    if (events.every((e) => e[k] != null && fieldText(e[k]) === text)) out.push([k, text])
  }
  return out
}

function Fields({ event, shared }: { event: PlayedEvent; shared: Set<string> }) {
  const out: ReactNode[] = []
  for (const k of LEAD) {
    const v = event[k]
    if (v == null || v === "") continue
    out.push(
      k === "commit_reason" ? (
        <span key={k} className="inline-block rounded px-1.5 text-[11px] font-semibold text-white" style={{ background: reasonColor(String(v)) }}>
          {String(v)}
        </span>
      ) : (
        <b key={k} className="font-sans text-[13px] font-normal text-foreground">
          {String(v)}
        </b>
      ),
    )
  }
  for (const [k, v] of Object.entries(event)) {
    if (SKIP.has(k) || shared.has(k) || v == null || v === "") continue
    out.push(`${k}=${fieldText(v)}`)
  }
  return out.flatMap((part, i) => (i ? [" ", part] : [part]))
}

const EventRow = memo(function EventRow({
  event,
  index,
  state,
  shared,
  columns,
  onSeek,
}: {
  event: PlayedEvent
  index: number
  state: "past" | "now" | "future"
  shared: Set<string>
  columns: Set<ExtraColumn>
  onSeek: (at: number) => void
}) {
  const lag = event.t != null ? event.t - event.audio : null
  return (
    <tr
      data-event={index}
      onClick={() => onSeek(event.at)}
      className={cn("cursor-pointer border-b hover:bg-muted/50", state === "future" && "opacity-30", state === "now" && "bg-accent")}
    >
      <td className="px-2.5 py-1.5 text-right align-top text-muted-foreground tabular-nums">{event.audio.toFixed(2)}</td>
      {columns.has("lag") && (
        <td className="px-2.5 py-1.5 text-right align-top text-muted-foreground tabular-nums">
          {lag == null ? "" : `${lag >= 0 ? "+" : ""}${lag.toFixed(2)}`}
        </td>
      )}
      {columns.has("item") && <td className="px-2.5 py-1.5 align-top font-mono whitespace-nowrap text-muted-foreground">{event.item ?? ""}</td>}
      <td className="px-2.5 py-1.5 align-top font-mono font-medium whitespace-nowrap">{laneOf(event)}</td>
      <td className="px-2.5 py-1.5 align-top font-mono break-words text-muted-foreground">
        <Fields event={event} shared={shared} />
      </td>
    </tr>
  )
})

const searchText = (e: PlayedEvent) => `${laneOf(e)} ${e.item ?? ""} ${JSON.stringify(e)}`.toLowerCase()

function CheckList<T extends string>({ items, checked, onFlip }: { items: [T, string][]; checked: (key: T) => boolean; onFlip: (key: T) => void }) {
  return (
    <div className="grid grid-cols-2 gap-x-3 gap-y-1">
      {items.map(([key, label]) => (
        <Label key={key} className="min-h-7 font-mono text-xs font-normal">
          <Checkbox checked={checked(key)} onCheckedChange={() => onFlip(key)} />
          {label}
        </Label>
      ))}
    </div>
  )
}

export function EventStream({ events, playback }: { events: PlayedEvent[]; playback: Playback }) {
  const [hidden, setHidden] = useState(() => new Set(NOISY))
  const [columns, setColumns] = useState(() => new Set<ExtraColumn>())
  const [query, setQuery] = useState("")
  const box = useRef<HTMLDivElement>(null)
  const { now, playing, jump, seek } = playback

  const edge = pastEdge(events, now)
  const types = useMemo(() => [...new Set(events.map(laneOf))].sort(), [events])
  const texts = useMemo(() => events.map(searchText), [events])
  const shared = useMemo(() => sharedFields(events), [events])
  const sharedKeys = useMemo(() => new Set(shared.map(([k]) => k)), [shared])
  const q = query.trim().toLowerCase()
  const shown = useMemo(
    () => events.map((e, i) => !hidden.has(laneOf(e)) && (!q || texts[i].includes(q))),
    [events, texts, hidden, q],
  )

  // The newest past event is often a filtered-out chunk or partial; follow the newest shown one.
  useFollow(edge, playing, jump, () => {
    let i = edge - 1
    while (i >= 0 && !shown[i]) i--
    if (i >= 0) scrollWithin(box.current, box.current?.querySelector(`[data-event="${i}"]`))
  })

  const rows = useMemo(
    () =>
      events.map((e, i) =>
        shown[i] ? (
          <EventRow
            key={i}
            event={e}
            index={i}
            state={i >= edge ? "future" : i === edge - 1 ? "now" : "past"}
            shared={sharedKeys}
            columns={columns}
            onSeek={seek}
          />
        ) : null,
      ),
    [events, shown, edge, sharedKeys, columns, seek],
  )

  const flip = <T,>(set: (fn: (prev: Set<T>) => Set<T>) => void) => (key: T) =>
    set((prev) => {
      const next = new Set(prev)
      if (!next.delete(key)) next.add(key)
      return next
    })
  const visibleTypes = types.filter((type) => !hidden.has(type)).length

  return (
    <Card className="h-[720px] gap-0 overflow-hidden py-0">
      <div className="flex flex-wrap items-center gap-2 border-b px-4 py-3">
        <Input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="내용으로 찾기"
          aria-label="이벤트 찾기"
          className="min-w-36 flex-1"
        />
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm">
              종류 {visibleTypes}/{types.length}
              <ChevronDownIcon className="opacity-50" />
            </Button>
          </PopoverTrigger>
          <PopoverContent align="end" className="flex w-72 flex-col gap-2">
            <CheckList items={types.map((type) => [type, type])} checked={(type) => !hidden.has(type)} onFlip={flip(setHidden)} />
          </PopoverContent>
        </Popover>
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm">
              열
              <ChevronDownIcon className="opacity-50" />
            </Button>
          </PopoverTrigger>
          <PopoverContent align="end" className="w-56">
            <CheckList
              items={Object.entries(EXTRA_COLUMNS) as [ExtraColumn, string][]}
              checked={(column) => columns.has(column)}
              onFlip={flip(setColumns)}
            />
          </PopoverContent>
        </Popover>
      </div>
      {shared.length > 0 && (
        <p className="border-b bg-muted/40 px-4 py-2 text-xs break-all text-muted-foreground">
          모든 이벤트에 같은 값이라 한 번만 보여줘요:{" "}
          <span className="font-mono">{shared.map(([k, v]) => `${k}=${v}`).join("  ")}</span>
        </p>
      )}
      <div ref={box} className="flex-1 overflow-auto">
        <table className="w-full border-collapse text-xs">
          <thead className="sticky top-0 z-10 bg-card text-muted-foreground">
            <tr className="border-b">
              <th className="w-16 px-2.5 py-2 text-right font-semibold">audio</th>
              {columns.has("lag") && <th className="w-14 px-2.5 py-2 text-right font-semibold">lag</th>}
              {columns.has("item") && <th className="w-20 px-2.5 py-2 text-left font-semibold">item</th>}
              <th className="w-28 px-2.5 py-2 text-left font-semibold">event</th>
              <th className="px-2.5 py-2 text-left font-semibold">fields</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        {!events.length && <p className="p-6 text-sm text-muted-foreground">이 항목에는 이벤트가 없어요.</p>}
      </div>
    </Card>
  )
}
