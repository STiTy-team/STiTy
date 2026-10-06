import { CircleHelpIcon, PauseIcon, PlayIcon, RotateCcwIcon, Volume2Icon, VolumeXIcon } from "lucide-react"
import type { KeyboardEvent, PointerEvent, ReactNode } from "react"

import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Toggle } from "@/components/ui/toggle"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import type { Playback } from "@/hooks/use-playback"
import { langHue, REASON_COLOR, reasonColor, seconds } from "@/lib/replay/format"
import type { PlayedItem } from "@/lib/replay/model"
import type { QuietTurn } from "@/lib/replay/turns"
import { cn } from "@/lib/utils"

const SPEEDS = [0.5, 1, 2, 4, 20]

export function Legend({ entries }: { entries: { label: string; color: string; className?: string }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
      {entries.map((entry) => (
        <span key={entry.label} className="flex items-center gap-1.5">
          <i className={cn("inline-block h-2.5 w-2.5 rounded-sm", entry.className)} style={{ background: entry.color }} />
          {entry.label}
        </span>
      ))}
    </div>
  )
}

export function HelpButton({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon-sm" className="text-muted-foreground" aria-label={label}>
          <CircleHelpIcon />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="flex w-80 flex-col gap-2.5 text-sm">
        {children}
      </PopoverContent>
    </Popover>
  )
}

const REASON_TEXT: Record<string, string> = {
  seg: "모델이 문장 끝을 표시함",
  vad: "말이 끝나고 침묵이 이어짐",
  dot: "마침표가 나옴",
  always: "chunk마다",
  finish: "오디오가 끝남",
}

function Ribbon({ item, turns, playback }: { item: PlayedItem; turns: QuietTurn[]; playback: Playback }) {
  const { now, seek } = playback
  const span = item.span
  const pct = (t: number) => `${(100 * t) / span}%`

  const onPointerDown = (ev: PointerEvent<HTMLDivElement>) => {
    const box = ev.currentTarget.getBoundingClientRect()
    const move = (e: { clientX: number }) => seek(span * Math.max(0, Math.min(1, (e.clientX - box.left) / box.width)))
    move(ev)
    const up = () => {
      removeEventListener("pointermove", move)
      removeEventListener("pointerup", up)
    }
    addEventListener("pointermove", move)
    addEventListener("pointerup", up)
  }
  const onKeyDown = (ev: KeyboardEvent<HTMLDivElement>) => {
    const step = ev.shiftKey ? 1 : 0.1
    if (ev.key === "ArrowRight") seek(now + step)
    else if (ev.key === "ArrowLeft") seek(now - step)
    else return
    ev.preventDefault()
  }

  return (
    <div
      className="relative h-12 cursor-pointer select-none outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
      role="slider"
      tabIndex={0}
      aria-label="재생 위치"
      aria-valuemin={0}
      aria-valuemax={span}
      aria-valuenow={Number(now.toFixed(2))}
      onPointerDown={onPointerDown}
      onKeyDown={onKeyDown}
    >
      <div className="absolute inset-x-0 top-3.5 bottom-4 overflow-hidden rounded bg-muted">
        {item.played
          .filter((e) => e.type === "translated" || e.type === "speech")
          .map((e, i) => {
            const commit = e.type === "translated"
            return (
              <div
                key={i}
                className={cn("absolute w-0.5 rounded-[1px]", commit ? "top-0.5 h-7" : "top-2 h-5.5")}
                style={{ left: pct(e.at), background: commit ? reasonColor(e.commit_reason) : "var(--muted-foreground)" }}
                title={commit ? `commit ${e.commit_reason ?? ""} · ${seconds(e.audio)}` : "말 끝"}
              />
            )
          })}
        {turns.map((turn) => (
          <div
            key={turn.id}
            className={cn("absolute bottom-0 h-1", turn.quiet ? "opacity-35" : "opacity-85")}
            style={{ left: pct(turn.start), width: pct(turn.end - turn.start), background: langHue(turn.src_lang) }}
            title={`${turn.src_lang} ${turn.start.toFixed(2)}–${turn.end.toFixed(2)}s${turn.quiet ? ` · 작음: ${turn.quiet}` : ""}`}
          />
        ))}
      </div>
      <div className="pointer-events-none absolute top-1.5 bottom-2 w-[1.5px] bg-foreground" style={{ left: pct(now) }}>
        <div className="absolute -top-1 -left-[3px] size-2 rounded-full bg-foreground" />
      </div>
    </div>
  )
}

export function PlayerCard({
  item,
  turns,
  playback,
  speed,
  onSpeed,
  soundOn,
  onSound,
}: {
  item: PlayedItem
  turns: QuietTurn[]
  playback: Playback
  speed: number
  onSpeed: (speed: number) => void
  soundOn: boolean
  onSound: (on: boolean) => void
}) {
  const { now, playing, toggle, restart } = playback
  return (
    <Card>
      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={toggle} className="min-w-24">
            {playing ? <PauseIcon /> : <PlayIcon />}
            {playing ? "일시정지" : "재생"}
          </Button>
          <Button variant="outline" onClick={restart}>
            <RotateCcwIcon />
            처음부터
          </Button>
          <Toggle variant="outline" pressed={soundOn} onPressedChange={onSound} aria-label={soundOn ? "소리 끄기" : "소리 켜기"}>
            {soundOn ? <Volume2Icon /> : <VolumeXIcon />}
          </Toggle>
          <ToggleGroup
            type="single"
            variant="outline"
            spacing={0}
            value={String(speed)}
            onValueChange={(v) => v && onSpeed(Number(v))}
            aria-label="재생 속도"
          >
            {SPEEDS.map((s) => (
              <ToggleGroupItem key={s} value={String(s)} className="tabular-nums">
                {s === 20 ? "최대" : `${s}×`}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <span className="flex-1" />
          <div className="text-sm text-muted-foreground tabular-nums">
            <b className="font-semibold text-foreground">{now.toFixed(2)}</b> / {seconds(item.span)}
            <span className="mx-2">·</span>
            <code className="font-mono text-xs">{item.id}</code>
          </div>
          <HelpButton label="재생 막대 표시 설명">
            <p className="font-medium">막대 위 표시</p>
            <ul className="flex flex-col gap-1.5">
              {Object.entries(REASON_COLOR).map(([reason, color]) => (
                <li key={reason} className="flex items-center gap-2">
                  <i className="inline-block size-2.5 rounded-sm" style={{ background: color }} />
                  <span className="font-mono text-xs">{reason}</span>
                  <span className="text-muted-foreground">{REASON_TEXT[reason]}</span>
                </li>
              ))}
            </ul>
            <p className="text-muted-foreground">
              긴 표시는 commit, 짧은 회색 표시는 말 끝이에요. 아래 얇은 색 띠는 참조 구간이에요. 스페이스로 재생·정지, 막대에서 화살표 키로 이동할 수
              있어요.
            </p>
          </HelpButton>
        </div>
        <Ribbon item={item} turns={turns} playback={playback} />
      </CardContent>
    </Card>
  )
}
