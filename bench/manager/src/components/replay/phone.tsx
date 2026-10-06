import { useEffect, useRef } from "react"

import { langHue } from "@/lib/replay/format"
import { pastEdge, type PlayedEvent, type PlayedItem } from "@/lib/replay/model"
import { cn } from "@/lib/utils"

// Inside the phone the bubbles follow DESIGN.md's conversation rules (mine blue, partner
// grey, language hue only on the avatar) -- the panel shows what the device shows, so its
// colors are the app's own and do not follow this page's theme.

function Bubble({ lang, main, sub, mine, live }: { lang: string; main: string; sub?: string; mine: boolean; live?: boolean }) {
  return (
    <div
      className={cn(
        "flex items-end gap-2 px-5 pt-2",
        mine && "flex-row-reverse",
        !live && "animate-in fade-in-0 slide-in-from-bottom-2 motion-reduce:animate-none",
      )}
    >
      <div
        className="grid size-7 shrink-0 place-items-center rounded-full text-[9px] font-bold text-white"
        style={{ background: langHue(lang) }}
      >
        {lang.toUpperCase()}
      </div>
      <div
        className={cn(
          "max-w-[72%] rounded-[14px] p-3",
          mine ? "rounded-br-[4px]" : "rounded-bl-[4px]",
          live ? (mine ? "bg-[#F0F3FA] text-[#37579F]" : "bg-[#F9FAFB] text-[#4E5968]") : mine ? "bg-[#4268BD] text-white" : "bg-[#F2F4F6] text-[#191F28]",
        )}
      >
        <div className="text-[15px] leading-normal">{main}</div>
        {sub && <div className={cn("mt-1 text-[13px] leading-normal", mine ? "text-white/85" : "text-[#4E5968]")}>{sub}</div>}
      </div>
    </div>
  )
}

function LangPill({ lang, mine }: { lang: string; mine?: boolean }) {
  return (
    <div
      className={cn(
        "flex h-7 items-center gap-1.5 rounded-full border px-3 text-[13px] font-semibold",
        mine ? "border-[#F0F3FA] bg-[#F0F3FA] text-[#4268BD]" : "border-[#E5E8EB] text-[#4E5968]",
      )}
    >
      <span className="size-2 rounded-full" style={{ background: langHue(lang) }} />
      {(lang || "?").toUpperCase()}
    </div>
  )
}

const lowerLang = (e: PlayedEvent) => (e.language ?? "").toLowerCase()

export function Phone({
  item,
  now,
  playing,
  srcLang,
  targetLang,
  instrumented = true,
}: {
  item: PlayedItem | null
  now: number
  playing: boolean
  srcLang: string
  targetLang: string
  instrumented?: boolean
}) {
  const feed = useRef<HTMLDivElement>(null)
  const events = item?.played ?? []
  const edge = pastEdge(events, now)
  const past = events.slice(0, edge)
  const finals = past.filter((e) => e.type === "translated")

  let live: PlayedEvent | null = null
  for (let i = edge - 1; i >= 0; i--) {
    const e = events[i]
    if (e.type === "translated" || e.type === "item_open") break
    if (e.type === "partial") {
      live = e
      break
    }
  }
  const liveText = instrumented && live?.text ? live.text : ""

  useEffect(() => {
    if (feed.current) feed.current.scrollTop = feed.current.scrollHeight
  }, [finals.length, liveText])

  const waiting = item?.pending.some(([from, to]) => from <= now && to > now) ?? false
  const speaking = past.filter((e) => e.type === "vad_speech_start" || e.type === "speech").at(-1)
  const hearing = speaking?.type === "vad_speech_start"

  const empty = !item ? "—" : now > 0 ? "Nothing committed yet" : "Press play"

  return (
    <div className="h-[720px] w-[384px] max-w-full rounded-[44px] bg-[#191F28] p-[11px] shadow-xl">
      <div className="relative flex h-full flex-col overflow-hidden rounded-[34px] bg-white text-[#191F28]">
        <div className="absolute top-[9px] left-1/2 z-10 h-6 w-[104px] -translate-x-1/2 rounded-full bg-[#191F28]" />
        <div className="flex h-[42px] shrink-0 items-center justify-between px-[22px] text-xs font-semibold tabular-nums">
          <span>9:41</span>
          <span>STiTy</span>
        </div>
        <div className="flex shrink-0 items-center gap-2 border-b border-[#E5E8EB] px-5 pt-1 pb-3">
          <LangPill lang={srcLang} mine />
          <span className="text-[13px] text-[#B0B8C1]">→</span>
          <LangPill lang={targetLang} />
        </div>
        <div ref={feed} className="flex-1 overflow-y-auto scroll-smooth pt-4 pb-2">
          {finals.length ? (
            finals.map((e, i) => (
              <Bubble
                key={i}
                lang={lowerLang(e)}
                mine={lowerLang(e) === srcLang}
                main={e.translation || e.original || ""}
                sub={e.translation ? e.original : undefined}
              />
            ))
          ) : (
            <div className="grid h-full place-items-center text-[15px] text-[#8B95A1]">{empty}</div>
          )}
          {liveText && live && <Bubble lang={lowerLang(live)} mine={lowerLang(live) === srcLang} main={liveText} live />}
        </div>
        {instrumented && (
          <>
            {waiting && !liveText && (
              <div className="pointer-events-none absolute inset-x-0 bottom-24 flex justify-center">
                <span className="rounded-full bg-[#F2F4F6] px-4 py-1.5 text-lg leading-5 font-extrabold tracking-[2px] text-[#8B95A1]">
                  •••
                </span>
              </div>
            )}
            <div className="mx-5 mb-5 flex h-14 shrink-0 items-center gap-2.5 rounded-2xl bg-[#F2F4F6] px-4">
              <span className={cn("size-2 shrink-0 rounded-full bg-[#B0B8C1]", hearing && "animate-pulse bg-[#7A9030]")} />
              <span className={cn("flex-1 text-[15px] font-semibold", hearing && "text-[#617326]")}>
                {hearing ? "Listening" : playing ? "Pause between utterances" : "Idle"}
              </span>
              <span className="text-[13px] text-[#6B7684] tabular-nums">{now.toFixed(1)}s</span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
