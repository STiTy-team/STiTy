import { CircleIcon, MicIcon, SquareIcon } from "lucide-react"
import { useEffect, useMemo, useState } from "react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Skeleton } from "@/components/ui/skeleton"
import { useRecorder } from "@/hooks/use-recorder"
import { formatClock } from "@/lib/format"
import { useDatasetItems, useDatasetShape, useRecordTake } from "@/lib/datasets/queries"
import type { RecordMeta } from "@/lib/datasets/types"
import { cn } from "@/lib/utils"
import { ItemRow } from "@/components/datasets/item-row"
import { ItemSheet } from "@/components/datasets/item-sheet"

const RECENT = 20

const STATUS_TEXT: Record<string, string> = {
  idle: "마이크 권한 필요",
  arming: "마이크 여는 중…",
  armed: "대기 · 16 kHz",
  recording: "녹음 중",
  saving: "저장 중…",
  error: "오류",
}

export function RecorderPanel({ ds }: { ds: string }) {
  const shape = useDatasetShape(ds)
  const items = useDatasetItems(ds, "", "recent")
  const record = useRecordTake(ds)
  const [prefix, setPrefix] = useState("rec")
  const [speakers, setSpeakers] = useState(2)
  const [srcLang, setSrcLang] = useState("")
  const [speaker, setSpeaker] = useState("")
  const [savedNote, setSavedNote] = useState<string | null>(null)
  const [openItem, setOpenItem] = useState<string | null>(null)

  const declared = shape.data?.spec.languages ?? []
  const effectiveSrcLang = declared.includes(srcLang) ? srcLang : declared[0] ?? ""

  const { status, error, elapsed, toggle } = useRecorder(async (pcm) => {
    const meta: RecordMeta = { prefix, speakers, srcLang: effectiveSrcLang, speaker }
    const saved = await record.mutateAsync({ pcm, meta })
    setSavedNote("discarded" in saved && saved.discarded ? `너무 짧아 버렸어요 (${saved.duration}초)` : `저장됐어요 · ${saved.id}`)
  })

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space") return
      const target = e.target as Element | null
      if (target?.matches("input, select, button, textarea")) return
      e.preventDefault()
      toggle()
    }
    document.addEventListener("keydown", onKey)
    return () => document.removeEventListener("keydown", onKey)
  }, [toggle])

  const recent = useMemo(() => items.data?.pages[0]?.items.slice(0, RECENT) ?? [], [items.data])
  const total = items.data?.pages[0]?.total ?? 0

  return (
    <div className="grid gap-6 md:grid-cols-[22rem_1fr]">
      <div className="flex flex-col gap-5 rounded-xl border p-5">
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "size-2 rounded-full",
              status === "recording" && "bg-status-running animate-pulse motion-reduce:animate-none",
              status === "armed" && "bg-status-done",
              status === "error" && "bg-status-failed",
              (status === "idle" || status === "arming" || status === "saving") && "bg-status-queued",
            )}
          />
          <span className="text-sm text-muted-foreground">{status === "error" ? error : STATUS_TEXT[status]}</span>
        </div>

        <div className="text-center text-4xl font-medium tabular-nums">{formatClock(elapsed)}</div>

        <Button
          size="lg"
          variant={status === "recording" ? "destructive" : "default"}
          disabled={status === "arming" || status === "saving"}
          onClick={toggle}
        >
          {status === "recording" ? <SquareIcon /> : status === "idle" || status === "error" ? <MicIcon /> : <CircleIcon />}
          {status === "recording" ? "정지" : status === "armed" ? "녹음 시작" : status === "idle" || status === "error" ? "마이크 켜기" : "…"}
        </Button>
        <p className="text-center text-xs text-muted-foreground">
          <kbd className="rounded border px-1 py-0.5 font-mono">Space</kbd> 로도 시작 / 정지
        </p>

        <div className="grid grid-cols-2 gap-3">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="rec-prefix">항목 접두어</Label>
            <Input id="rec-prefix" value={prefix} maxLength={24} onChange={(e) => setPrefix(e.target.value)} />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="rec-speakers">화자 수</Label>
            <Input
              id="rec-speakers"
              type="number"
              min={1}
              max={8}
              value={speakers}
              onChange={(e) => setSpeakers(Number(e.target.value) || 0)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="rec-lang">src_lang</Label>
            <Input id="rec-lang" value={effectiveSrcLang} maxLength={16} placeholder="en" list="rec-lang-options" autoComplete="off" onChange={(e) => setSrcLang(e.target.value)} />
            <datalist id="rec-lang-options">
              {declared.map((lang) => (
                <option key={lang} value={lang} />
              ))}
            </datalist>
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="rec-speaker">speaker</Label>
            <Input id="rec-speaker" value={speaker} maxLength={64} placeholder="비워 둬도 돼요" onChange={(e) => setSpeaker(e.target.value)} />
          </div>
        </div>

        {savedNote && <p className="text-xs text-muted-foreground">{savedNote}</p>}

        <p className="text-xs text-muted-foreground">
          브라우저 DSP (에코 제거 · 잡음 억제 · 자동 게인) 는 항상 꺼진 채로 녹음해요. 잡음 억제기는 겹친 두 번째 목소리를 잡음으로
          보고 지우도록 학습돼 있어서, 겹침 데이터셋이 담으려는 것을 정확히 망가뜨려요. 녹음은 16 kHz mono s16le wav로{" "}
          <code className="font-mono">data/sessions/</code>에 떨어지고 <code className="font-mono">manifest.jsonl</code>에 행이
          붙어요.
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-medium">
          최근 항목 {total > 0 && <span className="text-muted-foreground">전체 {total} 중 마지막 {Math.min(RECENT, total)}</span>}
        </h3>
        {items.isPending ? (
          <Skeleton className="h-64 w-full" />
        ) : recent.length === 0 ? (
          <p className="text-sm text-muted-foreground">아직 녹음한 항목이 없어요.</p>
        ) : (
          <div className="rounded-xl border">
            {recent.map((item) => (
              <ItemRow key={item.id} item={item} selected={item.id === openItem} onOpen={() => setOpenItem(item.id)} />
            ))}
          </div>
        )}
      </div>

      {openItem && <ItemSheet ds={ds} id={openItem} onClose={() => setOpenItem(null)} />}
    </div>
  )
}
