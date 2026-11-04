import { useEffect, useRef, useState, type RefObject } from "react"

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { datasetUrls } from "@/lib/datasets/api"
import { useDatasetItem, useDeleteAudio, useDeleteItem } from "@/lib/datasets/queries"
import type { AlignedWord } from "@/lib/datasets/types"
import { cn } from "@/lib/utils"

function DeleteAudioButton({ ds, id, onDeleted }: { ds: string; id: string; onDeleted: () => void }) {
  const remove = useDeleteAudio(ds)
  const [open, setOpen] = useState(false)
  return (
    <AlertDialog open={open} onOpenChange={setOpen}>
      <AlertDialogTrigger asChild>
        <Button variant="outline" size="sm">
          오디오 삭제
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>오디오만 지울까요?</AlertDialogTitle>
          <AlertDialogDescription>
            wav 파일만 지워져요. 행은 manifest에 남고, 다음 verify()에서 오디오 없음으로 잡혀요. <code className="font-mono">audio</code>는
            심볼릭 링크라 링크 너머의 원본 파일이 지워져요.
          </AlertDialogDescription>
        </AlertDialogHeader>
        {remove.error && <p className="text-sm text-destructive">{remove.error.message}</p>}
        <AlertDialogFooter>
          <AlertDialogCancel>그대로 두기</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            disabled={remove.isPending}
            onClick={(e) => {
              e.preventDefault()
              remove.mutate(id, { onSuccess: () => { setOpen(false); onDeleted() } })
            }}
          >
            오디오 지우기
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

function DeleteItemButton({ ds, id, onDeleted }: { ds: string; id: string; onDeleted: () => void }) {
  const remove = useDeleteItem(ds)
  const [open, setOpen] = useState(false)
  return (
    <AlertDialog open={open} onOpenChange={setOpen}>
      <AlertDialogTrigger asChild>
        <Button variant="outline" size="sm">
          항목 삭제
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>
            <span className="font-mono">{id}</span> 항목을 지울까요?
          </AlertDialogTitle>
          <AlertDialogDescription>manifest의 행과 오디오가 함께 지워지고 되돌릴 수 없어요.</AlertDialogDescription>
        </AlertDialogHeader>
        {remove.error && <p className="text-sm text-destructive">{remove.error.message}</p>}
        <AlertDialogFooter>
          <AlertDialogCancel>그대로 두기</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            disabled={remove.isPending}
            onClick={(e) => {
              e.preventDefault()
              remove.mutate(id, { onSuccess: () => { setOpen(false); onDeleted() } })
            }}
          >
            항목 지우기
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

function WordChips({ words, audioRef }: { words: AlignedWord[]; audioRef: RefObject<HTMLAudioElement | null> }) {
  const [now, setNow] = useState(0)
  useEffect(() => {
    const player = audioRef.current
    if (!player) return
    const onTime = () => setNow(player.currentTime)
    player.addEventListener("timeupdate", onTime)
    return () => player.removeEventListener("timeupdate", onTime)
  }, [audioRef])
  return (
    <div className="flex flex-wrap gap-1">
      {words.map((w, i) => (
        <button
          key={i}
          type="button"
          title={`${w.start.toFixed(2)} – ${w.end.toFixed(2)}s`}
          className={cn(
            "rounded-md border px-1.5 py-0.5 text-xs transition-colors",
            w.start <= now && now < w.end ? "border-primary bg-primary/10 text-primary" : "hover:bg-muted",
          )}
          onClick={() => {
            const player = audioRef.current
            if (!player) return
            player.currentTime = w.start
            void player.play()
          }}
        >
          {w.word}
        </button>
      ))}
    </div>
  )
}

export function ItemSheet({ ds, id, onClose }: { ds: string; id: string; onClose: () => void }) {
  const item = useDatasetItem(ds, id)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  return (
    <Sheet open modal={false} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full data-[side=right]:sm:max-w-lg">
        <SheetHeader className="pr-12">
          <SheetDescription>manifest.jsonl:{item.data?.line ?? "…"}</SheetDescription>
          <SheetTitle className="font-mono text-base font-medium break-all">{id}</SheetTitle>
        </SheetHeader>
        {item.isPending ? (
          <div className="px-4">
            <Skeleton className="h-48 w-full" />
          </div>
        ) : item.error ? (
          <p className="px-4 text-sm text-destructive">{item.error.message}</p>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-auto px-4 pb-4">
            {item.data.audio && (
              <audio
                ref={audioRef}
                controls
                preload="metadata"
                className="w-full"
                style={{ appearance: "auto" }}
                src={datasetUrls.audio(ds, id)}
              />
            )}
            <div className="flex flex-wrap gap-2">
              {item.data.audio && <DeleteAudioButton ds={ds} id={id} onDeleted={onClose} />}
              <DeleteItemButton ds={ds} id={id} onDeleted={onClose} />
            </div>

            {(item.data.row.reference?.transcript || Object.keys(item.data.row.reference?.translations ?? {}).length > 0) && (
              <dl className="flex flex-col gap-2 text-sm">
                {item.data.row.reference?.transcript && (
                  <div>
                    <dt className="text-xs text-muted-foreground">transcript</dt>
                    <dd>{item.data.row.reference.transcript}</dd>
                  </div>
                )}
                {Object.entries(item.data.row.reference?.translations ?? {}).map(([lang, text]) => (
                  <div key={lang}>
                    <dt className="text-xs text-muted-foreground">→ {lang}</dt>
                    <dd>{text}</dd>
                  </div>
                ))}
              </dl>
            )}

            {item.data.audio && (
              <p className={cn("text-xs", item.data.audio.problem ? "text-destructive" : "text-muted-foreground")}>
                {item.data.audio.file} · {item.data.audio.format}/{item.data.audio.subtype} {item.data.audio.sample_rate} Hz{" "}
                {item.data.audio.channels}ch · 파일 {item.data.audio.file_duration.toFixed(3)}s ·{" "}
                {Math.round(item.data.audio.bytes / 1024)} KB
                {item.data.audio.problem && ` · ${item.data.audio.problem}`}
              </p>
            )}

            {item.data.alignment && (
              <div className="flex flex-col gap-2">
                <p className={cn("text-xs text-muted-foreground", item.data.alignment.stale && "text-destructive")}>
                  정렬 · {item.data.alignment.aligner} · {item.data.alignment.words.length} 단어
                  {item.data.alignment.stale && " · 옛 전사 기준"} · 누르면 그 자리부터 재생
                </p>
                <WordChips words={item.data.alignment.words} audioRef={audioRef} />
              </div>
            )}

            <pre className="overflow-auto rounded-lg border bg-muted/40 p-3 font-mono text-xs leading-relaxed">
              {JSON.stringify(item.data.row, null, 2)}
            </pre>
          </div>
        )}
      </SheetContent>
    </Sheet>
  )
}
