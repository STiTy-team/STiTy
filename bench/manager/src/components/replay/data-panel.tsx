import { useRef } from "react"

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { scrollWithin, useFollow, type Playback } from "@/hooks/use-playback"
import { langHue } from "@/lib/replay/format"
import type { QuietTurn } from "@/lib/replay/turns"
import type { ItemData } from "@/lib/replay/types"
import { cn } from "@/lib/utils"

/** Alignment words are pieces of the transcript (Korean ones can be sub-word), so find each in order. */
function MarkedWords({ text, words, activeWord }: { text: string; words: QuietTurn["words"]; activeWord: number | null }) {
  const out = []
  const lower = text.toLowerCase()
  let at = 0
  for (const w of words) {
    const k = lower.indexOf(String(w.word).toLowerCase(), at)
    if (k < 0) continue
    out.push(text.slice(at, k))
    out.push(
      <span key={w.i} className={cn("rounded-sm transition-colors", w.i === activeWord && "bg-primary text-primary-foreground")}>
        {text.slice(k, k + w.word.length)}
      </span>,
    )
    at = k + w.word.length
  }
  out.push(text.slice(at))
  return <div>{out}</div>
}

function factsOf(data: ItemData, turns: number) {
  const a = data.audio
  return [
    data.dataset.name || "데이터셋 모름",
    data.dataset.manifest_sha256 && `sha ${data.dataset.manifest_sha256.slice(0, 12)}`,
    a?.sample_rate && `${a.sample_rate} Hz · ${a.channels} ch · ${a.format}`,
    a?.file_sec != null && `파일 ${a.file_sec.toFixed(2)}s`,
    a?.augment && `augment: ${Object.keys(a.augment).join(" → ")}`,
    a?.error && `오디오: ${a.error}`,
    `구간 ${turns}개`,
    !data.manifest_rows.length && "manifest가 없어서 실행 결과의 행으로 구간을 만들었어요",
  ]
    .filter(Boolean)
    .join("  ·  ")
}

function rawText(data: ItemData) {
  const raw = []
  if (data.audio) raw.push(`# audio\n${JSON.stringify(data.audio, null, 2)}`)
  if (data.dataset.spec) raw.push(`# ${data.dataset.root}/dataset.yml\n${data.dataset.spec.trim()}`)
  for (const r of data.manifest_rows) raw.push(`# manifest.jsonl:${r.line}\n${JSON.stringify(r.row, null, 2)}`)
  return raw.join("\n\n") || "기록된 내용이 없어요."
}

export function DataPanel({
  data,
  loading,
  turns,
  targetLang,
  playback,
}: {
  data: ItemData | undefined
  loading: boolean
  turns: QuietTurn[]
  targetLang: string
  playback: Playback
}) {
  const scrollBox = useRef<HTMLDivElement>(null)
  const { now, seek, playing, jump } = playback

  const active = turns.findIndex((t) => now >= t.start && now < t.end)
  const activeWord = turns[active]?.words.find((w) => now >= w.start && now < w.end)?.i ?? null

  useFollow(active, playing, jump, () => {
    if (active >= 0) scrollWithin(scrollBox.current, scrollBox.current?.querySelector(`[data-turn="${active}"]`))
  })

  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-muted-foreground">
        {loading ? "불러오는 중…" : data ? factsOf(data, turns.length) : "이 항목에는 데이터셋 정보가 없어요."}
      </p>
      <div ref={scrollBox} className="max-h-72 overflow-auto *:data-[slot=table-container]:overflow-visible">
        <Table>
          <TableHeader className="sticky top-0 bg-card">
            <TableRow>
              <TableHead className="w-36">언어 · 화자</TableHead>
              <TableHead>참조 문장</TableHead>
              <TableHead className="w-36 text-right">레벨 (peak / rms)</TableHead>
              <TableHead className="w-32 text-right">시간</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {turns.length === 0 && (
              <TableRow>
                <TableCell colSpan={4} className="text-muted-foreground">
                  구간이 없어요.
                </TableCell>
              </TableRow>
            )}
            {turns.map((turn, n) => {
              const translation = turn.translations[targetLang]
              return (
                <TableRow
                  key={turn.id}
                  data-turn={n}
                  data-state={n === active ? "selected" : undefined}
                  className="cursor-pointer"
                  onClick={() => seek(turn.start)}
                >
                  <TableCell className="align-top whitespace-nowrap text-muted-foreground">
                    <span className="mr-1.5 inline-block size-2 rounded-full" style={{ background: langHue(turn.src_lang) }} />
                    {turn.src_lang || "?"}
                    {turn.speaker && ` · ${turn.speaker}`}
                    {turn.partial && " · 잘림"}
                  </TableCell>
                  <TableCell className="align-top whitespace-normal">
                    <MarkedWords text={turn.transcript} words={turn.words} activeWord={n === active ? activeWord : null} />
                    {translation && turn.src_lang !== targetLang && (
                      <div className="mt-0.5 text-xs text-muted-foreground">
                        {targetLang}: {translation}
                      </div>
                    )}
                  </TableCell>
                  <TableCell className="text-right align-top text-muted-foreground tabular-nums">
                    {turn.peak_dbfs == null ? (
                      "—"
                    ) : (
                      <>
                        {turn.quiet && (
                          <span className="mr-1 font-semibold text-destructive" title={turn.quiet}>
                            작음
                          </span>
                        )}
                        {turn.peak_dbfs.toFixed(1)} / {turn.rms_dbfs?.toFixed(1)}
                      </>
                    )}
                  </TableCell>
                  <TableCell className="text-right align-top text-muted-foreground tabular-nums">
                    {turn.start.toFixed(2)}–{turn.end.toFixed(2)}s
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </div>
      {data && (
        <Collapsible>
          <CollapsibleTrigger className="text-xs font-semibold text-muted-foreground hover:text-foreground">
            원본 보기: 오디오 파일 · dataset.yml · manifest 행
          </CollapsibleTrigger>
          <CollapsibleContent>
            <pre className="mt-2 max-h-60 overflow-auto rounded-lg bg-muted px-3 py-2.5 font-mono text-xs whitespace-pre-wrap text-muted-foreground">
              {rawText(data)}
            </pre>
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  )
}
