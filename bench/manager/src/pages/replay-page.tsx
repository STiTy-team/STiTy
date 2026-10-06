import { useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { Navigate, useSearchParams } from "react-router"

import { Notice, PageBody, PageHeader } from "@/components/page-header"
import { DataPanel } from "@/components/replay/data-panel"
import { EventStream } from "@/components/replay/event-stream"
import { ItemMetricTiles, PipelineConfig, RunSummary } from "@/components/replay/kv"
import { LayerTiming } from "@/components/replay/layer-timing"
import { Phone } from "@/components/replay/phone"
import { PlayerCard } from "@/components/replay/player-card"
import { Reference } from "@/components/replay/readout"
import { ItemStats, ReplayControls } from "@/components/replay/replay-header"
import { Card, CardContent } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useAudioSync, usePlayback } from "@/hooks/use-playback"
import { compareCandidates, prepareItem } from "@/lib/replay/model"
import { replayUrls, useItemData, useReplay, useReplayItem } from "@/lib/replay/queries"
import { quietTurns } from "@/lib/replay/turns"
import type { CatalogRun, ReplayItem, ReplayPayload } from "@/lib/replay/types"
import { cn } from "@/lib/utils"

const NO_COMPARE = "__none"
const TAB_KEY = "replay.details.tab"
const DETAIL_TABS = [
  ["timing", "타이밍"],
  ["reference", "참조 문장"],
  ["data", "데이터"],
  ["metrics", "항목 지표"],
  ["config", "파이프라인 설정"],
  ["summary", "실행 요약"],
] as const
type DetailTab = (typeof DETAIL_TABS)[number][0]

function readTab(): DetailTab {
  try {
    const saved = localStorage.getItem(TAB_KEY)
    return DETAIL_TABS.find(([id]) => id === saved)?.[0] ?? "timing"
  } catch {
    return "timing"
  }
}

function saveTab(tab: DetailTab) {
  try {
    localStorage.setItem(TAB_KEY, tab)
  } catch {
    /* per-viewer preference only */
  }
}

function ColumnHead({ title, description, children }: { title?: string; description: ReactNode; children?: ReactNode }) {
  return (
    <div className="mb-3 flex min-h-9 flex-wrap items-center gap-x-3 gap-y-2">
      {title && <h2 className="font-semibold">{title}</h2>}
      <p className="text-sm text-muted-foreground">{description}</p>
      <span className="flex-1" />
      {children}
    </div>
  )
}

function ComparePicker({ runs, value, onChange }: { runs: CatalogRun[]; value: string | null; onChange: (run: string | null) => void }) {
  return (
    <Select value={value ?? NO_COMPARE} onValueChange={(v) => onChange(v === NO_COMPARE ? null : v)}>
      <SelectTrigger size="sm" className="max-w-full font-mono text-xs" aria-label="비교할 실행">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={NO_COMPARE}>다른 실행과 비교</SelectItem>
        {runs.map((run) => (
          <SelectItem key={run.name} value={run.name} className="font-mono text-xs">
            {run.name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function Side({ caption, children }: { caption?: string; children: ReactNode }) {
  return (
    <section className="flex min-w-0 flex-col gap-3">
      {caption && <p className="font-mono text-xs break-all text-muted-foreground">{caption}</p>}
      {children}
    </section>
  )
}

function useSpaceToToggle(toggle: () => void) {
  const latest = useRef(toggle)
  useEffect(() => {
    latest.current = toggle
  })
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (ev.code !== "Space") return
      if ((ev.target as Element).closest("input, textarea, button, [role=combobox], [role=option], [role=tab], [contenteditable]")) return
      ev.preventDefault()
      latest.current()
    }
    addEventListener("keydown", onKey)
    return () => removeEventListener("keydown", onKey)
  }, [])
}

function ItemReplay({
  data,
  item,
  compareRun,
  onCompare,
  speed,
  onSpeed,
  soundOn,
  onSound,
}: {
  data: ReplayPayload
  item: ReplayItem
  compareRun: string | null
  onCompare: (run: string | null) => void
  speed: number
  onSpeed: (speed: number) => void
  soundOn: boolean
  onSound: (on: boolean) => void
}) {
  const run = data.run.dir
  const played = useMemo(() => prepareItem(item), [item])
  const playback = usePlayback(played.span, speed)
  const [tab, setTab] = useState(readTab)
  useSpaceToToggle(playback.toggle)

  const audio = useRef<HTMLAudioElement>(null)
  useAudioSync(audio, {
    url: soundOn && item.audio_url ? replayUrls.audio(run, item.id) : null,
    now: playback.now,
    playing: playback.playing,
    speed,
  })

  const itemData = useItemData(run, item.id)
  const turns = useMemo(() => quietTurns(itemData.data), [itemData.data])

  const candidates = compareCandidates(data.runs, data.run, item.id)
  const comparing = compareRun != null && candidates.some((r) => r.name === compareRun)
  const compareInfo = comparing ? candidates.find((r) => r.name === compareRun)! : null
  const compareQuery = useReplayItem(comparing ? compareRun : null, item.id)
  const compared = useMemo(() => (compareQuery.data ? prepareItem(compareQuery.data) : null), [compareQuery.data])

  const picker = candidates.length > 0 && <ComparePicker runs={candidates} value={comparing ? compareRun : null} onChange={onCompare} />
  const compareCaption = compareQuery.isPending
    ? "불러오는 중…"
    : compareQuery.error
      ? `${compareRun}에는 ${item.id} 항목이 없어요`
      : compared?.wer == null
        ? item.id
        : `${item.id} · WER ${compared.wer.toFixed(2)}`

  const pair = cn("grid items-start gap-6", comparing && "xl:grid-cols-2")
  const ours = comparing ? data.run.name : undefined
  const theirs = compareInfo?.name

  return (
    <div className="flex flex-col gap-6">
      <audio ref={audio} preload="auto" />
      <ItemStats metrics={item.metrics} />
      <PlayerCard item={played} turns={turns} playback={playback} speed={speed} onSpeed={onSpeed} soundOn={soundOn} onSound={onSound} />

      <div className={cn("grid items-start gap-6", comparing ? "xl:grid-cols-2" : "xl:grid-cols-[384px_minmax(0,1fr)]")}>
        <section>
          <ColumnHead title="휴대폰 화면" description={<>클라이언트가 <code>final</code>로 그린 화면</>}>
            {!comparing && picker}
          </ColumnHead>
          <Phone item={played} now={playback.now} playing={playback.playing} srcLang={item.src_lang} targetLang={item.target_lang} />
        </section>
        {comparing && (
          <section>
            <ColumnHead description={compareCaption}>{picker}</ColumnHead>
            <Phone
              item={compared}
              now={playback.now}
              playing={playback.playing}
              srcLang={item.src_lang}
              targetLang={item.target_lang}
              instrumented={false}
            />
          </section>
        )}
        <section>
          <ColumnHead title="이벤트 흐름" description="휴대폰에 안 보이는 것까지 모든 이벤트, 오디오 순서" />
          <EventStream events={played.played} playback={playback} />
        </section>
        {comparing && (
          <section>
            <ColumnHead title="이벤트 흐름" description="비교 실행의 이벤트" />
            <EventStream events={compared?.played ?? []} playback={playback} />
          </section>
        )}
      </div>

      <Card className="gap-0 py-0">
        <Tabs
          value={tab}
          onValueChange={(next) => {
            setTab(next as DetailTab)
            saveTab(next as DetailTab)
          }}
        >
          <div className="overflow-x-auto border-b px-4">
            <TabsList variant="line" className="h-12 gap-4">
              {DETAIL_TABS.map(([id, label]) => (
                <TabsTrigger key={id} value={id} className="flex-none px-1">
                  {label}
                </TabsTrigger>
              ))}
            </TabsList>
          </div>
          <CardContent className="py-5">
            <TabsContent value="timing" className={pair}>
              <Side caption={ours}>
                <LayerTiming item={played} playback={playback} />
              </Side>
              {comparing && (
                <Side caption={theirs}>
                  <LayerTiming item={compared} playback={playback} />
                </Side>
              )}
            </TabsContent>
            <TabsContent value="reference">
              <Reference item={item} />
            </TabsContent>
            <TabsContent value="data">
              <DataPanel data={itemData.data} loading={itemData.isPending} turns={turns} targetLang={item.target_lang} playback={playback} />
            </TabsContent>
            <TabsContent value="metrics" className={pair}>
              <Side caption={ours}>
                <ItemMetricTiles metrics={item.metrics} />
              </Side>
              {comparing && (
                <Side caption={theirs}>
                  <ItemMetricTiles metrics={compareQuery.data?.metrics} />
                </Side>
              )}
            </TabsContent>
            <TabsContent value="config" className={pair}>
              <Side caption={ours}>
                <PipelineConfig config={data.run.pipeline_config} />
              </Side>
              {compareInfo && (
                <Side caption={theirs}>
                  <PipelineConfig config={compareInfo.pipeline_config} />
                </Side>
              )}
            </TabsContent>
            <TabsContent value="summary" className={pair}>
              <Side caption={ours}>
                <p className="mb-1 text-sm text-muted-foreground">불러온 항목만이 아니라 실행 전체를 모은 값이에요.</p>
                <RunSummary metrics={data.run.summary_metrics} running={data.run.status === "running"} />
              </Side>
              {compareInfo && (
                <Side caption={theirs}>
                  <p className="mb-1 text-sm text-muted-foreground">비교 실행 전체를 모은 값이에요.</p>
                  <RunSummary metrics={compareInfo.summary_metrics} running={compareInfo.status === "running"} />
                </Side>
              )}
            </TabsContent>
          </CardContent>
        </Tabs>
      </Card>
    </div>
  )
}

export function ReplayPage() {
  const [params, setParams] = useSearchParams()
  const [speed, setSpeed] = useState(1)
  const [soundOn, setSoundOn] = useState(true)
  const runName = params.get("run")
  const replay = useReplay(runName)
  const data = replay.data
  const itemId = params.get("item") ?? data?.items[0]?.id ?? null
  const item = useReplayItem(data?.run.dir ?? null, itemId, data?.items[0])

  const update = (next: Record<string, string | null>) =>
    setParams((prev) => {
      const merged = new URLSearchParams(prev)
      for (const [key, value] of Object.entries(next)) {
        if (value == null) merged.delete(key)
        else merged.set(key, value)
      }
      return merged
    })

  if (!runName) return <Navigate to="/runs" replace />

  return (
    <>
      <PageHeader title="Replay" />
      <PageBody className="max-w-[1380px]">
        {data && <ReplayControls data={data} itemId={itemId ?? ""} onItem={(id) => update({ run: data.run.dir, item: id })} />}
        {replay.isPending ? (
          <Skeleton className="h-9 w-96" />
        ) : replay.error ? (
          <Notice>리플레이를 불러오지 못했어요: {replay.error.message}</Notice>
        ) : !data ? null : item.isPending ? (
          <Skeleton className="h-40 w-full" />
        ) : item.error ? (
          <Notice>
            {data.run.dir}에서 {itemId} 항목을 불러오지 못했어요: {item.error.message}
          </Notice>
        ) : (
          <ItemReplay
            key={`${data.run.dir}/${item.data.id}`}
            data={data}
            item={item.data}
            compareRun={params.get("compare")}
            onCompare={(compare) => update({ run: data.run.dir, compare })}
            speed={speed}
            onSpeed={setSpeed}
            soundOn={soundOn}
            onSound={setSoundOn}
          />
        )}
      </PageBody>
    </>
  )
}
