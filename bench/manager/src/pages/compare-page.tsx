import { CircleHelpIcon, InfoIcon, TriangleAlertIcon } from "lucide-react"
import { useMemo, useState, type ReactNode } from "react"
import { useSearchParams } from "react-router"

import { DatasetPicker } from "@/components/compare/dataset-picker"
import { DistributionChart } from "@/components/compare/distribution-chart"
import { ModelMenu } from "@/components/compare/model-menu"
import { RunBoard } from "@/components/compare/run-board"
import { ScatterChart } from "@/components/compare/scatter-chart"
import { Notice, PageBody, PageHeader } from "@/components/page-header"
import { MetaLine } from "@/components/replay/replay-header"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useDistributions, useOverview } from "@/lib/compare/queries"
import { missingRuns, scoreKeys } from "@/lib/compare/runs"
import { BETTER_TEXT, SCORES } from "@/lib/compare/scores"
import type { RunOverview } from "@/lib/compare/types"
import { byGroup, groupOf, LANE_COLORS, stampText } from "@/lib/replay/format"

const MAX_COLS = 7
const ACCURACY_GROUPS = new Set(["Transcription accuracy", "Translation accuracy", "Language detection"])
const LATENCY_GROUPS = new Set(["Transcription latency", "Translation latency"])
const CHARTS = ["distribution", "scatter"] as const

function Help({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon-sm" className="text-muted-foreground" aria-label={label}>
          <CircleHelpIcon />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="flex w-80 flex-col gap-2 text-sm">
        {children}
      </PopoverContent>
    </Popover>
  )
}

function KeySelect({ label, keys, value, onChange }: { label: string; keys: string[]; value: string; onChange: (key: string) => void }) {
  return (
    <div className="flex items-center gap-2">
      <Label className="text-muted-foreground">{label}</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger size="sm">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {keys.map((k) => (
            <SelectItem key={k} value={k}>
              {SCORES[k].label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

function GroupedKeySelect({ keys, value, onChange }: { keys: string[]; value: string; onChange: (key: string) => void }) {
  return (
    <div className="flex items-center gap-2">
      <Label className="text-muted-foreground">지표</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger size="sm" className="min-w-32">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {byGroup(keys).map(([title, ks]) => (
            <SelectGroup key={title}>
              <SelectLabel>{title}</SelectLabel>
              {ks.map((k) => (
                <SelectItem key={k} value={k}>
                  {SCORES[k].label}
                </SelectItem>
              ))}
            </SelectGroup>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

function DatasetInfo({ runs }: { runs: RunOverview[] }) {
  const first = runs[0]
  const shas = new Set(runs.map((r) => r.dataset_sha).filter(Boolean))
  const items = new Set(runs.map((r) => r.counts.items).filter((n) => n != null))
  const facts: [string, ReactNode][] = [
    ["데이터셋", <code key="name">{first.dataset_name || "?"}</code>],
    ["번역 대상", first.target || "?"],
    ["항목 수", items.size ? [...items].join(" / ") : "—"],
  ]
  if (first.limit) facts.push(["범위", `${first.pick === "longest" ? "가장 긴" : "처음"} ${first.limit}개만`])
  if (shas.size === 1) facts.push(["manifest", <code key="sha">{[...shas][0].slice(0, 12)}</code>])
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm">
          <InfoIcon />
          데이터셋 정보
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="flex w-96 flex-col gap-3 text-sm">
        <MetaLine kind="dataset" meta={first.meta.dataset} />
        <dl className="grid grid-cols-[6rem_1fr] gap-x-3 gap-y-1">
          {facts.map(([term, value]) => (
            <div key={term} className="contents">
              <dt className="text-muted-foreground">{term}</dt>
              <dd className="min-w-0 break-all">{value}</dd>
            </div>
          ))}
        </dl>
      </PopoverContent>
    </Popover>
  )
}

function DatasetWarnings({ runs }: { runs: RunOverview[] }) {
  const hashes = new Set(runs.map((r) => r.meta.dataset?.hash).filter(Boolean))
  const shas = new Set(runs.map((r) => r.dataset_sha).filter(Boolean))
  const warnings = [
    hashes.size > 1 && "실행마다 데이터셋 설정이 달라요. 같은 조건의 비교가 아닐 수 있어요.",
    shas.size > 1 && "실행마다 manifest가 달라요. 같은 항목으로 채점되지 않았을 수 있어요.",
  ].filter(Boolean)
  if (!warnings.length) return null
  return (
    <div role="alert" className="flex items-start gap-2 rounded-lg border border-status-failed/30 bg-status-failed-soft px-4 py-3 text-sm text-status-failed">
      <TriangleAlertIcon className="mt-0.5 size-4 shrink-0" />
      <div className="flex flex-col gap-0.5">
        {warnings.map((text) => (
          <p key={String(text)}>{text}</p>
        ))}
      </div>
    </div>
  )
}

export function ComparePage() {
  const overview = useOverview()
  const [params, setParams] = useSearchParams()
  const [points, setPoints] = useState(true)
  const all = useMemo(() => overview.data?.runs ?? [], [overview.data])

  const dataset = all.some((r) => r.dataset_key === params.get("dataset")) ? params.get("dataset")! : (all[0]?.dataset_key ?? null)
  const runs = useMemo(
    () => all.filter((r) => r.dataset_key === dataset).sort((a, b) => a.pipeline.localeCompare(b.pipeline)),
    [all, dataset],
  )
  const colors = useMemo(() => new Map(runs.map((r, i) => [r.name, LANE_COLORS[i % LANE_COLORS.length]])), [runs])
  const hidden = useMemo(() => new Set((params.get("hide") ?? "").split(",").filter(Boolean)), [params])
  const shown = runs.filter((r) => !hidden.has(r.name))
  const dists = useDistributions(runs.map((r) => r.name))

  const update = (next: Record<string, string | null>) =>
    setParams(
      (prev) => {
        const merged = new URLSearchParams(prev)
        for (const [key, value] of Object.entries(next)) {
          if (value) merged.set(key, value)
          else merged.delete(key)
        }
        return merged
      },
      { replace: true },
    )

  const body = (() => {
    if (overview.isPending) return <Skeleton className="h-9 w-96" />
    if (overview.error) return <Notice>bench 서버에 연결하지 못했어요: {overview.error.message}</Notice>
    if (!all.length || !dataset)
      return (
        <Notice>
          bench/runs에 실행이 아직 없어요. Queue에서 실행을 추가하거나 <code>make bench CONFIG=… DATASET=…</code>로 돌려 주세요.
        </Notice>
      )

    const single = shown.length < 2
    const base = shown.find((r) => r.name === params.get("base")) ?? shown[0] ?? null

    const boardAll = byGroup(scoreKeys(shown)).flatMap(([, ks]) => ks)
    const sortParam = params.get("sort")
    const sort =
      sortParam && boardAll.includes(sortParam)
        ? { key: sortParam, desc: params.get("desc") === "1" }
        : { key: boardAll.find((k) => SCORES[k].better) ?? null, desc: false }
    const more = params.get("more") === "1"
    const boardKeys = more ? boardAll : boardAll.slice(0, MAX_COLS)
    if (sort.key && !boardKeys.includes(sort.key)) boardKeys.push(sort.key)
    const onSort = (key: string) =>
      update(key === sort.key ? { sort: key, desc: sort.desc ? null : "1" } : { sort: key, desc: null })
    const showDelta = !single && params.get("diff") !== "0"

    const distKeys = Object.keys(SCORES).filter((k) => SCORES[k].item && shown.some((r) => dists.byRun.get(r.name)?.[SCORES[k].item!]))
    const distKey = distKeys.includes(params.get("metric") ?? "")
      ? params.get("metric")!
      : (distKeys.find((k) => SCORES[k].better) ?? distKeys[0])

    const ranked = scoreKeys(shown).filter((k) => SCORES[k].better)
    const yKeys = ranked.filter((k) => ACCURACY_GROUPS.has(groupOf(k)))
    const xKeys = ranked.filter((k) => LATENCY_GROUPS.has(groupOf(k)))
    const yKey = yKeys.includes(params.get("y") ?? "") ? params.get("y")! : (["comet", "bleu", "wer"].find((k) => yKeys.includes(k)) ?? yKeys[0])
    const xKey = xKeys.includes(params.get("x") ?? "")
      ? params.get("x")!
      : (["laal_ms", "yaal_ms", "longyaal_ms", "avg_fsl_sec"].find((k) => xKeys.includes(k)) ?? xKeys[0])
    const missing = xKey && yKey ? missingRuns(shown, xKey, yKey) : []
    const chart = CHARTS.find((c) => c === params.get("chart")) ?? "distribution"

    const noneChosen = <Notice>선택한 실행이 없어요. 실행 메뉴에서 하나 이상 골라 주세요.</Notice>

    return (
      <>
        <DatasetWarnings runs={runs} />

        <Card>
          <CardHeader>
            <CardTitle>실행 지표</CardTitle>
            <CardDescription>열 제목을 누르면 정렬돼요. 굵은 값이 그 열에서 가장 좋아요.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
              <ModelMenu runs={runs} hidden={hidden} colors={colors} onChange={(next) => update({ hide: [...next].join(",") })} />
              {!single && (
                <>
                  <div className="flex items-center gap-2">
                    <Label className="text-muted-foreground">기준 실행</Label>
                    <Select value={base?.name ?? ""} onValueChange={(name) => update({ base: name })}>
                      <SelectTrigger size="sm" className="font-mono text-xs">
                        <SelectValue placeholder="—" />
                      </SelectTrigger>
                      <SelectContent>
                        {shown.map((r) => (
                          <SelectItem key={r.name} value={r.name} className="font-mono text-xs">
                            {r.pipeline}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <Label className="font-normal">
                    <Checkbox checked={showDelta} onCheckedChange={(on) => update({ diff: on === true ? null : "0" })} />
                    기준과의 차이
                  </Label>
                </>
              )}
              <span className="flex-1" />
              {boardAll.length > MAX_COLS && (
                <Button variant="outline" size="sm" onClick={() => update({ more: more ? null : "1" })}>
                  {more ? "지표 접기" : `지표 더 보기 (${boardAll.length - MAX_COLS})`}
                </Button>
              )}
            </div>
            {shown.length ? (
              <RunBoard runs={shown} keys={boardKeys} base={base} sort={sort} colors={colors} showDelta={showDelta} onSort={onSort} />
            ) : (
              noneChosen
            )}
          </CardContent>
        </Card>

        <Card className="pt-2">
          <Tabs value={chart} onValueChange={(next) => update({ chart: next === "distribution" ? null : next })}>
            <CardHeader className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b pb-2">
              <TabsList variant="line" className="h-11">
                <TabsTrigger value="distribution">항목별 분포</TabsTrigger>
                <TabsTrigger value="scatter">정확도 vs 지연</TabsTrigger>
              </TabsList>
              <span className="flex-1" />
              {chart === "distribution" && distKey && (
                <>
                  <GroupedKeySelect keys={distKeys} value={distKey} onChange={(k) => update({ metric: k })} />
                  <Label className="font-normal">
                    <Checkbox checked={points} onCheckedChange={(on) => setPoints(on === true)} />
                    항목 점 표시
                  </Label>
                  <Help label="분포 차트 읽는 법">
                    <p>평균 하나가 아니라 항목마다의 점수를 보여줘요. 중앙값이 더 좋은 실행이 왼쪽에 와요.</p>
                    <p className="text-muted-foreground">
                      상자는 가운데 절반(q1–q3), 굵은 선은 중앙값, 수염은 최소·최대, ◇는 평균이에요. 기둥 위 숫자는 중앙값이에요.
                    </p>
                    <p className="text-muted-foreground">
                      {SCORES[distKey].better ? BETTER_TEXT[SCORES[distKey].better!] : "방향 없음"}
                      {SCORES[distKey].note && ` · ${SCORES[distKey].note}`}
                    </p>
                  </Help>
                </>
              )}
              {chart === "scatter" && xKey && yKey && !single && (
                <>
                  <KeySelect label="정확도" keys={yKeys} value={yKey} onChange={(k) => update({ y: k })} />
                  <KeySelect label="지연" keys={xKeys} value={xKey} onChange={(k) => update({ x: k })} />
                  <Help label="산점도 읽는 법">
                    <p>실행 요약값으로 그려요. 두 축 모두 위쪽, 오른쪽이 좋아요.</p>
                    <p className="text-muted-foreground">점선은 파레토 경계예요. 두 축 모두에서 다른 실행에 지지 않는 실행들을 이어요.</p>
                  </Help>
                </>
              )}
            </CardHeader>
            <CardContent className="pt-4">
              <TabsContent value="distribution" className="flex flex-col gap-3">
                {!shown.length ? (
                  noneChosen
                ) : dists.pending ? (
                  <Notice>분포를 불러오는 중…</Notice>
                ) : !distKey ? (
                  <Notice>이 실행들에는 항목별 점수가 없어요.</Notice>
                ) : (
                  <DistributionChart runs={shown} scoreKey={distKey} dists={dists.byRun} colors={colors} points={points} />
                )}
              </TabsContent>
              <TabsContent value="scatter" className="flex flex-col gap-3">
                {single ? (
                  <Notice>실행이 2개 이상 있어야 비교할 수 있어요.</Notice>
                ) : !xKey || !yKey ? (
                  <Notice>이 실행들에는 그릴 정확도·지연 점수가 없어요.</Notice>
                ) : (
                  <>
                    <ScatterChart runs={shown} xKey={xKey} yKey={yKey} colors={colors} />
                    {missing.length > 0 && <Notice>빠진 실행: {missing.join(" · ")}</Notice>}
                  </>
                )}
              </TabsContent>
            </CardContent>
          </Tabs>
        </Card>
      </>
    )
  })()

  const latest = runs
    .map((r) => r.stamp)
    .filter(Boolean)
    .sort()
    .at(-1)

  return (
    <>
      <PageHeader title="Compare" />
      <PageBody>
        {dataset && (
          <div className="flex flex-wrap items-center gap-2">
            <DatasetPicker runs={all} value={dataset} onChange={(key) => update({ dataset: key, base: null, hide: null })} />
            <span className="text-sm text-muted-foreground tabular-nums">
              실행 {runs.length}개{latest && ` · 마지막 ${stampText(latest)}`}
            </span>
            {runs.length > 0 && <DatasetInfo runs={runs} />}
          </div>
        )}
        {body}
      </PageBody>
    </>
  )
}
