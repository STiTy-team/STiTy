import { useMutation, useQueryClient } from "@tanstack/react-query"
import { ArrowDownIcon, ArrowUpIcon } from "lucide-react"
import { Fragment, useMemo, useState, type ReactNode } from "react"
import { useNavigate, useSearchParams } from "react-router"

import { Notice, PageBody, PageHeader } from "@/components/page-header"
import { Details, ExpandButton, RunCell } from "@/components/queue/job-table"
import { StatusPill, type Tone } from "@/components/queue/status-pill"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { api, type RunRow } from "@/lib/api"
import { compareScores, formatScore, SCORES, unitOf } from "@/lib/compare/scores"
import { formatDateTime, formatDuration, shortSha, timeAgo } from "@/lib/format"
import { queryKeys, useRuns } from "@/lib/queries"
import { stampIso } from "@/lib/replay/format"
import { cn } from "@/lib/utils"

const ALL = "__all"
const LOCAL_HOST = "__local"
const PAGE = 50
const METRIC_COLUMNS = ["wer", "bleu", "comet", "laal_ms"]
const STATUSES = ["ok", "failed", "running"] as const
const NEWEST_FIRST = new Set(["stamp", "items", "wall_sec"])

const STATUS_LABEL: Record<string, string> = { ok: "완료", failed: "실패", running: "실행 중" }
const STATUS_TONE: Record<string, Tone> = { ok: "done", failed: "failed", running: "running" }

type SortKey = "stamp" | "pipeline" | "host" | "items" | "wall_sec" | string

const searchText = (run: RunRow) =>
  [run.dataset, run.pipeline, run.host, run.run_id, run.commit, run.status, ...run.pipeline_tags, ...run.dataset_tags]
    .filter(Boolean)
    .join(" ")
    .toLowerCase()

function sortValue(run: RunRow, key: SortKey): string | number | null {
  if (key === "stamp") return run.stamp
  if (key === "pipeline") return `${run.pipeline} ${run.dataset}`
  if (key === "host") return run.host ?? ""
  if (key === "items") return run.items
  if (key === "wall_sec") return run.wall_sec
  return run.metrics[key] ?? null
}

/** Runs with no value for the column always sink to the bottom, whichever way it is sorted. */
function compareRuns(a: RunRow, b: RunRow, key: SortKey, desc: boolean) {
  const va = sortValue(a, key)
  const vb = sortValue(b, key)
  if (va == null || vb == null) return va == null ? (vb == null ? 0 : 1) : -1
  const order =
    typeof va === "string" || typeof vb === "string"
      ? String(va).localeCompare(String(vb))
      : SCORES[key]
        ? compareScores(key, va, vb) || va - vb
        : va - vb
  return desc ? -order : order
}

function FilterSelect({ label, value, options, onChange }: { label: string; value: string; options: [string, string][]; onChange: (v: string) => void }) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger aria-label={label} className={cn("max-w-64 min-w-36", value !== ALL && "border-primary/50 bg-accent")}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={ALL}>{label}: 전체</SelectItem>
        {options.map(([id, text]) => (
          <SelectItem key={id} value={id} className="font-mono text-xs">
            {text}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function SortHead({ id, sort, onSort, className, children }: { id: SortKey; sort: { key: SortKey; desc: boolean }; onSort: (key: SortKey) => void; className?: string; children: ReactNode }) {
  const active = sort.key === id
  return (
    <TableHead className={className} aria-sort={active ? (sort.desc ? "descending" : "ascending") : undefined}>
      <button
        type="button"
        onClick={() => onSort(id)}
        className={cn("inline-flex items-center gap-1 hover:text-foreground", active ? "text-foreground" : "text-muted-foreground")}
      >
        {children}
        {active && (sort.desc ? <ArrowDownIcon className="size-3" /> : <ArrowUpIcon className="size-3" />)}
      </button>
    </TableHead>
  )
}

function OpenButton({ run }: { run: RunRow }) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const open = useMutation({
    mutationFn: async () => {
      if (run.local_name) return run.local_name
      const pulled = await api.pullRun({ dataset: run.dataset, pipeline: run.pipeline, run_id: run.run_id! })
      await queryClient.invalidateQueries({ queryKey: queryKeys.runs })
      return pulled.run
    },
    onSuccess: (name) => navigate(`/replay?${new URLSearchParams({ run: name })}`),
  })
  return (
    <div className="flex flex-col items-end gap-1">
      <Button
        variant="outline"
        size="sm"
        onClick={() => open.mutate()}
        disabled={open.isPending}
        title={run.local_name ? "이 컴퓨터에 있어요" : "S3에서 내려받은 뒤 열어요. 같은 파이프라인 × 데이터셋의 로컬 사본은 이걸로 바뀌어요."}
      >
        {open.isPending ? "내려받는 중…" : "리플레이"}
      </Button>
      {open.error && <span className="max-w-64 text-right text-xs whitespace-normal text-destructive">{open.error.message}</span>}
    </div>
  )
}

function RunDetails({ run }: { run: RunRow }) {
  const models = [run.models.transcription, run.models.correction, run.models.translation].filter(Boolean).join(" → ")
  const metrics = Object.entries(run.metrics).filter(([key]) => SCORES[key])
  return (
    <div className="flex flex-col gap-4 py-1">
      <Details
        facts={[
          ["run ID", <code key="id">{run.run_id ?? "이 컴퓨터에만 있음"}</code>],
          ["커밋", run.commit ? <code key="sha">{shortSha(run.commit)}</code> : "—"],
          ["시작", formatDateTime(run.started_at)],
          ["끝", formatDateTime(run.finished_at)],
          ["모델", models ? <code key="models">{models}</code> : "—"],
          ["실패한 항목", `${run.failed_items}개`],
          ["로컬 사본", run.local_name ? "있음" : "없음"],
        ]}
      />
      {(run.pipeline_description || run.dataset_description) && (
        <div className="grid gap-3 pl-10 sm:grid-cols-2">
          {[
            ["파이프라인", run.pipeline_description, run.pipeline_tags],
            ["데이터셋", run.dataset_description, run.dataset_tags],
          ].map(([label, text, tags]) => (
            <div key={label as string} className="flex min-w-0 flex-col gap-1">
              <span className="text-xs text-muted-foreground">{label as string}</span>
              {text && <p className="text-[13px]">{text as string}</p>}
              <div className="flex flex-wrap gap-1">
                {(tags as string[]).map((tag) => (
                  <Badge key={tag} variant="secondary">
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      {metrics.length > 0 && (
        <div className="flex flex-col gap-1.5 pl-10">
          <span className="text-xs text-muted-foreground">모든 지표</span>
          <div className="flex flex-wrap gap-x-5 gap-y-1.5 text-[13px] tabular-nums">
            {metrics.map(([key, value]) => (
              <span key={key}>
                <span className="text-muted-foreground">{SCORES[key].label}</span> {formatScore(key, value)}
              </span>
            ))}
          </div>
        </div>
      )}
      {run.failure && <p className="ml-10 rounded-md bg-status-failed-soft px-3 py-2 font-mono text-xs break-all text-status-failed">{run.failure}</p>}
    </div>
  )
}

export function RunsPage() {
  const runs = useRuns()
  const [params, setParams] = useSearchParams()
  const [open, setOpen] = useState<string | null>(null)
  const [limit, setLimit] = useState(PAGE)

  const update = (next: Record<string, string | null>) => {
    setLimit(PAGE)
    setParams(
      (prev) => {
        const merged = new URLSearchParams(prev)
        for (const [key, value] of Object.entries(next)) {
          if (value && value !== ALL) merged.set(key, value)
          else merged.delete(key)
        }
        return merged
      },
      { replace: true },
    )
  }

  const all = useMemo(() => runs.data?.runs ?? [], [runs.data])
  const query = params.get("q") ?? ""
  const dataset = params.get("dataset") ?? ALL
  const pipeline = params.get("pipeline") ?? ALL
  const host = params.get("host") ?? ALL
  const status = params.get("status") ?? ALL
  const sort = { key: params.get("sort") ?? "stamp", desc: params.get("asc") !== "1" }
  const onSort = (key: SortKey) => {
    const desc = sort.key === key ? !sort.desc : NEWEST_FIRST.has(key)
    update({ sort: key === "stamp" ? null : key, asc: desc ? null : "1" })
  }

  const texts = useMemo(() => new Map(all.map((run) => [run.key, searchText(run)])), [all])
  const q = query.trim().toLowerCase()
  const filtered = all
    .filter((run) => dataset === ALL || run.dataset === dataset)
    .filter((run) => pipeline === ALL || run.pipeline === pipeline)
    .filter((run) => host === ALL || (host === LOCAL_HOST ? run.host == null : run.host === host))
    .filter((run) => status === ALL || run.status === status)
    .filter((run) => !q || q.split(/\s+/).every((word) => texts.get(run.key)!.includes(word)))
    .sort((a, b) => compareRuns(a, b, sort.key, sort.desc))
  const shown = filtered.slice(0, limit)
  const metricColumns = METRIC_COLUMNS.filter((key) => all.some((run) => run.metrics[key] != null))

  const unique = (values: (string | null)[]) => [...new Set(values.filter((v): v is string => Boolean(v)))].sort()
  const statusCount = (s: string) => all.filter((run) => s === ALL || run.status === s).length
  const filtering = q || dataset !== ALL || pipeline !== ALL || host !== ALL || status !== ALL

  const body = (() => {
    if (runs.isPending) return <Skeleton className="h-9 w-96" />
    if (runs.error)
      return (
        <Notice>
          실행 목록을 불러오지 못했어요: {runs.error.message}
          {runs.error.message === "HTTP 404" && " API 서버가 이 페이지보다 오래된 버전이에요. make bench-manager 를 다시 띄워 주세요."}
        </Notice>
      )
    if (!all.length) return <Notice>실행이 아직 없어요. Queue에서 머신에 실행을 추가해 보세요.</Notice>
    return (
      <>
        {!runs.data.shared && <Notice>S3가 설정되지 않아서 이 컴퓨터에 있는 실행만 보여요.</Notice>}
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <Input
              type="search"
              value={query}
              onChange={(e) => update({ q: e.target.value })}
              placeholder="데이터셋, 파이프라인, 머신, 커밋, 태그로 찾기"
              aria-label="실행 찾기"
              className="max-w-sm min-w-60 flex-1"
            />
            <FilterSelect label="데이터셋" value={dataset} options={unique(all.map((r) => r.dataset)).map((d) => [d, d])} onChange={(v) => update({ dataset: v })} />
            <FilterSelect label="파이프라인" value={pipeline} options={unique(all.map((r) => r.pipeline)).map((p) => [p, p])} onChange={(v) => update({ pipeline: v })} />
            <FilterSelect
              label="머신"
              value={host}
              options={[...unique(all.map((r) => r.host)).map((h): [string, string] => [h, h]), ...(all.some((r) => r.host == null) ? [[LOCAL_HOST, "이 컴퓨터"] as [string, string]] : [])]}
              onChange={(v) => update({ host: v })}
            />
          </div>
          <div className="flex flex-wrap items-center gap-2" role="group" aria-label="상태로 거르기">
            {[ALL, ...STATUSES].map((s) => (
              <Button
                key={s}
                size="sm"
                variant={status === s ? "default" : "outline"}
                aria-pressed={status === s}
                className={cn("rounded-full", status === s && "bg-foreground text-background hover:bg-foreground/90")}
                onClick={() => update({ status: s })}
              >
                {s === ALL ? "전체" : STATUS_LABEL[s]}
                <span className="tabular-nums opacity-70">{statusCount(s)}</span>
              </Button>
            ))}
            <span className="flex-1" />
            <span className="text-sm text-muted-foreground tabular-nums">
              {filtering ? `${all.length}개 중 ${filtered.length}개` : `실행 ${all.length}개`}
            </span>
            {filtering && (
              <Button variant="ghost" size="sm" onClick={() => setParams({}, { replace: true })}>
                필터 지우기
              </Button>
            )}
          </div>
        </div>

        {!filtered.length ? (
          <Notice>조건에 맞는 실행이 없어요.</Notice>
        ) : (
          <div className="rounded-xl border">
            <Table className="min-w-[60rem] table-fixed">
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="w-12" />
                  <SortHead id="stamp" sort={sort} onSort={onSort} className="w-24">
                    시간
                  </SortHead>
                  <SortHead id="pipeline" sort={sort} onSort={onSort}>
                    파이프라인 · 데이터셋
                  </SortHead>
                  <SortHead id="host" sort={sort} onSort={onSort} className="w-24">
                    머신
                  </SortHead>
                  <TableHead className="w-20">상태</TableHead>
                  <SortHead id="items" sort={sort} onSort={onSort} className="w-16 text-right">
                    항목
                  </SortHead>
                  <SortHead id="wall_sec" sort={sort} onSort={onSort} className="w-24">
                    걸린 시간
                  </SortHead>
                  {metricColumns.map((key) => (
                    <SortHead key={key} id={key} sort={sort} onSort={onSort} className="w-20 text-right">
                      {SCORES[key].label}
                      {unitOf(key) && <span className="font-normal">({unitOf(key)})</span>}
                    </SortHead>
                  ))}
                  <TableHead className="w-28" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {shown.map((run) => {
                  const expanded = open === run.key
                  const iso = stampIso(run.stamp)
                  return (
                    <Fragment key={run.key}>
                      <TableRow className={cn(expanded && "border-b-0 bg-muted/40")}>
                        <TableCell className="pl-2">
                          <ExpandButton open={expanded} onToggle={() => setOpen(expanded ? null : run.key)} />
                        </TableCell>
                        <TableCell title={formatDateTime(iso)}>{timeAgo(iso)}</TableCell>
                        <TableCell title={`${run.pipeline}\n${run.dataset}`}>
                          <RunCell pipeline={run.pipeline} dataset={run.dataset} />
                        </TableCell>
                        <TableCell className={cn("truncate font-mono text-xs", !run.host && "font-sans text-muted-foreground")}>{run.host ?? "이 컴퓨터"}</TableCell>
                        <TableCell>
                          <StatusPill tone={STATUS_TONE[run.status] ?? "queued"}>{STATUS_LABEL[run.status] ?? (run.status || "—")}</StatusPill>
                        </TableCell>
                        <TableCell className="text-right tabular-nums">{run.items ?? "—"}</TableCell>
                        <TableCell className="tabular-nums">{formatDuration(run.wall_sec)}</TableCell>
                        {metricColumns.map((key) => (
                          <TableCell key={key} className="text-right tabular-nums">
                            {formatScore(key, run.metrics[key], { unit: false })}
                          </TableCell>
                        ))}
                        <TableCell className="pr-3 text-right">
                          <OpenButton run={run} />
                        </TableCell>
                      </TableRow>
                      {expanded && (
                        <TableRow className="bg-muted/40 hover:bg-muted/40">
                          <TableCell colSpan={8 + metricColumns.length} className="whitespace-normal">
                            <RunDetails run={run} />
                          </TableCell>
                        </TableRow>
                      )}
                    </Fragment>
                  )
                })}
              </TableBody>
            </Table>
          </div>
        )}
        {filtered.length > limit && (
          <Button variant="outline" size="sm" className="self-start" onClick={() => setLimit(limit + PAGE)}>
            {Math.min(PAGE, filtered.length - limit)}개 더 보기
          </Button>
        )}
      </>
    )
  })()

  return (
    <>
      <PageHeader title="Runs" description="모든 머신의 실행 기록" />
      <PageBody>{body}</PageBody>
    </>
  )
}
