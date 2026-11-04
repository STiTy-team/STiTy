import { InfoIcon } from "lucide-react"
import { Link } from "react-router"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { modelLine } from "@/lib/compare/runs"
import { BETTER_TEXT, compareScores, formatScore, metric, SCORES, unitOf } from "@/lib/compare/scores"
import type { RunOverview } from "@/lib/compare/types"
import { byGroup } from "@/lib/replay/format"
import { cn } from "@/lib/utils"

export type Sort = { key: string | null; desc: boolean }

function Delta({ run, base, scoreKey }: { run: RunOverview; base: RunOverview | null; scoreKey: string }) {
  if (run === base) return <span className="block text-xs text-muted-foreground">기준</span>
  const v = metric(run.summary_metrics, scoreKey)
  const b = base && metric(base.summary_metrics, scoreKey)
  if (v == null || b == null) return <span className="block text-xs text-muted-foreground">—</span>
  const shown = formatScore(scoreKey, Math.abs(v - b), { unit: false })
  if (!/[1-9]/.test(shown)) return <span className="block text-xs text-muted-foreground">0</span>
  const c = compareScores(scoreKey, v, b)
  return (
    <span className={cn("block text-xs font-medium", c < 0 ? "text-emerald-600" : c > 0 ? "text-destructive" : "text-muted-foreground")}>
      {v > b ? "+" : "−"}
      {shown}
    </span>
  )
}

function RunInfo({ run }: { run: RunOverview }) {
  const pipeline = run.meta.pipeline
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon-sm" className="size-7 text-muted-foreground" aria-label={`${run.pipeline} 정보`}>
          <InfoIcon />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="flex w-80 flex-col gap-2 text-sm">
        {pipeline?.description && <p>{pipeline.description}</p>}
        <p className="font-mono text-xs text-muted-foreground">{modelLine(run) || "모델 정보 없음"}</p>
        {pipeline && pipeline.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {pipeline.tags.map((tag) => (
              <Badge key={tag} variant="secondary">
                {tag}
              </Badge>
            ))}
          </div>
        )}
        <p className="font-mono text-xs text-muted-foreground">config hash {pipeline?.hash || "기록 없음"}</p>
      </PopoverContent>
    </Popover>
  )
}

const arrow = (key: string) => (SCORES[key].better === "lower" ? " ↓" : SCORES[key].better === "higher" ? " ↑" : "")

export function RunBoard({
  runs,
  keys,
  base,
  sort,
  colors,
  showDelta,
  onSort,
}: {
  runs: RunOverview[]
  keys: string[]
  base: RunOverview | null
  sort: Sort
  colors: Map<string, string>
  showDelta: boolean
  onSort: (key: string) => void
}) {
  const groups = byGroup(keys)
  const ordered = groups.flatMap(([, ks]) => ks)

  const sorted = [...runs].sort((a, b) => {
    if (!sort.key) return 0
    const va = metric(a.summary_metrics, sort.key)
    const vb = metric(b.summary_metrics, sort.key)
    if (va == null) return 1
    if (vb == null) return -1
    const c = compareScores(sort.key, va, vb) || va - vb
    return sort.desc ? -c : c
  })

  const best: Record<string, number> = {}
  for (const k of ordered) {
    const values = runs.map((r) => metric(r.summary_metrics, k)).filter((v): v is number => v != null)
    if (SCORES[k].better && values.length > 1) best[k] = values.reduce((a, b) => (compareScores(k, a, b) <= 0 ? a : b))
  }

  return (
    <Table>
      <TableHeader>
        <TableRow className="border-b-0 hover:bg-transparent">
          <TableHead colSpan={2} />
          {groups.map(([title, ks]) => (
            <TableHead key={title} colSpan={ks.length} className="h-6 pl-6 text-xs font-normal text-muted-foreground">
              {title}
            </TableHead>
          ))}
        </TableRow>
        <TableRow>
          <TableHead className="min-w-64">실행</TableHead>
          <TableHead className="text-right">항목</TableHead>
          {groups.map(([, ks]) =>
            ks.map((k, i) => (
              <TableHead
                key={k}
                onClick={() => onSort(k)}
                title={[SCORES[k].note, BETTER_TEXT[SCORES[k].better ?? ""]].filter(Boolean).join(" · ")}
                className={cn(
                  "cursor-pointer text-right select-none hover:text-foreground",
                  i === 0 && "pl-6",
                  k === sort.key ? "text-foreground" : "text-muted-foreground",
                )}
              >
                {SCORES[k].label}
                {unitOf(k) && <span className="font-normal"> ({unitOf(k)})</span>}
                {arrow(k)}
                {k === sort.key && (sort.desc ? " ▴" : " ▾")}
              </TableHead>
            )),
          )}
        </TableRow>
      </TableHeader>
      <TableBody>
        {sorted.map((run) => (
          <TableRow key={run.name} className={cn(run === base && "bg-muted/50")}>
            <TableCell className="align-top whitespace-normal">
              <div className="flex flex-wrap items-center gap-2">
                <span className="size-2 shrink-0 rounded-full" style={{ background: colors.get(run.name) }} />
                <Link to={`/replay?run=${encodeURIComponent(run.name)}`} className="font-mono text-[13px] font-semibold hover:underline">
                  {run.pipeline}
                </Link>
                {run.status && run.status !== "ok" && (
                  <Badge variant={run.status === "running" ? "outline" : "destructive"}>{run.status}</Badge>
                )}
                <RunInfo run={run} />
              </div>
            </TableCell>
            <TableCell className="text-right align-top font-medium tabular-nums">
              {run.counts.items ?? "—"}
              {run.counts.of ? ` / ${run.counts.of}` : ""}
            </TableCell>
            {groups.map(([, ks]) =>
              ks.map((k, i) => {
                const v = metric(run.summary_metrics, k)
                const isBest = v != null && best[k] === v
                return (
                  <TableCell
                    key={k}
                    title={isBest ? "이 열에서 가장 좋은 값" : undefined}
                    className={cn("text-right align-top tabular-nums", i === 0 && "pl-6", isBest ? "font-bold" : "font-medium")}
                  >
                    {formatScore(k, v, { unit: false })}
                    {showDelta && <Delta run={run} base={base} scoreKey={k} />}
                  </TableCell>
                )
              }),
            )}
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}
