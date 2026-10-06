import { useMutation } from "@tanstack/react-query"
import { Fragment, useState } from "react"
import { useNavigate } from "react-router"

import { FailureLogDialog } from "@/components/queue/failure-log-dialog"
import { Details, ExpandButton, RunCell } from "@/components/queue/job-table"
import { StatusPill, type Tone } from "@/components/queue/status-pill"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { api, type FinalOutcome, type Finished, type Machine, type RunLocation } from "@/lib/api"
import { failureReason, formatDateTime, formatDuration, secondsBetween, shortSha, timeAgo } from "@/lib/format"
import { cn } from "@/lib/utils"

const PAGE = 10

const OUTCOME_LABEL: Record<FinalOutcome, string> = {
  done: "완료",
  failed: "실패",
  cancelled: "취소됨",
}

const OUTCOME_TONE: Record<FinalOutcome, Tone> = {
  done: "done",
  failed: "failed",
  cancelled: "queued",
}

type Filter = FinalOutcome | "all"
const FILTERS: Filter[] = ["all", "done", "failed", "cancelled"]

function OpenReplayButton({ run }: { run: RunLocation }) {
  const navigate = useNavigate()
  const pull = useMutation({
    mutationFn: () => api.pullRun(run),
    onSuccess: ({ run: name }) => navigate(`/replay?${new URLSearchParams({ run: name })}`),
  })
  return (
    <div className="flex flex-col items-end gap-1">
      <Button variant="outline" size="sm" onClick={() => pull.mutate()} disabled={pull.isPending}>
        {pull.isPending ? "여는 중…" : "리플레이 열기"}
      </Button>
      {pull.error && <span className="text-xs text-destructive">{pull.error.message}</span>}
    </div>
  )
}

function tookSec(finished: Finished) {
  const last = finished.job.attempts.at(-1)
  return last?.ran_sec ?? secondsBetween(last?.started_at ?? null, finished.ended_at)
}

export function HistoryTable({ machine }: { machine: Machine }) {
  const [filter, setFilter] = useState<Filter>("all")
  const [limit, setLimit] = useState(PAGE)
  const [open, setOpen] = useState<string | null>(null)

  if (!machine.history.length) {
    return <p className="rounded-xl border border-dashed px-4 py-5 text-sm text-muted-foreground">끝난 작업이 아직 없어요.</p>
  }
  const count = (f: Filter) => (f === "all" ? machine.history.length : machine.history.filter((h) => h.outcome === f).length)
  const matching = machine.history.filter((h) => filter === "all" || h.outcome === filter)
  const rows = matching.slice(0, limit)

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-2" role="group" aria-label="결과로 거르기">
        {FILTERS.map((f) => (
          <Button
            key={f}
            size="sm"
            variant={filter === f ? "default" : "outline"}
            aria-pressed={filter === f}
            className={cn("rounded-full", filter === f && "bg-foreground text-background hover:bg-foreground/90")}
            onClick={() => {
              setFilter(f)
              setLimit(PAGE)
            }}
          >
            {f === "all" ? "전체" : OUTCOME_LABEL[f]}
            <span className="tabular-nums opacity-70">{count(f)}</span>
          </Button>
        ))}
      </div>
      {!rows.length ? (
        <p className="text-sm text-muted-foreground">이 결과의 작업이 없어요.</p>
      ) : (
        <div className="rounded-xl border">
          <Table className="min-w-[44rem] table-fixed">
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="w-12" />
                <TableHead className="w-24">결과</TableHead>
                <TableHead>실행</TableHead>
                <TableHead className="w-28">끝남</TableHead>
                <TableHead className="w-28">걸린 시간</TableHead>
                <TableHead className="w-36" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((finished) => {
                const { job } = finished
                const last = job.attempts.at(-1)
                const expanded = open === job.id
                return (
                  <Fragment key={job.id}>
                    <TableRow className={cn(expanded && "border-b-0 bg-muted/40")}>
                      <TableCell className="pl-2">
                        <ExpandButton open={expanded} onToggle={() => setOpen(expanded ? null : job.id)} />
                      </TableCell>
                      <TableCell>
                        <StatusPill tone={OUTCOME_TONE[finished.outcome]}>{OUTCOME_LABEL[finished.outcome]}</StatusPill>
                      </TableCell>
                      <TableCell>
                        <RunCell pipeline={job.pipeline} dataset={job.dataset}>
                          {finished.error && !expanded && (
                            <span className="truncate text-xs text-status-failed" title={failureReason(finished.error)}>
                              {failureReason(finished.error)}
                            </span>
                          )}
                        </RunCell>
                      </TableCell>
                      <TableCell title={formatDateTime(finished.ended_at)}>{timeAgo(finished.ended_at)}</TableCell>
                      <TableCell>{formatDuration(tookSec(finished))}</TableCell>
                      <TableCell className="pr-3 text-right">
                        {finished.run ? (
                          <OpenReplayButton run={finished.run} />
                        ) : finished.error ? (
                          <FailureLogDialog finished={finished} host={machine.host} />
                        ) : null}
                      </TableCell>
                    </TableRow>
                    {expanded && (
                      <TableRow className="bg-muted/40 hover:bg-muted/40">
                        <TableCell colSpan={6} className="whitespace-normal">
                          <Details
                            facts={[
                              ["브랜치", <code key="branch">{job.branch}</code>],
                              ["커밋", last?.commit ? <code key="sha">{shortSha(last.commit)}</code> : "—"],
                              ["제출", formatDateTime(job.submitted_at)],
                              ["시작", formatDateTime(last?.started_at ?? null)],
                              ["끝남", formatDateTime(finished.ended_at)],
                              ["작업 ID", <code key="id">{job.id}</code>],
                              ...(finished.error ? [["실패 원인", failureReason(finished.error)] as [string, string]] : []),
                            ]}
                          />
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
      {matching.length > limit && (
        <Button variant="outline" size="sm" className="self-start" onClick={() => setLimit(limit + PAGE)}>
          {Math.min(PAGE, matching.length - limit)}개 더 보기
        </Button>
      )}
    </div>
  )
}
