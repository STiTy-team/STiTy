import { ChevronRightIcon } from "lucide-react"
import { Fragment, useState, type ReactNode } from "react"

import { CancelJobButton } from "@/components/queue/cancel-job-button"
import { StatusPill, type Tone } from "@/components/queue/status-pill"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import type { Job, JobState, Machine } from "@/lib/api"
import { formatDateTime, formatDuration, secondsBetween, shortSha } from "@/lib/format"
import { cn } from "@/lib/utils"

const STATE_LABEL: Record<JobState, string> = {
  queued: "대기 중",
  running: "실행 중",
  stopping: "멈추는 중",
  cancelling: "취소 중",
}

const STATE_TONE: Record<JobState, Tone> = {
  queued: "queued",
  running: "running",
  stopping: "failed",
  cancelling: "failed",
}

const startedAt = (job: Job) => (job.state === "queued" ? null : (job.attempts.at(-1)?.started_at ?? null))

function timeText(job: Job) {
  const started = startedAt(job)
  if (started) return { text: `실행 ${formatDuration(secondsBetween(started))}`, exact: `${formatDateTime(started)} 시작` }
  return { text: `대기 ${formatDuration(secondsBetween(job.submitted_at))}`, exact: `${formatDateTime(job.submitted_at)} 제출` }
}

export function RunCell({ pipeline, dataset, children }: { pipeline: string; dataset: string; children?: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col">
      <span className="truncate font-mono text-[13px] font-medium">{pipeline}</span>
      <span className="truncate font-mono text-xs text-muted-foreground">{dataset}</span>
      {children}
    </div>
  )
}

export function ExpandButton({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  return (
    <Button variant="ghost" size="icon-sm" aria-expanded={open} aria-label={open ? "자세히 닫기" : "자세히 보기"} onClick={onToggle}>
      <ChevronRightIcon className={cn("transition-transform", open && "rotate-90")} />
    </Button>
  )
}

export function Details({ facts }: { facts: [string, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] gap-x-6 gap-y-3 py-1 pl-10">
      {facts.map(([term, value]) => (
        <div key={term} className="flex min-w-0 flex-col gap-0.5">
          <dt className="text-xs text-muted-foreground">{term}</dt>
          <dd className="text-[13px] break-all">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

export function JobTable({ machine }: { machine: Machine }) {
  const [open, setOpen] = useState<string | null>(null)
  if (!machine.jobs.length) {
    return <p className="rounded-xl border border-dashed px-4 py-5 text-sm text-muted-foreground">대기 중인 작업이 없어요.</p>
  }
  return (
    <div className="rounded-xl border">
      <Table className="min-w-[44rem] table-fixed">
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className="w-12" />
            <TableHead className="w-28">상태</TableHead>
            <TableHead>실행</TableHead>
            <TableHead className="w-48">브랜치</TableHead>
            <TableHead className="w-36">시간</TableHead>
            <TableHead className="w-28" />
          </TableRow>
        </TableHeader>
        <TableBody>
          {machine.jobs.map((job) => {
            const expanded = open === job.id
            const time = timeText(job)
            const commit = job.attempts.at(-1)?.commit ?? null
            return (
              <Fragment key={job.id}>
                <TableRow className={cn(expanded && "border-b-0 bg-muted/40")}>
                  <TableCell className="pl-2">
                    <ExpandButton open={expanded} onToggle={() => setOpen(expanded ? null : job.id)} />
                  </TableCell>
                  <TableCell>
                    <StatusPill tone={STATE_TONE[job.state]}>{STATE_LABEL[job.state]}</StatusPill>
                  </TableCell>
                  <TableCell>
                    <RunCell pipeline={job.pipeline} dataset={job.dataset} />
                  </TableCell>
                  <TableCell className="truncate font-mono text-xs">{job.branch}</TableCell>
                  <TableCell title={time.exact}>{time.text}</TableCell>
                  <TableCell className="pr-3 text-right">
                    <CancelJobButton host={machine.host} job={job} />
                  </TableCell>
                </TableRow>
                {expanded && (
                  <TableRow className="bg-muted/40 hover:bg-muted/40">
                    <TableCell colSpan={6} className="whitespace-normal">
                      <Details
                        facts={[
                          ["작업 ID", <code key="id">{job.id}</code>],
                          ["커밋", commit ? <code key="sha">{shortSha(commit)}</code> : "아직 시작 전"],
                          ["제출", formatDateTime(job.submitted_at)],
                          ["시작", formatDateTime(startedAt(job))],
                          ["시도 횟수", `${job.attempts.length}회`],
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
  )
}
