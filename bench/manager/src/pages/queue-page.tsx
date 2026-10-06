import { CalendarClockIcon, EllipsisIcon, PencilIcon, TriangleAlertIcon } from "lucide-react"
import { useState } from "react"
import { useNavigate, useParams } from "react-router"

import { Notice, PageBody, PageHeader } from "@/components/page-header"
import { AddRunDialog } from "@/components/queue/add-run-dialog"
import { EditNotesDialog } from "@/components/queue/edit-notes-dialog"
import { HistoryTable } from "@/components/queue/history-table"
import { JobTable } from "@/components/queue/job-table"
import { GpuBars, MachineDot } from "@/components/queue/machine-gpu"
import { StatusPill } from "@/components/queue/status-pill"
import { EditTimetableDialog } from "@/components/queue/timetable-dialog"
import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type { Machine } from "@/lib/api"
import { useMachines } from "@/lib/queries"
import { cn } from "@/lib/utils"

const SHORT_NOTES = 110

function MachineMenu({ machine }: { machine: Machine }) {
  const [menu, setMenu] = useState(false)
  const [dialog, setDialog] = useState<"notes" | "timetable" | null>(null)
  const pick = (next: "notes" | "timetable") => {
    setMenu(false)
    setDialog(next)
  }
  return (
    <>
      <Popover open={menu} onOpenChange={setMenu}>
        <PopoverTrigger asChild>
          <Button variant="outline" size="icon" aria-label="머신 설정">
            <EllipsisIcon />
          </Button>
        </PopoverTrigger>
        <PopoverContent align="end" className="flex w-52 flex-col gap-0.5 p-1">
          <Button variant="ghost" className="justify-start" onClick={() => pick("notes")}>
            <PencilIcon />
            설명 수정
          </Button>
          <Button variant="ghost" className="justify-start" onClick={() => pick("timetable")}>
            <CalendarClockIcon />
            작업 시간표 수정
          </Button>
        </PopoverContent>
      </Popover>
      <EditNotesDialog machine={machine} open={dialog === "notes"} onOpenChange={(open) => setDialog(open ? "notes" : null)} />
      <EditTimetableDialog machine={machine} open={dialog === "timetable"} onOpenChange={(open) => setDialog(open ? "timetable" : null)} />
    </>
  )
}

function Notes({ text }: { text: string }) {
  const [more, setMore] = useState(false)
  const long = text.length > SHORT_NOTES || text.includes("\n")
  return (
    <div className="flex max-w-3xl items-start gap-2 text-sm text-foreground/80">
      <p className={cn("min-w-0 whitespace-pre-wrap", long && !more && "line-clamp-1")}>{text}</p>
      {long && (
        <button type="button" className="shrink-0 font-medium text-primary hover:underline" onClick={() => setMore(!more)}>
          {more ? "접기" : "더 보기"}
        </button>
      )}
    </div>
  )
}

function MachineTab({ machine }: { machine: Machine }) {
  const running = machine.jobs.filter((job) => job.state !== "queued").length
  const waiting = machine.jobs.length - running
  const notes = machine.notes.trim()
  const noTimetable = !machine.settings.blocks.length
  return (
    <div className="flex flex-col gap-8 pt-4">
      <section className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
        <div className="flex min-w-0 flex-1 flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="font-mono text-xl font-medium">{machine.host}</h2>
            <GpuBars machine={machine} />
            {machine.connected ? (
              <StatusPill tone="done" dot>
                연결됨
              </StatusPill>
            ) : (
              <StatusPill tone="failed" dot>
                연결 끊김 · 5분 넘게 worker 응답이 없어요
              </StatusPill>
            )}
            {machine.settings.paused && <StatusPill tone="queued">일시정지</StatusPill>}
            {machine.jobs.length > 0 && (
              <span className="text-sm text-muted-foreground tabular-nums">
                {[running && `실행 중 ${running}개`, waiting && `대기 ${waiting}개`].filter(Boolean).join(" · ")}
              </span>
            )}
          </div>
          {notes ? <Notes text={notes} /> : <p className="text-sm text-muted-foreground">설명이 아직 없어요. ⋯ 메뉴에서 추가할 수 있어요.</p>}
        </div>
        <div className="flex shrink-0 gap-2">
          <AddRunDialog host={machine.host} />
          <MachineMenu machine={machine} />
        </div>
      </section>

      {noTimetable && (
        <div role="alert" className="flex items-center gap-2 rounded-lg border border-status-failed/30 bg-status-failed-soft px-4 py-3 text-sm text-status-failed">
          <TriangleAlertIcon className="size-4 shrink-0" />
          작업 시간표가 비어 있어서 이 머신은 작업을 시작하지 않아요. ⋯ 메뉴에서 시간표를 정해 주세요.
        </div>
      )}

      <section className="flex flex-col gap-3">
        <div className="flex items-baseline gap-2">
          <h3 className="text-base font-semibold">대기열</h3>
          {machine.jobs.length > 1 && <span className="text-sm text-muted-foreground">위에서부터 하나씩 실행돼요</span>}
        </div>
        <JobTable machine={machine} />
      </section>

      <section className="flex flex-col gap-3">
        <h3 className="text-base font-semibold">기록</h3>
        <HistoryTable machine={machine} />
      </section>
    </div>
  )
}

export function QueuePage() {
  const { host } = useParams()
  const navigate = useNavigate()
  const machines = useMachines()

  const body = (() => {
    if (machines.isPending) return <Skeleton className="h-9 w-80" />
    if (machines.error) return <Notice>bench 서버에 연결하지 못했어요: {machines.error.message}</Notice>
    if (!machines.data.configured) return <Notice>{machines.data.message}</Notice>

    const list = machines.data.machines
    if (!list.length) return <Notice>머신이 아직 없어요. scripts/bench/worker/start.sh로 worker를 띄우면 여기에 나타나요.</Notice>
    const active = list.find((machine) => machine.host === host) ?? list.find((machine) => machine.connected) ?? list[0]

    return (
      <Tabs value={active.host} onValueChange={(next) => navigate(`/queue/${encodeURIComponent(next)}`)}>
        <TabsList className="h-auto flex-wrap">
          {list.map((machine) => (
            <TabsTrigger key={machine.host} value={machine.host} className="h-8 px-3 font-mono">
              <MachineDot machine={machine} />
              {machine.host}
              {machine.jobs.length > 0 && <span className="font-sans text-xs text-muted-foreground tabular-nums">{machine.jobs.length}</span>}
            </TabsTrigger>
          ))}
        </TabsList>
        {list.map((machine) => (
          <TabsContent key={machine.host} value={machine.host}>
            <MachineTab machine={machine} />
          </TabsContent>
        ))}
      </Tabs>
    )
  })()

  return (
    <>
      <PageHeader title="Queue" description="머신마다 기다리거나 실행 중인 작업" />
      <PageBody className="max-w-[1180px]">{body}</PageBody>
    </>
  )
}
