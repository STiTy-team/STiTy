import FullCalendar from "@fullcalendar/react"
import interactionPlugin from "@fullcalendar/react/interaction"
import "@fullcalendar/react/skeleton.css"
import monarchTheme from "@fullcalendar/react/themes/monarch"
import "@fullcalendar/react/themes/monarch/theme.css"
import timeGridPlugin from "@fullcalendar/react/timegrid"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { useState } from "react"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { api, type Machine, type WeeklyBlock } from "@/lib/api"
import { queryKeys } from "@/lib/queries"
import {
  REFERENCE_WEEK_START,
  blockRange,
  normalizeBlocks,
  rangeToBlocks,
  sameBlocks,
} from "@/lib/timetable"

import "./timetable-palette.css"

function TimetableEditor({ machine, onSaved }: { machine: Machine; onSaved: () => void }) {
  const queryClient = useQueryClient()
  const [draft, setDraft] = useState<WeeklyBlock[] | null>(null)
  const [draftEtag, setDraftEtag] = useState<string | null>(null)
  const saved = machine.settings.blocks
  const blocks = draft ?? saved
  const dirty = draft !== null && !sameBlocks(draft, saved)

  const save = useMutation({
    mutationFn: () =>
      api.saveSettings(machine.host, { ...machine.settings, blocks: normalizeBlocks(blocks) }, draftEtag),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: queryKeys.machines })
      setDraft(null)
      onSaved()
    },
    onError: () => queryClient.invalidateQueries({ queryKey: queryKeys.machines }),
  })

  function change(next: WeeklyBlock[]) {
    if (draft === null) setDraftEtag(machine.settings_etag)
    save.reset()
    setDraft(normalizeBlocks(next))
  }

  function revert() {
    save.reset()
    setDraft(null)
  }

  const events = blocks.map((block, index) => ({
    id: String(index),
    title: `${block.start}–${block.end}`,
    ...blockRange(block),
  }))
  const withoutEvent = (id: string) => blocks.filter((_, index) => String(index) !== id)

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-muted-foreground">
          worker는 이 시간 안에서만 작업을 시작해요 ({machine.settings.timezone}). 빈 칸을 끌어 시간을 더하고, 블록을
          끌어 옮기거나 가장자리로 길이를 바꾸고, 블록을 누르면 지워요.
        </p>
        <div className="flex items-center gap-2">
          {dirty && <span className="text-sm text-muted-foreground">저장 안 한 변경</span>}
          <Button variant="outline" size="sm" onClick={revert} disabled={!dirty || save.isPending}>
            되돌리기
          </Button>
          <Button size="sm" onClick={() => save.mutate()} disabled={!dirty || save.isPending}>
            {save.isPending ? "저장 중…" : "시간표 저장"}
          </Button>
        </div>
      </div>
      {save.error && (
        <p className="text-sm text-destructive">
          {save.error.message} · 되돌리기로 저장된 시간표를 불러오거나, 다시 저장해서 덮어쓸 수 있어요.
        </p>
      )}
      {!blocks.length && (
        <p className="text-sm text-destructive">시간표가 비어 있어서 이 머신은 작업을 시작하지 않아요.</p>
      )}
      <FullCalendar
        plugins={[timeGridPlugin, interactionPlugin, monarchTheme]}
        initialView="timeGridWeek"
        initialDate={REFERENCE_WEEK_START}
        timeZone="UTC"
        firstDay={1}
        headerToolbar={false}
        dayHeaderFormat={{ weekday: "short" }}
        allDaySlot={false}
        slotDuration="01:00:00"
        snapDuration="00:30:00"
        slotHeaderFormat={{ hour: "2-digit", minute: "2-digit", hour12: false }}
        displayEventTime={false}
        height="auto"
        selectable
        editable
        eventResizableFromStart
        events={events}
        eventClass="timetable-block"
        select={(info) => change([...blocks, ...rangeToBlocks(info.start, info.end)])}
        eventChange={(info) => {
          if (!info.event.start || !info.event.end) return
          change([...withoutEvent(info.event.id), ...rangeToBlocks(info.event.start, info.event.end)])
        }}
        eventClick={(info) => change(withoutEvent(info.event.id))}
      />
    </div>
  )
}

export function EditTimetableDialog({ machine, open, onOpenChange }: { machine: Machine; open: boolean; onOpenChange: (open: boolean) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[92vh] overflow-y-auto sm:max-w-6xl">
        <DialogHeader>
          <DialogTitle>
            작업 시간표 · <span className="font-mono">{machine.host}</span>
          </DialogTitle>
          <DialogDescription>이 머신의 worker가 매주 작업을 시작해도 되는 시간이에요.</DialogDescription>
        </DialogHeader>
        {open && <TimetableEditor machine={machine} onSaved={() => onOpenChange(false)} />}
      </DialogContent>
    </Dialog>
  )
}
