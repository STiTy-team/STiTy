import { useState } from "react"

import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Textarea } from "@/components/ui/textarea"
import { ApiError, type Machine } from "@/lib/api"
import { useSaveNotes } from "@/lib/queries"

function NotesForm({ machine, onDone }: { machine: Machine; onDone: () => void }) {
  const [notes, setNotes] = useState(machine.notes)
  const [etag] = useState(machine.notes_etag)
  const save = useSaveNotes(machine.host)

  const error =
    save.error instanceof ApiError && save.error.status === 409
      ? "수정하는 사이에 다른 사람이 설명을 바꿨어요. 닫았다가 다시 열면 바뀐 내용이 보여요."
      : save.error?.message

  return (
    <>
      <Textarea
        aria-label="설명"
        autoFocus
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="어떤 머신인지, 무엇에 쓰면 좋은지, 대기열에 넣기 전에 알아야 할 것"
        className="h-48 resize-y"
      />
      {error && <p className="text-sm text-destructive">{error}</p>}
      <DialogFooter>
        <Button variant="outline" onClick={onDone}>
          취소
        </Button>
        <Button
          onClick={() => save.mutate({ notes, etag }, { onSuccess: onDone })}
          disabled={save.isPending || notes === machine.notes}
        >
          {save.isPending ? "저장 중…" : "저장"}
        </Button>
      </DialogFooter>
    </>
  )
}

export function EditNotesDialog({ machine, open, onOpenChange }: { machine: Machine; open: boolean; onOpenChange: (open: boolean) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>
            머신 설명 · <span className="font-mono">{machine.host}</span>
          </DialogTitle>
          <DialogDescription>이 머신의 Queue 탭에서 모두에게 보여요.</DialogDescription>
        </DialogHeader>
        {open && <NotesForm machine={machine} onDone={() => onOpenChange(false)} />}
      </DialogContent>
    </Dialog>
  )
}
