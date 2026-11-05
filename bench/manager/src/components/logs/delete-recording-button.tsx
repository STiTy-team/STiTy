import { Trash2Icon } from "lucide-react"

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { useDeleteRecording } from "@/lib/logs/queries"
import type { Recording } from "@/lib/logs/types"

export function DeleteRecordingButton({ recording, onDeleted }: { recording: Recording; onDeleted?: () => void }) {
  const remove = useDeleteRecording()
  const live = recording.state === "live"
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon-sm"
          className="text-muted-foreground hover:text-destructive"
          disabled={live || remove.isPending}
          aria-label="녹음 삭제"
          title={live ? "아직 녹음 중이라 지울 수 없어요" : "녹음 삭제"}
        >
          <Trash2Icon />
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>이 녹음을 지울까요?</AlertDialogTitle>
          <AlertDialogDescription>
            <span className="font-mono break-all">{recording.id}</span>
            <br />
            오디오와 이벤트 기록이 이 머신에서 완전히 지워져요. 되돌릴 수 없어요.
          </AlertDialogDescription>
        </AlertDialogHeader>
        {remove.error && <p className="text-sm text-destructive">{remove.error.message}</p>}
        <AlertDialogFooter>
          <AlertDialogCancel>그대로 두기</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            onClick={() => remove.mutate(recording.id, { onSuccess: () => onDeleted?.() })}
          >
            삭제
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
