import { useNavigate } from "react-router"

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import type { Machine } from "@/lib/api"
import { useRemoveMachine } from "@/lib/queries"

export function RemoveMachineDialog({
  machine,
  open,
  onOpenChange,
}: {
  machine: Machine
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const remove = useRemoveMachine()
  const navigate = useNavigate()
  const waiting = machine.jobs.length
  const removeAndLeave = () =>
    remove.mutate(machine.host, {
      onSuccess: () => {
        onOpenChange(false)
        navigate("/queue")
      },
    })
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>
            <span className="font-mono">{machine.host}</span> 머신을 지울까요?
          </AlertDialogTitle>
          <AlertDialogDescription>
            {machine.connected
              ? "이 머신의 worker가 아직 연결돼 있어서 지울 수 없어요. 그 머신에서 scripts/bench/worker/stop.sh 로 worker를 끈 뒤, 5분 지나 연결 끊김으로 바뀌면 지울 수 있어요."
              : `설명, 작업 시간표, 기록이 모두 지워지고 되돌릴 수 없어요.${waiting ? ` 대기열에 남은 작업 ${waiting}개도 함께 사라져요.` : ""} 올라간 실행 결과는 남아요.`}
          </AlertDialogDescription>
        </AlertDialogHeader>
        {remove.error && <p className="text-sm text-destructive">{remove.error.message}</p>}
        <AlertDialogFooter>
          <AlertDialogCancel>그대로 두기</AlertDialogCancel>
          {!machine.connected && (
            <AlertDialogAction
              variant="destructive"
              disabled={remove.isPending}
              onClick={(event) => {
                event.preventDefault()
                removeAndLeave()
              }}
            >
              머신 지우기
            </AlertDialogAction>
          )}
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
