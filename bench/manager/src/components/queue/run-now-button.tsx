import { ZapIcon } from "lucide-react"

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
import type { Job, Machine } from "@/lib/api"
import { useRunNow } from "@/lib/queries"

export function RunNowButton({ machine, job }: { machine: Machine; job: Job }) {
  const runNow = useRunNow(machine.host)
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button variant="outline" size="sm" disabled={runNow.isPending}>
          <ZapIcon />
          지금 실행
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>시간표와 상관없이 바로 실행할까요?</AlertDialogTitle>
          <AlertDialogDescription>
            <span className="font-mono">
              {job.pipeline} × {job.dataset}
            </span>{" "}
            · {machine.host}
            <br />
            지금 도는 작업이 없으면 worker가 다음 확인(1분 안) 때 시작하고, 있으면 그 작업이 끝난 뒤 가장 먼저 시작해요.
            시간대가 끝나도 멈추지 않고 끝까지 돌아요.
            {machine.settings.paused && " 이 머신은 일시정지 중이라 풀릴 때까지 기다려요."}
            {!machine.connected && " 이 머신은 지금 연결이 끊겨 있어서 worker가 다시 켜져야 시작해요."}
          </AlertDialogDescription>
        </AlertDialogHeader>
        {runNow.error && <p className="text-sm text-destructive">{runNow.error.message}</p>}
        <AlertDialogFooter>
          <AlertDialogCancel>그대로 두기</AlertDialogCancel>
          <AlertDialogAction onClick={() => runNow.mutate(job.id)}>지금 실행</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
