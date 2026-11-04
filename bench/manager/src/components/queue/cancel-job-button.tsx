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
import type { Job } from "@/lib/api"
import { useCancelJob } from "@/lib/queries"

export function CancelJobButton({ host, job }: { host: string; job: Job }) {
  const cancel = useCancelJob(host)
  const running = job.state !== "queued"
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button variant="outline" size="sm" disabled={job.state === "cancelling" || cancel.isPending}>
          {job.state === "cancelling" ? "취소 중…" : "취소"}
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>이 작업을 취소할까요?</AlertDialogTitle>
          <AlertDialogDescription>
            <span className="font-mono">
              {job.pipeline} × {job.dataset}
            </span>{" "}
            · {host}
            <br />
            {running ? "worker가 다음 확인 때 멈추고, 대기열에서 빠져요." : "지금 바로 대기열에서 빠져요."}
          </AlertDialogDescription>
        </AlertDialogHeader>
        {cancel.error && <p className="text-sm text-destructive">{cancel.error.message}</p>}
        <AlertDialogFooter>
          <AlertDialogCancel>그대로 두기</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={() => cancel.mutate(job.id)}>
            작업 취소
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
