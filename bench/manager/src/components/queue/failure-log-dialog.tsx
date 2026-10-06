import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import type { Finished } from "@/lib/api"

export function FailureLogDialog({ finished, host }: { finished: Finished; host: string }) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          로그 보기
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>실패 원인</DialogTitle>
          <DialogDescription>
            <span className="font-mono">
              {finished.job.pipeline} × {finished.job.dataset}
            </span>{" "}
            · {host} · 작업 로그의 끝부분
          </DialogDescription>
        </DialogHeader>
        <pre className="max-h-[60vh] overflow-auto rounded-md border bg-muted/40 p-3 font-mono text-xs whitespace-pre-wrap">
          {finished.error}
        </pre>
      </DialogContent>
    </Dialog>
  )
}
