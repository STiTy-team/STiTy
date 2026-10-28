import { RotateCwIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import type { Job } from "@/lib/api"
import { useRetryJob } from "@/lib/queries"

export function RetryJobButton({ host, job }: { host: string; job: Job }) {
  const retry = useRetryJob(host)
  return (
    <div className="flex flex-col items-end gap-1">
      <Button
        variant="outline"
        size="sm"
        title="같은 설정과 브랜치로 대기열에 다시 넣어요"
        onClick={() => retry.mutate(job)}
        disabled={retry.isPending}
      >
        <RotateCwIcon />
        {retry.isPending ? "추가 중…" : retry.isSuccess ? "대기열에 추가됨" : "다시 시도하기"}
      </Button>
      {retry.error && <span className="text-xs text-destructive">{retry.error.message}</span>}
    </div>
  )
}
