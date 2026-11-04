import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card"
import type { Machine, WorkerState } from "@/lib/api"
import { formatDateTime, shortSha, timeAgo } from "@/lib/format"
import { cn } from "@/lib/utils"

const STATE_LABEL: Record<WorkerState, string> = {
  idle: "작업 기다리는 중",
  running: "작업 실행 중",
  outside_window: "시간표 밖이라 쉬는 중",
  backoff: "GPU가 차 있어 잠시 미루는 중",
  paused: "일시정지",
  stopping: "멈추는 중",
}

const GB = 1024

function Meter({ fraction }: { fraction: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
      <div
        className={cn("h-full rounded-full", fraction >= 0.9 ? "bg-status-failed" : "bg-primary")}
        style={{ width: `${Math.round(100 * Math.min(1, fraction))}%` }}
      />
    </div>
  )
}

function Row({ label, value, fraction }: { label: string; value: string; fraction?: number }) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between gap-3">
        <span className="min-w-0 truncate text-muted-foreground">{label}</span>
        <span className="shrink-0 tabular-nums">{value}</span>
      </div>
      {fraction != null && <Meter fraction={fraction} />}
    </div>
  )
}

function Details({ machine }: { machine: Machine }) {
  const health = machine.health
  if (!health) return <p className="text-muted-foreground">이 머신의 worker가 아직 상태를 보낸 적이 없어요.</p>
  const disk = health.disk_free_gb != null && health.disk_total_gb ? 1 - health.disk_free_gb / health.disk_total_gb : null
  return (
    <>
      <div className="flex flex-col gap-0.5">
        <p className="font-medium">{machine.connected ? STATE_LABEL[health.state] : "연결 끊김"}</p>
        <p className="text-xs text-muted-foreground" title={formatDateTime(health.updated_at)}>
          {timeAgo(health.updated_at)} 갱신
          {health.window && (health.window.open
            ? health.window.ends_at && ` · 시간표 ${formatDateTime(health.window.ends_at)}까지`
            : health.window.next_opens_at && ` · 다음 시간표 ${formatDateTime(health.window.next_opens_at)}`)}
        </p>
      </div>

      {health.gpus.length ? (
        health.gpus.map((gpu) => (
          <div key={gpu.index} className="flex flex-col gap-2 border-t pt-3">
            <p className="truncate text-xs font-medium">
              GPU {gpu.index} · {gpu.name}
            </p>
            <Row
              label="VRAM"
              value={`${(gpu.memory_used_mb / GB).toFixed(1)} / ${(gpu.memory_total_mb / GB).toFixed(1)} GB`}
              fraction={gpu.memory_total_mb ? gpu.memory_used_mb / gpu.memory_total_mb : 0}
            />
            <Row label="사용률" value={gpu.util == null ? "—" : `${gpu.util.toFixed(0)}%`} fraction={gpu.util == null ? undefined : gpu.util / 100} />
          </div>
        ))
      ) : (
        <p className="border-t pt-3 text-muted-foreground">GPU가 없어요 (nvidia-smi 없음).</p>
      )}

      {health.disk_free_gb != null && (
        <div className="border-t pt-3">
          <Row
            label="디스크"
            value={health.disk_total_gb ? `${health.disk_free_gb.toFixed(0)} GB 남음 / ${health.disk_total_gb.toFixed(0)} GB` : `${health.disk_free_gb.toFixed(0)} GB 남음`}
            fraction={disk ?? undefined}
          />
        </div>
      )}

      {health.backoff && (
        <p className="text-xs text-muted-foreground">
          {health.backoff.level}단계 대기 · {formatDateTime(health.backoff.until)}에 다시 시도
        </p>
      )}
      {health.last_error && <p className="rounded-md bg-status-failed-soft px-2 py-1.5 text-xs break-all text-status-failed">{health.last_error}</p>}
      {health.worker_commit && <p className="font-mono text-[11px] text-muted-foreground">worker {shortSha(health.worker_commit)}</p>}
    </>
  )
}

export function MachineDot({ machine }: { machine: Machine }) {
  const running = machine.connected && machine.health?.state === "running"
  const label = !machine.connected ? "연결 끊김" : running ? "작업 실행 중" : "연결됨"
  return (
    <span
      role="img"
      aria-label={label}
      title={label}
      className={cn("size-2 shrink-0 rounded-full", !machine.connected ? "bg-status-failed" : running ? "bg-status-running" : "bg-status-done")}
    />
  )
}

/** One thin bar per GPU, each filled with that GPU's VRAM use. Numbers stay behind hover or focus. */
export function GpuBars({ machine }: { machine: Machine }) {
  const gpus = machine.health?.gpus ?? []
  const summary = gpus.length
    ? gpus.map((gpu) => `GPU ${gpu.index} VRAM ${Math.round((100 * gpu.memory_used_mb) / (gpu.memory_total_mb || 1))}%`).join(", ")
    : "GPU 없음"
  return (
    <HoverCard>
      <HoverCardTrigger asChild>
        <button
          type="button"
          aria-label={`${machine.host} GPU 상태: ${summary}`}
          className="inline-flex h-7 items-center gap-1.5 rounded-md border px-2 outline-none hover:bg-muted focus-visible:ring-3 focus-visible:ring-ring/50"
        >
          <span className="text-[11px] font-medium text-muted-foreground">GPU</span>
          {gpus.length ? (
            <span className="flex h-4 items-end gap-[3px]">
              {gpus.map((gpu) => {
                const used = gpu.memory_total_mb ? gpu.memory_used_mb / gpu.memory_total_mb : 0
                return (
                  <span key={gpu.index} className="relative h-full w-2 overflow-hidden rounded-[2px] bg-muted-foreground/25">
                    <span
                      className={cn(
                        "absolute inset-x-0 bottom-0 rounded-[2px] transition-[height] duration-700 motion-reduce:transition-none",
                        !machine.connected ? "bg-muted-foreground/40" : used >= 0.9 ? "bg-status-failed" : "bg-primary",
                      )}
                      style={{ height: `${Math.max(used > 0 ? 8 : 0, Math.round(100 * Math.min(1, used)))}%` }}
                    />
                  </span>
                )
              })}
            </span>
          ) : (
            <span className="text-[11px] text-muted-foreground">없음</span>
          )}
        </button>
      </HoverCardTrigger>
      <HoverCardContent>
        <Details machine={machine} />
      </HoverCardContent>
    </HoverCard>
  )
}
