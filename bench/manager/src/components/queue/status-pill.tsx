import type { ReactNode } from "react"

import { cn } from "@/lib/utils"

export type Tone = "running" | "done" | "failed" | "queued"

const TONE: Record<Tone, string> = {
  running: "bg-status-running-soft text-status-running",
  done: "bg-status-done-soft text-status-done",
  failed: "bg-status-failed-soft text-status-failed",
  queued: "bg-status-queued-soft text-status-queued",
}

export function StatusPill({ tone, dot, children }: { tone: Tone; dot?: boolean; children: ReactNode }) {
  return (
    <span className={cn("inline-flex h-6 items-center gap-1.5 rounded-full px-2.5 text-xs font-medium whitespace-nowrap", TONE[tone])}>
      {dot && <span className="size-1.5 rounded-full bg-current" />}
      {children}
    </span>
  )
}
