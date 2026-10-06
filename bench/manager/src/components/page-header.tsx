import { useIsFetching, useQueryClient } from "@tanstack/react-query"
import { RefreshCwIcon } from "lucide-react"
import { useEffect, useState, type ReactNode } from "react"

import { Button } from "@/components/ui/button"
import { SidebarTrigger } from "@/components/ui/sidebar"
import { REFRESH_MS, useReload } from "@/lib/queries"
import { cn } from "@/lib/utils"

export function PageHeader({ title, description }: { title: string; description?: ReactNode }) {
  return (
    <header className="sticky top-0 z-20 flex min-h-14 shrink-0 flex-wrap items-center gap-x-3 gap-y-2 border-b bg-background/95 px-4 py-2.5 backdrop-blur md:px-6">
      <SidebarTrigger className="-ml-1" />
      <h1 className="text-lg font-semibold">{title}</h1>
      {description && <p className="hidden text-sm text-muted-foreground md:block">{description}</p>}
      <span className="flex-1" />
      <ReloadButton />
    </header>
  )
}

function useAutoRefresh(): number | null {
  const queryClient = useQueryClient()
  const [now, setNow] = useState(Date.now)

  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [])

  const polled = queryClient
    .getQueryCache()
    .findAll({ type: "active", predicate: (query) => query.meta?.autoRefresh === true })
  const due = polled.map((query) => Math.max(query.state.dataUpdatedAt, query.state.errorUpdatedAt) + REFRESH_MS)
  const next = due.length ? Math.min(...due) : null

  useEffect(() => {
    if (next === null || now < next || document.hidden) return
    for (const query of polled) {
      if (query.state.fetchStatus === "idle") void queryClient.refetchQueries({ queryKey: query.queryKey, exact: true })
    }
  })

  return next === null ? null : Math.max(0, Math.ceil((next - now) / 1000))
}

function ReloadButton() {
  const reload = useReload()
  const fetching = useIsFetching()
  const secondsLeft = useAutoRefresh()
  const busy = reload.isPending || fetching > 0
  return (
    <Button variant="outline" size="sm" onClick={() => reload.mutate()} disabled={busy}>
      <RefreshCwIcon className={cn(busy && "animate-spin motion-reduce:animate-none")} />
      새로고침
      {secondsLeft !== null && !busy && <span className="tabular-nums text-muted-foreground">{secondsLeft}초</span>}
    </Button>
  )
}

export function PageBody({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("mx-auto flex w-full max-w-[1280px] flex-col gap-6 px-4 py-6 md:px-6", className)}>{children}</div>
}

export function Notice({ children }: { children: ReactNode }) {
  return <p className="text-sm text-muted-foreground">{children}</p>
}
