import { useQueries, useQuery } from "@tanstack/react-query"

import { get } from "@/lib/api"
import type { Distributions, RunOverview } from "@/lib/compare/types"

export function useOverview(enabled = true) {
  return useQuery({
    queryKey: ["overview"],
    queryFn: () => get<{ runs: RunOverview[] }>("/api/overview"),
    enabled,
  })
}

/** Per-item score distributions of each run; a run whose request fails counts as having none. */
export function useDistributions(runs: string[]) {
  return useQueries({
    queries: runs.map((run) => ({
      queryKey: ["distributions", run],
      queryFn: () => get<Distributions>(`/api/distributions/${encodeURIComponent(run)}`).catch((): Distributions => ({})),
    })),
    combine: (results) => ({
      pending: results.some((r) => r.isPending),
      byRun: new Map(runs.map((run, i) => [run, results[i].data ?? {}])),
    }),
  })
}
