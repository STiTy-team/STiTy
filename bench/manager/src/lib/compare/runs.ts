import { metric, SCORES } from "@/lib/compare/scores"
import type { RunOverview } from "@/lib/compare/types"

export const modelLine = (run: RunOverview) =>
  [run.models.transcription, run.models.correction, run.models.translation].filter(Boolean).join(" → ")

/** The SCORES keys at least one of the runs has a summary value for. */
export const scoreKeys = (runs: RunOverview[]) =>
  Object.keys(SCORES).filter((k) => runs.some((r) => metric(r.summary_metrics, k) != null))

/** Runs that are missing one of the two scores, and why. */
export function missingRuns(runs: RunOverview[], xKey: string, yKey: string) {
  return runs
    .filter((r) => metric(r.summary_metrics, xKey) == null || metric(r.summary_metrics, yKey) == null)
    .map((r) => {
      if (r.status === "running") return `${r.pipeline} (아직 실행 중이라 요약이 없어요)`
      const absent = [metric(r.summary_metrics, yKey) == null && SCORES[yKey].label, metric(r.summary_metrics, xKey) == null && SCORES[xKey].label]
      return `${r.pipeline} (${absent.filter(Boolean).join(", ")} 값 없음)`
    })
}
