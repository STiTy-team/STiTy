import { ChevronDownIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { modelLine } from "@/lib/compare/runs"
import type { RunOverview } from "@/lib/compare/types"

export function ModelMenu({
  runs,
  hidden,
  colors,
  onChange,
}: {
  runs: RunOverview[]
  hidden: Set<string>
  colors: Map<string, string>
  onChange: (hidden: Set<string>) => void
}) {
  const shown = runs.filter((r) => !hidden.has(r.name)).length
  const flip = (name: string, on: boolean) => {
    const next = new Set(hidden)
    if (on) next.delete(name)
    else next.add(name)
    onChange(next)
  }
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline">
          실행: {shown === runs.length ? `전체 ${runs.length}개` : `${runs.length}개 중 ${shown}개`}
          <ChevronDownIcon className="opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-96 p-2">
        <div className="mb-1 flex gap-1.5 border-b pb-2">
          <Button size="sm" variant="secondary" onClick={() => onChange(new Set())}>
            모두 선택
          </Button>
          <Button size="sm" variant="secondary" onClick={() => onChange(new Set(runs.map((r) => r.name)))}>
            모두 해제
          </Button>
        </div>
        <div className="flex max-h-96 flex-col gap-0.5 overflow-auto">
          {runs.map((run) => (
            <label key={run.name} className="flex cursor-pointer items-start gap-2.5 rounded-md p-2 hover:bg-muted">
              <Checkbox checked={!hidden.has(run.name)} onCheckedChange={(on) => flip(run.name, on === true)} className="mt-0.5" />
              <span className="mt-1.5 size-2 shrink-0 rounded-full" style={{ background: colors.get(run.name) }} />
              <span className="min-w-0">
                <span className="font-mono text-xs">{run.pipeline}</span>
                {run.status && run.status !== "ok" && <span className="ml-1.5 text-xs text-muted-foreground">{run.status}</span>}
                <span className="block text-xs text-muted-foreground">{modelLine(run)}</span>
              </span>
            </label>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  )
}
