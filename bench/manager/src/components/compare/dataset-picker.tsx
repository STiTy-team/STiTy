import { CheckIcon, ChevronsUpDownIcon } from "lucide-react"
import { useState } from "react"

import { Button } from "@/components/ui/button"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import type { RunOverview } from "@/lib/compare/types"
import { formatDateTime, timeAgo } from "@/lib/format"
import { stampIso } from "@/lib/replay/format"
import { cn } from "@/lib/utils"

type DatasetRow = { key: string; corpus: string; runs: number; latest: string | null }

function datasetRows(runs: RunOverview[]): DatasetRow[] {
  const byKey = new Map<string, DatasetRow>()
  for (const run of runs) {
    const row = byKey.get(run.dataset_key) ?? { key: run.dataset_key, corpus: run.corpus, runs: 0, latest: null }
    row.runs += 1
    const at = stampIso(run.stamp)
    if (at && (!row.latest || at > row.latest)) row.latest = at
    byKey.set(run.dataset_key, row)
  }
  return [...byKey.values()].sort((a, b) => a.key.localeCompare(b.key))
}

/** Searchable, always opens downward; each dataset shows how many runs it has and when it last ran. */
export function DatasetPicker({ runs, value, onChange }: { runs: RunOverview[]; value: string; onChange: (key: string) => void }) {
  const [open, setOpen] = useState(false)
  const rows = datasetRows(runs)
  const corpora = [...new Set(rows.map((row) => row.corpus))].sort()
  const pick = (key: string) => {
    onChange(key)
    setOpen(false)
  }
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" role="combobox" aria-expanded={open} aria-label="데이터셋" className="max-w-full min-w-64 justify-between font-mono text-xs">
          <span className="truncate">{value}</span>
          <ChevronsUpDownIcon className="opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent side="bottom" align="start" avoidCollisions={false} className="w-[30rem] max-w-[calc(100vw-2rem)] p-0">
        <Command>
          <CommandInput placeholder="데이터셋 찾기…" />
          <CommandList className="max-h-[min(24rem,60vh)]">
            <CommandEmpty>맞는 데이터셋이 없어요.</CommandEmpty>
            {corpora.map((corpus) => (
              <CommandGroup key={corpus} heading={corpus}>
                {rows
                  .filter((row) => row.corpus === corpus)
                  .map((row) => (
                    <CommandItem key={row.key} value={row.key} onSelect={pick} className="gap-2">
                      <CheckIcon className={cn("shrink-0", row.key === value ? "opacity-100" : "opacity-0")} />
                      <span className="min-w-0 flex-1 truncate font-mono text-xs" title={row.key}>
                        {row.key}
                      </span>
                      <span className="shrink-0 text-xs text-muted-foreground tabular-nums" title={formatDateTime(row.latest)}>
                        실행 {row.runs}개 · {timeAgo(row.latest)}
                      </span>
                    </CommandItem>
                  ))}
              </CommandGroup>
            ))}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
