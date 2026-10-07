import { CheckIcon, ChevronsUpDownIcon } from "lucide-react"
import { useState } from "react"

import { Button } from "@/components/ui/button"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { formatClock } from "@/lib/format"
import type { DatasetSummary } from "@/lib/datasets/types"
import { cn } from "@/lib/utils"

const corpusOf = (name: string) => name.split("/")[0]

/** Searchable, grouped by top-level corpus (fleurs/en_us and fleurs/ko_kr group under
 * "fleurs"), each row showing its item count and total recorded length. */
export function DatasetPicker({
  datasets,
  value,
  onChange,
}: {
  datasets: DatasetSummary[]
  value: string | null
  onChange: (name: string) => void
}) {
  const [open, setOpen] = useState(false)
  const corpora = [...new Set(datasets.map((d) => corpusOf(d.name)))]
  const pick = (name: string) => {
    onChange(name)
    setOpen(false)
  }
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" role="combobox" aria-expanded={open} aria-label="데이터셋" className="max-w-full min-w-56 justify-between font-mono text-xs">
          <span className="truncate">{value ?? "데이터셋 없음"}</span>
          <ChevronsUpDownIcon className="opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent side="bottom" align="start" avoidCollisions={false} className="w-[28rem] max-w-[calc(100vw-2rem)] p-0">
        <Command>
          <CommandInput placeholder="데이터셋 찾기…" />
          <CommandList className="max-h-[min(24rem,60vh)]">
            <CommandEmpty>맞는 데이터셋이 없어요.</CommandEmpty>
            {corpora.map((corpus) => (
              <CommandGroup key={corpus} heading={corpus}>
                {datasets
                  .filter((d) => corpusOf(d.name) === corpus)
                  .map((d) => (
                    <CommandItem key={d.name} value={d.name} onSelect={pick} className="gap-2">
                      <CheckIcon className={cn("shrink-0", d.name === value ? "opacity-100" : "opacity-0")} />
                      <span className="min-w-0 flex-1 truncate font-mono text-xs" title={d.name}>
                        {d.name}
                      </span>
                      <span className="shrink-0 text-xs text-muted-foreground tabular-nums">
                        {d.items}개 · {formatClock(d.duration)}
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
