import { ArrowLeftIcon, CheckIcon, ChevronLeftIcon, ChevronRightIcon, ChevronsUpDownIcon, InfoIcon } from "lucide-react"
import { useState } from "react"
import { Link } from "react-router"

import { StatTile } from "@/components/replay/kv"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { ITEM_METRICS, stampText } from "@/lib/replay/format"
import type { ConfigMeta, ItemMetrics, ItemRow, ReplayPayload } from "@/lib/replay/types"
import { cn } from "@/lib/utils"

const itemLabel = (row: ItemRow) => (row.wer == null ? row.id : `${row.id} · WER ${row.wer.toFixed(2)}`)

function ItemPicker({ data, value, onChange }: { data: ReplayPayload; value: string; onChange: (item: string) => void }) {
  const [open, setOpen] = useState(false)
  const worstIds = new Set(data.worst.map((row) => row.id))
  const rest = data.all_items.filter((row) => !worstIds.has(row.id))
  const groups: [string, ItemRow[]][] = [
    [`실패 + 하위 ${data.run.top_k}개`, data.worst],
    [`나머지 ${rest.length}개`, rest],
  ]
  const pick = (id: string) => {
    onChange(id)
    setOpen(false)
  }
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" role="combobox" aria-expanded={open} aria-label="항목" className="max-w-full font-mono text-xs">
          <span className="truncate">{value}</span>
          <ChevronsUpDownIcon className="opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-96 p-0" align="start">
        <Command>
          <CommandInput placeholder="항목 찾기…" />
          <CommandList>
            <CommandEmpty>맞는 항목이 없어요.</CommandEmpty>
            {groups
              .filter(([, rows]) => rows.length)
              .map(([heading, rows]) => (
                <CommandGroup key={heading} heading={heading}>
                  {rows.map((row) => (
                    <CommandItem key={row.id} value={row.id} onSelect={pick} className="font-mono text-xs">
                      <CheckIcon className={cn(row.id === value ? "opacity-100" : "opacity-0")} />
                      {itemLabel(row)}
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

export function MetaLine({ kind, meta }: { kind: string; meta?: ConfigMeta }) {
  if (!meta) return null
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <span className="w-16 text-xs text-muted-foreground">{kind}</span>
      <span className="font-mono text-xs" title={`config hash ${meta.hash || "기록 없음"}`}>
        {meta.ref}
      </span>
      {meta.tags.map((tag) => (
        <Badge key={tag} variant="secondary">
          {tag}
        </Badge>
      ))}
      {meta.description && <span className="text-muted-foreground">{meta.description}</span>}
    </div>
  )
}

const LEAD_KEYS = ["wer", "sentence_bleu", "avg_fsl_sec"]
const HEADER_KEYS = ["wer", "sentence_bleu", "avg_fsl_sec", "laal_ms", "cer", "n_segments"]

/** At most four of the item's numbers; the one the worst-first list ranks by is the big one. */
export function ItemStats({ metrics }: { metrics?: ItemMetrics }) {
  const has = (k: string) => metrics?.[k] != null
  const lead = LEAD_KEYS.find(has)
  const keys = [lead, ...HEADER_KEYS.filter((k) => k !== lead && has(k))].filter(Boolean).slice(0, 4) as string[]
  if (!keys.length) return null
  return (
    <div className="flex flex-wrap items-end gap-x-8 gap-y-2">
      {keys.map((k) => (
        <StatTile key={k} label={ITEM_METRICS[k].label} value={ITEM_METRICS[k].format(metrics![k]!)} lead={k === lead} />
      ))}
    </div>
  )
}

function RunInfo({ data }: { data: ReplayPayload }) {
  const { run, worst } = data
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm">
          <InfoIcon />
          실행 정보
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="flex w-[26rem] max-w-[calc(100vw-2rem)] flex-col gap-3 text-sm">
        <p className="font-mono text-xs break-all">{run.name}</p>
        {run.stamp && <p className="text-muted-foreground tabular-nums">{stampText(run.stamp)} 실행</p>}
        <p className="text-muted-foreground">
          전체 {run.n_total}개 항목 중 {worst.length}개를 먼저 보여줘요.
          {worst.length < run.n_total && " 실패한 항목이 먼저, 그다음 WER(없으면 BLEU, FSL)이 나쁜 순서예요."}
        </p>
        <div className="flex flex-col gap-1">
          <MetaLine kind="dataset" meta={run.meta.dataset} />
          <MetaLine kind="pipeline" meta={run.meta.pipeline} />
        </div>
      </PopoverContent>
    </Popover>
  )
}

export function ReplayControls({
  data,
  itemId,
  onItem,
}: {
  data: ReplayPayload
  itemId: string
  onItem: (item: string) => void
}) {
  const { run } = data
  const failed = run.status && !["ok", "running"].includes(run.status)
  const worstIds = new Set(data.worst.map((row) => row.id))
  const order = [...data.worst, ...data.all_items.filter((row) => !worstIds.has(row.id))].map((row) => row.id)
  const at = order.indexOf(itemId)
  const step = (by: number) => order[at + by] && onItem(order[at + by])
  return (
    <div className="flex min-w-0 flex-wrap items-center gap-2">
      <Button variant="ghost" size="sm" asChild className="text-muted-foreground">
        <Link to="/runs">
          <ArrowLeftIcon />
          실행 목록
        </Link>
      </Button>
      <span className="max-w-[28rem] truncate font-mono text-xs" title={run.name}>
        {run.name}
      </span>
      <div className="flex items-center gap-1">
        <Button variant="outline" size="icon" aria-label="이전 항목" disabled={at <= 0} onClick={() => step(-1)}>
          <ChevronLeftIcon />
        </Button>
        <ItemPicker data={data} value={itemId} onChange={onItem} />
        <Button variant="outline" size="icon" aria-label="다음 항목" disabled={at < 0 || at >= order.length - 1} onClick={() => step(1)}>
          <ChevronRightIcon />
        </Button>
      </div>
      {run.status && <Badge variant={failed ? "destructive" : "outline"}>{run.status}</Badge>}
      <RunInfo data={data} />
    </div>
  )
}
