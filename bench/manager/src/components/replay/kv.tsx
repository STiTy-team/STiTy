import { ChevronRightIcon } from "lucide-react"
import type { ReactNode } from "react"

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { byGroup, formatValue, isRecord, ITEM_METRICS, labelOf, leaves } from "@/lib/replay/format"
import type { ItemMetrics } from "@/lib/replay/types"
import { cn } from "@/lib/utils"

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-2 text-sm text-muted-foreground">{children}</p>
}

export function StatTile({ label, value, code, lead }: { label: string; value: string; code?: boolean; lead?: boolean }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <span className={cn("text-xs text-muted-foreground", code && "font-mono")}>{label}</span>
      <span
        className={cn(
          "text-sm font-medium tabular-nums",
          lead && "text-2xl font-bold tracking-tight",
          code && "font-mono text-xs font-normal break-all",
        )}
      >
        {value}
      </span>
    </div>
  )
}

function Tiles({ pairs, parent, labels = true }: { pairs: [string, unknown][]; parent?: string; labels?: boolean }) {
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(8rem,1fr))] gap-x-6 gap-y-3">
      {pairs.map(([key, value]) => {
        const last = key.split(".").at(-1)!
        const name = labels ? labelOf(last) : null
        return (
          <StatTile
            key={key}
            label={name ?? (labels ? last : key)}
            value={formatValue(last, value, parent)}
            code={typeof value !== "number"}
          />
        )
      })}
    </div>
  )
}

function KvGroup({ title, sub, children }: { title: string; sub?: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="text-sm font-semibold">
        {title}
        {sub && <span className="ml-1 font-mono text-xs font-normal text-muted-foreground">{sub}</span>}
      </div>
      {children}
    </div>
  )
}

function FoldGroup({ title, sub, defaultOpen, children }: { title: string; sub?: string; defaultOpen: boolean; children: ReactNode }) {
  return (
    <Collapsible defaultOpen={defaultOpen} className="group/fold border-t first:border-t-0">
      <CollapsibleTrigger className="flex w-full items-center gap-2 py-3 text-left text-sm font-semibold hover:text-foreground/80">
        <ChevronRightIcon className="size-4 text-muted-foreground transition-transform group-data-[state=open]/fold:rotate-90" />
        {title}
        {sub && <span className="font-mono text-xs font-normal text-muted-foreground">{sub}</span>}
      </CollapsibleTrigger>
      <CollapsibleContent className="pb-4 pl-6">{children}</CollapsibleContent>
    </Collapsible>
  )
}

export function ItemMetricTiles({ metrics }: { metrics?: ItemMetrics }) {
  const keys = Object.keys(ITEM_METRICS).filter((k) => metrics?.[k] != null)
  if (!keys.length) return <Empty>이 항목에는 항목별 지표가 없어요.</Empty>
  return (
    <div className="flex flex-col gap-4">
      {byGroup(keys).map(([title, ks]) => (
        <KvGroup key={title} title={title}>
          <div className="flex flex-wrap gap-x-6 gap-y-3">
            {ks.map((k) => (
              <StatTile key={k} label={ITEM_METRICS[k].label} value={ITEM_METRICS[k].format(metrics![k]!)} />
            ))}
          </div>
        </KvGroup>
      ))}
    </div>
  )
}

/** The loose top-level values first, then one group per pipeline component. */
export function PipelineConfig({ config }: { config: Record<string, unknown> }) {
  if (!Object.keys(config).length) return <Empty>기록된 파이프라인 설정이 없어요.</Empty>
  const { pipeline = {}, ...top } = config
  const loose: [string, unknown][] = []
  const groups: ReactNode[] = []
  for (const [k, v] of Object.entries(isRecord(pipeline) ? pipeline : {})) {
    if (isRecord(v)) {
      const { name, ...rest } = v
      groups.push(
        <FoldGroup key={`p.${k}`} title={k} sub={name ? String(name) : undefined} defaultOpen={false}>
          <Tiles pairs={leaves(rest)} labels={false} />
        </FoldGroup>,
      )
    } else loose.push([k === "name" ? "pipeline" : k, v])
  }
  for (const [k, v] of Object.entries(top)) {
    if (isRecord(v))
      groups.push(
        <FoldGroup key={k} title={k} defaultOpen={false}>
          <Tiles pairs={leaves(v)} labels={false} />
        </FoldGroup>,
      )
    else loose.push([k, v])
  }
  return (
    <div className="flex flex-col">
      {loose.length > 0 && (
        <FoldGroup title="실행" defaultOpen>
          <Tiles pairs={loose} labels={false} />
        </FoldGroup>
      )}
      {groups}
    </div>
  )
}

/** One group per kind of score; a breakdown (by lang, confusion, ...) sits inside its kind's group. */
export function RunSummary({ metrics, running }: { metrics: Record<string, unknown>; running: boolean }) {
  if (!Object.keys(metrics).length)
    return <Empty>{running ? "아직 실행 중이에요. 끝나면 요약이 나와요." : "실행 요약이 아직 없어요."}</Empty>
  return (
    <div className="flex flex-col">
      {byGroup(Object.keys(metrics)).map(([title, keys], index) => {
        const flat = keys.filter((k) => !isRecord(metrics[k]))
        const nested = keys.filter((k) => isRecord(metrics[k]))
        return (
          <FoldGroup key={title} title={title} defaultOpen={index === 0}>
            {flat.length > 0 && <Tiles pairs={flat.map((k) => [k, metrics[k]])} />}
            {nested.map((k) => (
              <div key={k} className="flex flex-col gap-2">
                <div className="text-xs font-semibold text-muted-foreground">{labelOf(k) ?? k}</div>
                <Tiles pairs={leaves(metrics[k] as Record<string, unknown>)} parent={k} labels={false} />
              </div>
            ))}
          </FoldGroup>
        )
      })}
    </div>
  )
}
