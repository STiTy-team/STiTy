import { Badge } from "@/components/ui/badge"
import { formatClock } from "@/lib/format"
import type { ItemRow as ItemRowData } from "@/lib/datasets/types"
import { cn } from "@/lib/utils"

export function ItemRow({ item, selected, onOpen }: { item: ItemRowData; selected: boolean; onOpen: () => void }) {
  const ref = item.reference ?? {}
  const langs = Object.keys(ref.translations ?? {})
  return (
    <div
      role="button"
      tabIndex={0}
      data-state={selected ? "selected" : undefined}
      className="flex cursor-pointer flex-col gap-1.5 border-b px-3 py-2.5 last:border-b-0 hover:bg-accent/50 data-[state=selected]:bg-accent"
      onClick={onOpen}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          onOpen()
        }
      }}
    >
      <div className="flex min-w-0 items-center gap-2">
        <span className="shrink-0 text-xs text-muted-foreground tabular-nums">#{item.line}</span>
        <span className="min-w-0 truncate font-mono text-xs font-medium">{item.id}</span>
        <span className="shrink-0 text-xs font-medium tabular-nums">{formatClock(item.duration ?? 0)}</span>
        {item.src_lang && (
          <Badge variant="outline" className="shrink-0 font-mono text-[11px]">
            {item.src_lang}
          </Badge>
        )}
        {item.speaker && <Badge variant="secondary" className="shrink-0 text-[11px]">{item.speaker}</Badge>}
        {item.speakers != null && item.speakers > 0 && (
          <Badge variant="secondary" className="shrink-0 text-[11px]">
            화자 {item.speakers}
          </Badge>
        )}
        {item.group !== item.id && (
          <span className="shrink-0 text-xs text-muted-foreground">group {item.group}</span>
        )}
        {item.offset != null && <span className="shrink-0 text-xs text-muted-foreground">+{item.offset}s</span>}
        {item.partial && (
          <Badge variant="outline" className="shrink-0 text-[11px] text-destructive">
            partial
          </Badge>
        )}
        {!item.has_audio && (
          <Badge variant="outline" className="shrink-0 text-[11px] text-destructive">
            오디오 없음
          </Badge>
        )}
      </div>
      <p className={cn("truncate text-sm", !ref.transcript && "text-muted-foreground italic")}>
        {ref.transcript || "전사 없음"}
        {langs.length > 0 && <span className="text-muted-foreground"> · 번역 {langs.join(", ")}</span>}
      </p>
    </div>
  )
}
