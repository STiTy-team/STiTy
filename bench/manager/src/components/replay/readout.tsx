import { Empty } from "@/components/replay/kv"
import type { ReplayItem } from "@/lib/replay/types"

export function Reference({ item }: { item: ReplayItem | null }) {
  if (!item) return <Empty>—</Empty>
  const lines: [string, string][] = []
  if (item.reference) lines.push(["전사", item.reference])
  for (const [lang, text] of Object.entries(item.reference_translations))
    if (text) lines.push([lang === item.target_lang ? `${lang} · 번역 대상` : lang, text])
  if (!lines.length) return <Empty>이 항목에는 참조 문장이 없어요.</Empty>
  return (
    <dl className="grid gap-4 sm:grid-cols-[repeat(auto-fill,minmax(16rem,1fr))]">
      {lines.map(([label, text]) => (
        <div key={label} className="flex flex-col gap-1">
          <dt className="text-xs text-muted-foreground">{label}</dt>
          <dd className="text-[15px] leading-relaxed whitespace-pre-wrap">{text}</dd>
        </div>
      ))}
    </dl>
  )
}
