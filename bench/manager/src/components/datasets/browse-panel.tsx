import { useState } from "react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { ItemRow } from "@/components/datasets/item-row"
import { ItemSheet } from "@/components/datasets/item-sheet"
import { ShapePanel } from "@/components/datasets/shape-panel"
import { useDatasetItems } from "@/lib/datasets/queries"
import { useDebounced } from "@/hooks/use-debounced"

const DEBOUNCE_MS = 250

export function BrowsePanel({ ds }: { ds: string }) {
  const [query, setQuery] = useState("")
  const [openItem, setOpenItem] = useState<string | null>(null)
  const q = useDebounced(query.trim(), DEBOUNCE_MS)
  const items = useDatasetItems(ds, q)

  const rows = items.data?.pages.flatMap((p) => p.items) ?? []
  const total = items.data?.pages[0]?.total ?? 0
  const matched = items.data?.pages[0]?.matched ?? 0

  return (
    <div className="flex flex-col gap-6">
      <ShapePanel ds={ds} />

      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-3">
          <h3 className="text-sm font-medium">
            항목{" "}
            {total > 0 && (
              <span className="text-muted-foreground">
                {rows.length} / {q ? `${matched} 일치 (전체 ${total})` : total}
              </span>
            )}
          </h3>
          <span className="flex-1" />
          <Input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="id · 전사 · 번역 · speaker 검색"
            aria-label="항목 검색"
            className="max-w-xs"
          />
        </div>

        {items.isPending ? (
          <Skeleton className="h-64 w-full" />
        ) : items.error ? (
          <p className="text-sm text-destructive">항목을 불러오지 못했어요: {items.error.message}</p>
        ) : rows.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            {q ? "찾는 항목이 없어요." : "이 데이터셋에는 manifest.jsonl이 없어요. convert.py를 돌리거나, 녹음 탭에서 바로 녹음하면 만들어져요."}
          </p>
        ) : (
          <>
            <div className="rounded-xl border">
              {rows.map((item) => (
                <ItemRow key={item.id} item={item} selected={item.id === openItem} onOpen={() => setOpenItem(item.id)} />
              ))}
            </div>
            {items.hasNextPage && (
              <Button
                variant="outline"
                className="self-start"
                disabled={items.isFetchingNextPage}
                onClick={() => items.fetchNextPage()}
              >
                {items.isFetchingNextPage ? "불러오는 중…" : "더 보기"}
              </Button>
            )}
          </>
        )}
      </div>

      {openItem && <ItemSheet ds={ds} id={openItem} onClose={() => setOpenItem(null)} />}
    </div>
  )
}
