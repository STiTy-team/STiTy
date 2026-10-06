import { keepPreviousData, useQuery } from "@tanstack/react-query"

import { get } from "@/lib/api"
import type { ItemData, ReplayItem, ReplayPayload } from "@/lib/replay/types"

const params = (run: string, item?: string) =>
  new URLSearchParams(item == null ? { run } : { run, item }).toString()

export const replayUrls = {
  run: (run: string | null) => `/api/replay${run ? `?${params(run)}` : ""}`,
  item: (run: string, item: string) => `/api/replay/item?${params(run, item)}`,
  data: (run: string, item: string) => `/api/replay/data?${params(run, item)}`,
  audio: (run: string, item: string) => `/api/replay/audio?${params(run, item)}`,
}

export function useReplay(run: string | null) {
  return useQuery({
    queryKey: ["replay", run],
    queryFn: () => get<ReplayPayload>(replayUrls.run(run)),
    placeholderData: keepPreviousData,
  })
}

export function useReplayItem(run: string | null, item: string | null, initial?: ReplayItem) {
  return useQuery({
    queryKey: ["replay", run, "item", item],
    queryFn: () => get<ReplayItem>(replayUrls.item(run!, item!)),
    enabled: Boolean(run && item),
    initialData: initial && initial.id === item ? initial : undefined,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  })
}

export function useItemData(run: string, item: string) {
  return useQuery({
    queryKey: ["replay", run, "data", item],
    queryFn: () => get<ItemData>(replayUrls.data(run, item)),
    staleTime: Infinity,
  })
}
