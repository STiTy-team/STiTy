import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import { datasetsApi } from "@/lib/datasets/api"
import type { RecordMeta } from "@/lib/datasets/types"

export const datasetKeys = {
  list: ["datasets"] as const,
  shape: (ds: string) => ["datasets", "shape", ds] as const,
  items: (ds: string, q: string, order: string) => ["datasets", "items", ds, q, order] as const,
  item: (ds: string, id: string) => ["datasets", "item", ds, id] as const,
}

export function useDatasetsList() {
  return useQuery({ queryKey: datasetKeys.list, queryFn: datasetsApi.list })
}

export function useCreateDataset() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ name, languages }: { name: string; languages: string }) => datasetsApi.create(name, languages),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: datasetKeys.list }),
  })
}

export function useDatasetShape(ds: string | null) {
  return useQuery({
    queryKey: datasetKeys.shape(ds ?? ""),
    queryFn: () => datasetsApi.shape(ds!),
    enabled: ds != null,
  })
}

export function useDatasetItems(ds: string | null, q: string, order: "" | "recent" = "") {
  return useInfiniteQuery({
    queryKey: datasetKeys.items(ds ?? "", q, order),
    queryFn: ({ pageParam }) => datasetsApi.items(ds!, { offset: pageParam, q, order }),
    enabled: ds != null,
    initialPageParam: 0,
    getNextPageParam: (last) => {
      const shown = last.offset + last.items.length
      return shown < last.matched ? shown : undefined
    },
  })
}

export function useDatasetItem(ds: string | null, id: string | null) {
  return useQuery({
    queryKey: datasetKeys.item(ds ?? "", id ?? ""),
    queryFn: () => datasetsApi.item(ds!, id!),
    enabled: ds != null && id != null,
  })
}

export function useVerifyDataset() {
  return useMutation({ mutationFn: (ds: string) => datasetsApi.verify(ds) })
}

function useInvalidateDataset(ds: string) {
  const queryClient = useQueryClient()
  return () => {
    queryClient.invalidateQueries({ queryKey: datasetKeys.list })
    queryClient.invalidateQueries({ queryKey: datasetKeys.shape(ds) })
    queryClient.invalidateQueries({ queryKey: ["datasets", "items", ds] })
  }
}

export function useDeleteAudio(ds: string) {
  const invalidate = useInvalidateDataset(ds)
  return useMutation({
    mutationFn: (id: string) => datasetsApi.deleteAudio(ds, id),
    onSuccess: invalidate,
  })
}

export function useDeleteItem(ds: string) {
  const invalidate = useInvalidateDataset(ds)
  return useMutation({
    mutationFn: (id: string) => datasetsApi.deleteItem(ds, id),
    onSuccess: invalidate,
  })
}

export function useRecordTake(ds: string) {
  const invalidate = useInvalidateDataset(ds)
  return useMutation({
    mutationFn: ({ pcm, meta }: { pcm: Int16Array; meta: RecordMeta }) => datasetsApi.record(ds, pcm, meta),
    onSuccess: invalidate,
  })
}
