import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import { del, get } from "@/lib/api"
import type { RecordingReplay, RecordingsReply } from "@/lib/logs/types"

const byId = (id: string) => new URLSearchParams({ id }).toString()

export const logKeys = {
  list: ["logs"] as const,
  replay: (id: string) => ["logs", "replay", id] as const,
}

export const logUrls = {
  list: "/api/logs",
  replay: (id: string) => `/api/logs/replay?${byId(id)}`,
  audio: (id: string) => `/api/logs/audio?${byId(id)}`,
  remove: (id: string) => `/api/logs?${byId(id)}`,
}

export function useRecordings() {
  return useQuery({
    queryKey: logKeys.list,
    queryFn: () => get<RecordingsReply>(logUrls.list),
    meta: { autoRefresh: true },
  })
}

export function useRecordingReplay(id: string) {
  return useQuery({ queryKey: logKeys.replay(id), queryFn: () => get<RecordingReplay>(logUrls.replay(id)) })
}

export function useDeleteRecording() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => del<{ deleted: string }>(logUrls.remove(id)),
    onSuccess: (_, id) => {
      queryClient.removeQueries({ queryKey: logKeys.replay(id) })
      return queryClient.invalidateQueries({ queryKey: logKeys.list })
    },
  })
}
