import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import { api, type ConfigKind, type ConfigMetaEdit, type Configs, type NewRun } from "@/lib/api"

export const REFRESH_MS = 30_000

export const queryKeys = {
  machines: ["machines"] as const,
  configs: ["configs"] as const,
  branches: ["branches"] as const,
  runs: ["runs"] as const,
}

export function useReload() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: api.reload,
    onSuccess: () => queryClient.invalidateQueries(),
  })
}

export function useRuns() {
  return useQuery({ queryKey: queryKeys.runs, queryFn: api.runs })
}

export function useMachines() {
  return useQuery({
    queryKey: queryKeys.machines,
    queryFn: api.machines,
    meta: { autoRefresh: true },
  })
}

export function useConfigs(enabled = true) {
  return useQuery({ queryKey: queryKeys.configs, queryFn: api.configs, enabled })
}

export function useBranches(enabled = true) {
  return useQuery({ queryKey: queryKeys.branches, queryFn: api.branches, enabled })
}

export function useAddRun(host: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ run, configs }: { run: NewRun; configs: Configs }) =>
      api.addRun(host, run, configs),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.machines }),
  })
}

export function useCancelJob(host: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (jobId: string) => api.cancelJob(host, jobId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.machines }),
  })
}

export function useRunNow(host: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (jobId: string) => api.runNow(host, jobId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.machines }),
  })
}

export function useRemoveMachine() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (host: string) => api.removeMachine(host),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.machines }),
  })
}

export function useCheckConfig(kind: ConfigKind, name: string, yaml: string) {
  return useQuery({
    queryKey: ["config-check", kind, name, yaml],
    queryFn: () => api.checkConfig(kind, name, yaml),
    enabled: name !== "",
    staleTime: Infinity,
  })
}

export function useCreateConfig(kind: ConfigKind) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ name, yaml }: { name: string; yaml: string }) => api.createConfig(kind, name, yaml),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: queryKeys.configs }),
  })
}

export function useEditConfigMeta(kind: ConfigKind) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ name, edit }: { name: string; edit: ConfigMetaEdit }) => api.editConfigMeta(kind, name, edit),
    onSettled: () => queryClient.invalidateQueries({ queryKey: queryKeys.configs }),
  })
}

export function useSaveNotes(host: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ notes, etag }: { notes: string; etag: string | null }) => api.saveNotes(host, notes, etag),
    onSettled: () => queryClient.invalidateQueries({ queryKey: queryKeys.machines }),
  })
}
