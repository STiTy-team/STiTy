export type JobState = "queued" | "running" | "stopping" | "cancelling"

export type Attempt = {
  host: string
  started_at: string
  commit: string | null
  outcome: string | null
  ran_sec: number | null
  error: string | null
}

export type Job = {
  id: string
  branch: string
  pipeline: string
  dataset: string
  submitted_at: string
  state: JobState
  run_now: boolean
  attempts: Attempt[]
}

export type FinalOutcome = "done" | "failed" | "cancelled"

export type RunLocation = {
  dataset: string
  pipeline: string
  run_id: string
}

export type Finished = {
  job: Job
  outcome: FinalOutcome
  ended_at: string
  run: RunLocation | null
  error: string | null
}

export type WeeklyBlock = {
  day: number
  start: string
  end: string
}

export type MachineSettings = {
  timezone: string
  blocks: WeeklyBlock[]
  paused: boolean
}

export type GpuInfo = {
  index: number
  name: string
  memory_used_mb: number
  memory_total_mb: number
  util: number | null
}

export type WorkerState = "idle" | "outside_window" | "running" | "backoff" | "paused" | "stopping"

export type Health = {
  worker_commit: string | null
  worker_started_at: string
  updated_at: string
  state: WorkerState
  job: string | null
  window: { open: boolean; ends_at: string | null; next_opens_at: string | null } | null
  backoff: { level: number; until: string } | null
  gpus: GpuInfo[]
  disk_free_gb: number | null
  disk_total_gb: number | null
  last_error: string | null
}

export type Machine = {
  host: string
  connected: boolean
  health: Health | null
  notes: string
  notes_etag: string | null
  settings: MachineSettings
  settings_etag: string | null
  jobs: Job[]
  history: Finished[]
}

export type MachinesReply =
  | { configured: false; message: string }
  | { configured: true; now: string; machines: Machine[] }

export type ConfigFile = {
  yaml: string
  etag: string
  modified_at: string | null
  description: string
  tags: string[]
  version: number
}

export type ConfigMetaEdit = {
  etag: string
  name: string
  meta: { description: string; tags: string[]; version?: number }
}

export type ConfigCheck =
  | { valid: false; error: string }
  | { valid: true; ref: string; taken: null }
  | { valid: true; ref: string; taken: { by: string; at: string }; free_version: number }

export type ConfigKind = "pipeline" | "dataset"

export type Configs = Record<ConfigKind, Record<string, ConfigFile>>

export type Branch = {
  name: string
  sha: string
  subject: string
  author: string
}

export type Branches = {
  current: string | null
  branches: Branch[]
}

export type RunRow = {
  key: string
  dataset: string
  pipeline: string
  run_id: string | null
  host: string | null
  local_name: string | null
  stamp: string
  started_at: string | null
  finished_at: string | null
  status: string
  failure: string | null
  items: number | null
  failed_items: number
  wall_sec: number | null
  commit: string | null
  models: Partial<Record<"transcription" | "correction" | "translation", string>>
  pipeline_description: string
  pipeline_tags: string[]
  dataset_description: string
  dataset_tags: string[]
  metrics: Record<string, number>
}

export type RunsReply = { runs: RunRow[]; shared: boolean }

export type NewRun = {
  branch: string
  pipeline: string
  dataset: string
}

async function readJson(response: Response) {
  return response.json().catch(() => ({ error: `HTTP ${response.status}` }))
}

export async function get<T>(path: string): Promise<T> {
  const response = await fetch(path)
  const body = await readJson(response)
  if (!response.ok) throw new Error(body.error ?? `HTTP ${response.status}`)
  return body
}

/** POSTs a JSON-free request (query params carry the payload) and reads back raw JSON,
 * same convention as {@link get}. Used by the replay- and datasets-style endpoints,
 * which answer with the resource itself rather than the `{ok,data}` envelope `post` expects. */
export async function postFor<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { method: "POST", ...init })
  const body = await readJson(response)
  if (!response.ok) throw new Error(body.error ?? `HTTP ${response.status}`)
  return body
}

export async function del<T>(path: string): Promise<T> {
  const response = await fetch(path, { method: "DELETE" })
  const body = await readJson(response)
  if (!response.ok) throw new Error(body.error ?? `HTTP ${response.status}`)
  return body
}

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function post<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
  const body = await readJson(response)
  if (!response.ok || !body.ok) throw new ApiError(body.error ?? `HTTP ${response.status}`, response.status)
  return body.data
}

export const api = {
  machines: () => get<MachinesReply>("/api/machines"),
  configs: () => get<Configs>("/api/configs"),
  branches: () => get<Branches>("/api/branches"),
  runs: () => get<RunsReply>("/api/runs"),
  reload: () => post("/api/reload", {}),
  addRun: (host: string, run: NewRun, configs: Configs) =>
    post(`/api/machines/${encodeURIComponent(host)}/jobs`, {
      branch: run.branch,
      pipeline: { name: run.pipeline, yaml: configs.pipeline[run.pipeline].yaml },
      dataset: { name: run.dataset, yaml: configs.dataset[run.dataset].yaml },
    }),
  checkConfig: (kind: ConfigKind, name: string, yaml: string) =>
    post<ConfigCheck>("/api/configs/check", { kind, name, yaml }),
  createConfig: (kind: ConfigKind, name: string, yaml: string) =>
    post<{ name: string; etag: string }>(`/api/configs/${kind}/${encodeURIComponent(name)}`, { yaml, etag: null }),
  editConfigMeta: (kind: ConfigKind, name: string, edit: ConfigMetaEdit) =>
    post<{ name: string; etag: string }>(`/api/configs/${kind}/${encodeURIComponent(name)}/meta`, edit),
  saveSettings: (host: string, settings: MachineSettings, etag: string | null) =>
    post<{ settings: MachineSettings; etag: string }>(`/api/machines/${encodeURIComponent(host)}/settings`, {
      settings,
      etag,
    }),
  saveNotes: (host: string, notes: string, etag: string | null) =>
    post<{ notes: string; etag: string }>(`/api/machines/${encodeURIComponent(host)}/notes`, { notes, etag }),
  cancelJob: (host: string, jobId: string) =>
    post(`/api/machines/${encodeURIComponent(host)}/jobs/${encodeURIComponent(jobId)}/cancel`, {}),
  runNow: (host: string, jobId: string) =>
    post<Job>(`/api/machines/${encodeURIComponent(host)}/jobs/${encodeURIComponent(jobId)}/run-now`, {}),
  removeMachine: (host: string) => post<{ removed: string }>(`/api/machines/${encodeURIComponent(host)}/remove`, {}),
  pullRun: (run: RunLocation) => post<{ run: string }>("/api/runs/pull", run),
}
