export type ReplayEvent = {
  type: string
  tag?: string
  item?: string
  t?: number | null
  audio?: number | null
  dur?: number
  commit_reason?: string
  original?: string
  translation?: string
  language?: string
  text?: string
  seq?: number
  reason?: string
  silence?: boolean
  ended_at?: number
  silence_waited_out_sec?: number
  committed_elapsed_sec?: number
  decision_audio_sec?: number
  translated_elapsed_sec?: number
  [key: string]: unknown
}

export type ItemMetrics = Partial<Record<string, number>>

export type ReplayItem = {
  id: string
  events: ReplayEvent[]
  duration: number
  src_lang: string
  target_lang: string
  wer?: number
  metrics?: ItemMetrics
  reference: string
  reference_translations: Record<string, string>
  audio_url?: string
}

export type ConfigMeta = {
  ref: string
  version: number
  hash: string
  description: string
  tags: string[]
}

export type RunMeta = { dataset?: ConfigMeta; pipeline?: ConfigMeta }

type RunConfig = {
  meta: RunMeta
  dataset_sha: string
  pipeline_config: Record<string, unknown>
  config_source: string
  status: string
  summary_metrics: Record<string, unknown>
}

export type RunInfo = RunConfig & {
  name: string
  stamp: string
  dir: string
  n_total: number
  top_k: number
}

export type CatalogRun = RunConfig & {
  name: string
  dataset: string
  pipeline: string
  data_key: string
  item_ids: string[]
}

export type ItemRow = { id: string; wer: number | null }

export type ReplayPayload = {
  run: RunInfo
  items: ReplayItem[]
  worst: ItemRow[]
  all_items: ItemRow[]
  runs: CatalogRun[]
}

export type Turn = {
  id: string
  start: number
  end: number
  src_lang: string
  speaker: string
  partial: boolean
  transcript: string
  translations: Record<string, string>
  peak_dbfs?: number
  rms_dbfs?: number
}

export type AlignedWord = { word: string; start: number; end: number }

export type ItemData = {
  id: string
  dataset: { name: string; root: string; manifest_sha256: string; spec: string }
  audio: {
    path: string
    offset: number | null
    duration: number | null
    augment: Record<string, unknown> | null
    format?: string
    sample_rate?: number
    channels?: number
    file_sec?: number
    error?: string
  } | null
  turns: Turn[]
  words: AlignedWord[]
  manifest_rows: { line: number; row: Record<string, unknown> }[]
}
