export type DatasetSummary = {
  name: string
  items: number
  duration: number
  converted: boolean
}

export type DatasetsList = {
  root: string
  datasets: DatasetSummary[]
}

export type Spread = {
  min: number
  p50: number
  p90: number
  max: number
  mean: number
}

export type HistogramBin = {
  lo: number
  hi: number | null
  count: number
}

export type FieldStat = {
  key: string
  count: number
  types: Record<string, number>
}

export type DatasetSpec = {
  name?: string
  languages?: string[]
  primary_metric?: string
  provides?: { transcript?: boolean; translations?: string[] }
  [key: string]: unknown
}

export type DatasetShape = {
  name: string
  path: string
  spec: DatasetSpec
  files: {
    manifest: boolean
    manifest_bytes: number
    manifest_sha256: string | null
    alignment: boolean
    readme: boolean
    convert: boolean
  }
  items: number
  duration: { total: number } & Partial<Spread>
  histogram: HistogramBin[]
  sessions: number
  sessions_contiguous: boolean
  speakers: number
  src_lang: Record<string, number>
  audio_files: number
  audio_missing: number
  offset_items: number
  partial_items: number
  transcript: { filled: number; units: Spread | null }
  translations: Record<string, number>
  fields: FieldStat[]
  alignment: {
    items: number
    stale: number
    orphan: number
    aligners: Record<string, number>
  }
}

export type ItemReference = {
  transcript?: string
  translations?: Record<string, string>
}

export type ItemRow = {
  id: string
  audio?: string
  duration?: number
  group?: string
  speaker?: string
  speakers?: number
  src_lang?: string
  offset?: number
  partial?: boolean
  reference?: ItemReference
  line: number
  has_audio: boolean
  [key: string]: unknown
}

export type ItemsPage = {
  total: number
  matched: number
  offset: number
  items: ItemRow[]
}

export type AudioInfo = {
  file: string
  resolved: string
  bytes: number
  format: string
  subtype: string
  sample_rate: number
  channels: number
  file_duration: number
  problem: string | null
}

export type AlignedWord = {
  word: string
  start: number
  end: number
}

export type Alignment = {
  aligner: string
  words: AlignedWord[]
  stale: boolean
  [key: string]: unknown
}

export type ItemDetail = {
  line: number
  row: ItemRow
  audio: AudioInfo | null
  alignment: Alignment | null
}

export type VerifyResult = {
  ok: boolean
  report: string
}

export type RecordResult =
  | { discarded: true; duration: number }
  | (ItemRow & { discarded?: false })

export type NewDataset = {
  name: string
  path: string
  languages: string[]
}

export type RecordMeta = {
  prefix: string
  speakers: number
  srcLang: string
  speaker: string
}
