import type { RunMeta } from "@/lib/replay/types"

export type RunOverview = {
  name: string
  pipeline: string
  dataset_key: string
  dataset_name: string
  corpus: string
  target: string
  limit: number | null
  pick: string
  meta: RunMeta
  dataset_sha: string
  stamp: string
  status: string
  counts: { items?: number; of?: number | null }
  models: Partial<Record<"transcription" | "correction" | "translation", string>>
  summary_metrics: Record<string, unknown>
}

export type Stats = {
  n: number
  min: number
  q1: number
  median: number
  mean: number
  q3: number
  max: number
  points: number[]
}

export type Distributions = Record<string, Stats>
