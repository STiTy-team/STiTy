import type { ConfigKind } from "@/lib/api"
import type { RunOverview } from "@/lib/compare/types"

export const KIND_LABEL: Record<ConfigKind, string> = { pipeline: "파이프라인", dataset: "데이터셋" }

export const NAME_EXAMPLE: Record<ConfigKind, string> = {
  pipeline: "asr.qwen-seg-ko+mt.qwen3.5-4b",
  dataset: "fleurs_ko-en_top50",
}

export const TEMPLATES: Record<ConfigKind, string> = {
  pipeline: `meta:
  description: ""
  tags: []
pipeline:
  name: cascade
  transcription: mock
  vad: mock
  correction: mock
  translation: mock
commit: seg
gpu_memory_utilization: 0.5
`,
  dataset: `meta:
  description: ""
  tags: []
dataset:
  name: fleurs/ko_kr
  limit: 10
target: en
`,
}

/** The YAML with `meta.version` set, adding the key (or the whole `meta:` block) when it is missing. */
export function withVersion(text: string, version: number) {
  const lines = text.split("\n")
  const metaAt = lines.findIndex((line) => /^meta:\s*$/.test(line))
  if (metaAt < 0) return `meta:\n  version: ${version}\n${text}`
  for (let i = metaAt + 1; i < lines.length && /^(\s+\S.*|\s*)$/.test(lines[i]); i++) {
    if (/^\s+version:/.test(lines[i])) {
      lines[i] = lines[i].replace(/version:.*/, `version: ${version}`)
      return lines.join("\n")
    }
  }
  lines.splice(metaAt + 1, 0, `  version: ${version}`)
  return lines.join("\n")
}

export const configRef = (name: string, version: number) => (version === 1 ? name : `${name}@v${version}`)

/** Where the Compare page shows this config's runs: its dataset, or for a pipeline the dataset it last ran on. */
export function compareLink(kind: ConfigKind, ref: string, runs: RunOverview[]) {
  if (kind === "dataset") return runs.some((r) => r.dataset_key === ref) ? `/compare?${new URLSearchParams({ dataset: ref })}` : null
  const latest = runs
    .filter((r) => r.pipeline === ref)
    .sort((a, b) => b.stamp.localeCompare(a.stamp))[0]
  return latest ? `/compare?${new URLSearchParams({ dataset: latest.dataset_key, base: latest.name })}` : null
}
