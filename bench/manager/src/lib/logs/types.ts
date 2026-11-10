import type { ReplayItem } from "@/lib/replay/types"

export type RecordingState = "done" | "live" | "unfinished"

export type Recording = {
  id: string
  participant_id: string
  pipeline: string
  started_at: string | null
  ended_at: string | null
  audio_seconds: number
  bytes: number
  state: RecordingState
}

export type RecordingsReply = { dir: string; recordings: Recording[] }

export type RecordingReplay = {
  recording: Recording
  server_config: Record<string, unknown>
  item: ReplayItem
}
