import type { AlignedWord, ItemData, Turn } from "@/lib/replay/types"

// A turn is flagged quiet when it is far below full scale, or far below the item's
// loudest turn -- either way a VAD can miss it while the ASR still hears it.
const QUIET_ABS_DBFS = -40
const QUIET_REL_DB = 20

export type QuietTurn = Turn & { words: (AlignedWord & { i: number })[]; quiet: string }

export function quietTurns(data: ItemData | undefined): QuietTurn[] {
  if (!data) return []
  const loudest = Math.max(...data.turns.map((t) => t.peak_dbfs ?? -Infinity))
  return data.turns.map((turn) => {
    const words = data.words
      .map((w, i) => ({ ...w, i }))
      .filter((w) => w.start >= turn.start - 0.05 && w.start < turn.end + 0.05)
    const why = []
    if (turn.peak_dbfs != null && turn.peak_dbfs < QUIET_ABS_DBFS) why.push(`peak is below ${QUIET_ABS_DBFS} dBFS`)
    if (turn.peak_dbfs != null && loudest - turn.peak_dbfs >= QUIET_REL_DB)
      why.push(`${(loudest - turn.peak_dbfs).toFixed(0)} dB below the loudest turn`)
    return { ...turn, words, quiet: why.join(", ") }
  })
}
