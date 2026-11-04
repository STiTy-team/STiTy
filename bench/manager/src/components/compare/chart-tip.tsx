import { useCallback, useState, type MouseEvent } from "react"

type Tip = { x: number; y: number; text: string } | null

/** A hover card that follows the pointer; `bind(text)` goes on any SVG mark. */
export function useChartTip() {
  const [tip, setTip] = useState<Tip>(null)
  const bind = useCallback(
    (text: string) => ({
      onMouseMove: (ev: MouseEvent) => setTip({ x: ev.clientX, y: ev.clientY, text }),
      onMouseLeave: () => setTip(null),
    }),
    [],
  )
  const element = tip && (
    <div
      className="pointer-events-none fixed z-50 rounded-lg border bg-popover px-3 py-2 font-mono text-xs whitespace-pre text-popover-foreground shadow-md tabular-nums"
      style={{ left: Math.min(innerWidth - 220, tip.x + 14), top: Math.min(innerHeight - 180, tip.y + 14) }}
    >
      {tip.text}
    </div>
  )
  return [element, bind] as const
}
