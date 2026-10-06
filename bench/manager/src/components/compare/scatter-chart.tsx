import { useChartTip } from "@/components/compare/chart-tip"
import { useWidth } from "@/hooks/use-width"
import { axisText, BETTER_TEXT, clip, compareScores, displayScale, formatScore, metric, niceTicks, SCORES, tickText } from "@/lib/compare/scores"
import type { RunOverview } from "@/lib/compare/types"

const H = 380
const L = 64
const R = 24
const T = 16
const B = 46

type Point = { run: RunOverview; x: number; y: number }

const spread = (lo: number, hi: number) => (hi > lo ? (hi - lo) * 0.12 : Math.abs(hi || 1) * 0.1)

function paddedTicks(values: number[], count: number) {
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  return niceTicks(lo - spread(lo, hi), hi + spread(lo, hi), count)
}

/** Accuracy against latency from the run summaries; better is always up and to the right. */
export function ScatterChart({ runs, xKey, yKey, colors }: { runs: RunOverview[]; xKey: string; yKey: string; colors: Map<string, string> }) {
  const [box, measured] = useWidth()
  const [tip, bindTip] = useChartTip()

  const pts: Point[] = runs.flatMap((run) => {
    const x = metric(run.summary_metrics, xKey)
    const y = metric(run.summary_metrics, yKey)
    return x != null && y != null ? [{ run, x, y }] : []
  })
  if (!pts.length)
    return (
      <div ref={box}>
        <p className="py-6 text-sm text-muted-foreground">No run has both scores.</p>
      </div>
    )

  const sx = displayScale(xKey)
  const sy = displayScale(yKey)
  const xt = paddedTicks(
    pts.map((p) => p.x * sx),
    6,
  )
  const yt = paddedTicks(
    pts.map((p) => p.y * sy),
    5,
  )
  const W = Math.max(560, measured || 900)
  const flipX = SCORES[xKey].better === "lower"
  const flipY = SCORES[yKey].better === "lower"
  const fx = (v: number) => {
    const f = (v - xt[0]) / (xt[xt.length - 1] - xt[0])
    return L + (W - L - R) * (flipX ? 1 - f : f)
  }
  const fy = (v: number) => {
    const f = (v - yt[0]) / (yt[yt.length - 1] - yt[0])
    return T + (H - T - B) * (flipY ? f : 1 - f)
  }
  const px = (p: Point) => fx(p.x * sx)
  const py = (p: Point) => fy(p.y * sy)

  const beats = (a: Point, b: Point) => {
    const cx = compareScores(xKey, a.x, b.x)
    const cy = compareScores(yKey, a.y, b.y)
    return cx <= 0 && cy <= 0 && (cx < 0 || cy < 0)
  }
  const front = pts.filter((p) => !pts.some((q) => q !== p && beats(q, p))).sort((a, b) => px(a) - px(b))
  const [fa, fb] = front.slice(-2)
  const midY = (T + H - B) / 2

  return (
    <div ref={box} className="overflow-x-auto">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block max-w-none">
        {xt.map((t) => (
          <g key={`x${t}`}>
            <line x1={fx(t)} x2={fx(t)} y1={T} y2={H - B} className="stroke-border" />
            <text x={fx(t)} y={H - B + 16} textAnchor="middle" className="fill-muted-foreground text-[11px] tabular-nums">
              {tickText(xKey, t)}
            </text>
          </g>
        ))}
        {yt.map((t) => (
          <g key={`y${t}`}>
            <line x1={L} x2={W - R} y1={fy(t)} y2={fy(t)} className="stroke-border" />
            <text x={L - 8} y={fy(t) + 3} textAnchor="end" className="fill-muted-foreground text-[11px] tabular-nums">
              {tickText(yKey, t)}
            </text>
          </g>
        ))}
        <text x={(L + W - R) / 2} y={H - 8} textAnchor="middle" className="fill-muted-foreground text-[11px]">
          {axisText(xKey)} · {BETTER_TEXT[SCORES[xKey].better!]} · 오른쪽이 좋음
        </text>
        <text x={14} y={midY} textAnchor="middle" transform={`rotate(-90 14 ${midY})`} className="fill-muted-foreground text-[11px]">
          {axisText(yKey)} · {BETTER_TEXT[SCORES[yKey].better!]} · 위쪽이 좋음
        </text>
        {front.length > 1 && (
          <>
            <polyline
              points={front.map((p) => `${px(p)},${py(p)}`).join(" ")}
              fill="none"
              strokeWidth={1.5}
              strokeDasharray="5 4"
              className="stroke-muted-foreground"
            />
            <text x={(px(fa) + px(fb)) / 2} y={(py(fa) + py(fb)) / 2 - 8} textAnchor="middle" className="fill-muted-foreground text-[11px]">
              파레토 경계
            </text>
          </>
        )}
        {pts.map((p) => (
          <g key={p.run.name}>
            <circle
              cx={px(p)}
              cy={py(p)}
              r={6}
              fill={colors.get(p.run.name)}
              strokeWidth={1.5}
              className="stroke-card"
              {...bindTip(
                [
                  p.run.pipeline,
                  `${SCORES[yKey].label}  ${formatScore(yKey, p.y)}`,
                  `${SCORES[xKey].label}  ${formatScore(xKey, p.x)}`,
                  front.includes(p) ? "파레토 경계 위" : "",
                ]
                  .filter(Boolean)
                  .join("\n"),
              )}
            />
            <text x={px(p) + 9} y={py(p) + 4} className="fill-muted-foreground font-mono text-[11.5px]">
              {clip(p.run.pipeline, 34)}
            </text>
          </g>
        ))}
      </svg>
      {tip}
    </div>
  )
}
