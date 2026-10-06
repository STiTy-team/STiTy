import type { ReactNode } from "react"

import { useChartTip } from "@/components/compare/chart-tip"
import { useWidth } from "@/hooks/use-width"
import { axisText, BETTER_TEXT, clip, compareScores, displayScale, formatScore, niceTicks, SCORES, tickText } from "@/lib/compare/scores"
import type { Distributions, RunOverview } from "@/lib/compare/types"

const PAD_L = 64
const PAD_R = 16
const TOP = 26
const PLOT_H = 320

function statsText(run: RunOverview, key: string, s: Distributions[string]) {
  const rows: [string, number][] = [
    ["max", s.max],
    ["q3", s.q3],
    ["median", s.median],
    ["mean", s.mean],
    ["q1", s.q1],
    ["min", s.min],
  ]
  return [run.pipeline, `n       ${s.n}`, ...rows.map(([name, v]) => `${name.padEnd(8)}${formatScore(key, v)}`)].join("\n")
}

/** One column per run, best median on the left: box, whiskers, mean diamond and optional item dots. */
export function DistributionChart({
  runs,
  scoreKey,
  dists,
  colors,
  points,
}: {
  runs: RunOverview[]
  scoreKey: string
  dists: Map<string, Distributions>
  colors: Map<string, string>
  points: boolean
}) {
  const [box, measured] = useWidth()
  const [tip, bindTip] = useChartTip()
  const spec = SCORES[scoreKey]
  const scale = displayScale(scoreKey)

  const cols = runs
    .map((run) => ({ run, s: dists.get(run.name)?.[spec.item!] }))
    .filter((c): c is { run: RunOverview; s: Distributions[string] } => c.s != null)
    .sort((a, b) => compareScores(scoreKey, a.s.median, b.s.median) || a.s.median - b.s.median)

  const lo = Math.min(...cols.map((c) => c.s.min)) * scale
  const hi = Math.max(...cols.map((c) => c.s.max)) * scale
  const ticks = niceTicks(lo, hi, 6)
  const avail = Math.max(560, measured || 900) - PAD_L - PAD_R
  const colW = Math.max(64, Math.min(220, avail / Math.max(cols.length, 1)))
  const tilt = colW < 118
  const labelH = tilt ? 96 : 48
  const W = Math.max(PAD_L + PAD_R + colW * cols.length, PAD_L + PAD_R + avail)
  const H = TOP + PLOT_H + labelH
  const y = (v: number) => TOP + PLOT_H * (1 - (v - ticks[0]) / (ticks[ticks.length - 1] - ticks[0]))
  const dir = spec.better ? ` · ${BETTER_TEXT[spec.better]}` : ""

  const marks: ReactNode[] = cols.map(({ run, s }, i) => {
    const cx = PAD_L + colW * (i + 0.5)
    const color = colors.get(run.name)
    const half = Math.min(26, colW * 0.28)
    const v = (k: "min" | "q1" | "median" | "mean" | "q3" | "max") => y(s[k] * scale)
    const ly = TOP + PLOT_H + 16
    return (
      <g key={run.name}>
        {points &&
          s.points.map((p, j) => (
            <circle
              key={j}
              cx={cx + (((j * 0.6180339887) % 1) - 0.5) * colW * 0.62}
              cy={y(p * scale)}
              r={1.8}
              fill={color}
              opacity={0.28}
            />
          ))}
        <line x1={cx} x2={cx} y1={v("max")} y2={v("q3")} className="stroke-muted-foreground" strokeWidth={1.2} />
        <line x1={cx} x2={cx} y1={v("q1")} y2={v("min")} className="stroke-muted-foreground" strokeWidth={1.2} />
        <line x1={cx - half / 2} x2={cx + half / 2} y1={v("max")} y2={v("max")} className="stroke-muted-foreground" strokeWidth={1.2} />
        <line x1={cx - half / 2} x2={cx + half / 2} y1={v("min")} y2={v("min")} className="stroke-muted-foreground" strokeWidth={1.2} />
        <rect
          x={cx - half}
          y={v("q3")}
          width={half * 2}
          height={Math.max(1, v("q1") - v("q3"))}
          rx={2}
          fill={color}
          fillOpacity={0.28}
          stroke={color}
          strokeWidth={1.4}
        />
        <line x1={cx - half - 2} x2={cx + half + 2} y1={v("median")} y2={v("median")} className="stroke-foreground" strokeWidth={2.4} />
        <path d={`M ${cx} ${v("mean") - 5} l 5 5 l -5 5 l -5 -5 z`} className="fill-card stroke-foreground" strokeWidth={1.3} />
        <text x={cx} y={v("max") - 7} textAnchor="middle" className="fill-muted-foreground text-[11px] font-medium tabular-nums">
          {formatScore(scoreKey, s.median)}
        </text>
        {tilt ? (
          <text x={cx} y={ly} textAnchor="end" transform={`rotate(-35 ${cx} ${ly})`} className="fill-foreground font-mono text-xs">
            {clip(run.pipeline, 26)}
          </text>
        ) : (
          <>
            <text x={cx} y={ly} textAnchor="middle" className="fill-foreground font-mono text-xs">
              {clip(run.pipeline, Math.floor(colW / 7))}
            </text>
            <text x={cx} y={ly + 16} textAnchor="middle" className="fill-muted-foreground text-[11px] tabular-nums">
              n={s.n}
            </text>
          </>
        )}
        <rect
          x={cx - colW / 2}
          y={TOP}
          width={colW}
          height={PLOT_H + labelH}
          fill="transparent"
          {...bindTip(statsText(run, scoreKey, s))}
        />
      </g>
    )
  })

  return (
    <div ref={box} className="overflow-x-auto">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block max-w-none">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={PAD_L} x2={W - PAD_R} y1={y(t)} y2={y(t)} className="stroke-border" />
            <text x={PAD_L - 8} y={y(t) + 3} textAnchor="end" className="fill-muted-foreground text-[11px] tabular-nums">
              {tickText(scoreKey, t)}
            </text>
          </g>
        ))}
        <text
          x={14}
          y={TOP + PLOT_H / 2}
          textAnchor="middle"
          transform={`rotate(-90 14 ${TOP + PLOT_H / 2})`}
          className="fill-muted-foreground text-[11px]"
        >
          {axisText(scoreKey)}
          {dir}
        </text>
        {marks}
      </svg>
      {tip}
    </div>
  )
}
