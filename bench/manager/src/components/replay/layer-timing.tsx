import { useId, useMemo, type ReactNode } from "react"

import { HelpButton, Legend } from "@/components/replay/player-card"
import { useFollow, type Playback } from "@/hooks/use-playback"
import { useWidth } from "@/hooks/use-width"
import { LANE_COLORS, reasonColor } from "@/lib/replay/format"
import { laneOf, laneOrder, timingModel, type PlayedItem, type TimingModel } from "@/lib/replay/model"

const AUDIO_COLOR = "var(--muted-foreground)"
const HEARD_SHADE = [0.45, 0.35]
const SILENT_SHADE = [0.15, 0.08]
const TL = { padL: 64, padR: 18, padT: 12, lane: 9, bar: 7, label: 14, gap: 8, axis: 24 }
const TICKS = [0.25, 0.5, 1, 2, 5, 10, 30, 60]

function Plot({ model, lanes, width, hatch }: { model: TimingModel; lanes: string[]; width: number; hatch: string }) {
  const plot = width - TL.padL - TL.padR
  const rowH = TL.label + (lanes.length + 1) * TL.lane + TL.gap
  const axisY = TL.padT + model.rows.length * rowH
  const x = (t: number) => TL.padL + plot * Math.min(1, Math.max(0, t / model.end))
  const laneColor = (lane: string) => LANE_COLORS[Math.max(0, lanes.indexOf(lane)) % LANE_COLORS.length]
  const step = TICKS.find((s) => model.end / s <= 12) ?? TICKS.at(-1)!

  const grid: ReactNode[] = []
  for (let t = 0; t <= model.end + 1e-9; t += step) {
    grid.push(
      <line key={`g${t}`} x1={x(t)} y1={TL.padT} x2={x(t)} y2={axisY} className="stroke-border" />,
      <text key={`a${t}`} x={x(t)} y={axisY + 14} textAnchor="middle" className="fill-muted-foreground text-[10px] tabular-nums">
        {+t.toFixed(2)}
      </text>,
    )
    const half = t + step / 2
    if (half < model.end)
      grid.push(<line key={`h${t}`} x1={x(half)} y1={TL.padT} x2={x(half)} y2={axisY} className="stroke-border opacity-60" />)
  }

  return (
    <>
      <defs>
        <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="5" height="5" fill={AUDIO_COLOR} opacity="0.14" />
          <line x1="0" y1="0" x2="0" y2="5" stroke={AUDIO_COLOR} strokeWidth="1.6" opacity="0.5" />
        </pattern>
      </defs>
      {grid}
      {model.rows.map((row, i) => {
        const y0 = TL.padT + i * rowH
        const laneY = (lane: number) => y0 + TL.label + lane * TL.lane
        const commit = row.commit
        const reason = commit?.commit_reason ?? ""
        const parts: ReactNode[] = []
        for (const [k, chunk] of row.chunks.entries()) {
          const heard = chunk.audio == null ? "" : ` · 오디오 ${chunk.audio.toFixed(2)}s`
          parts.push(
            <rect
              key={`c${k}`}
              x={x(chunk.from)}
              y={laneY(0)}
              width={Math.max(0.6, x(chunk.t) - x(chunk.from))}
              height={TL.bar}
              fill={AUDIO_COLOR}
              opacity={(chunk.silence ? SILENT_SHADE : HEARD_SHADE)[chunk.read % 2]}
            >
              <title>{`chunk${heard}${chunk.silence ? " (끝 침묵)" : ""}`}</title>
            </rect>,
          )
        }
        for (const [k, v] of row.vad.entries()) {
          // `ended_at` is when VAD reported the end, after waiting out the silence.
          const spoke = (v.ended_at ?? v.audio ?? 0) - (v.silence_waited_out_sec ?? 0)
          const from = model.wallAt(spoke)
          if (v.t! - from <= 0.005) continue
          parts.push(
            <rect key={`v${k}`} x={x(from)} y={laneY(0)} width={x(v.t!) - x(from)} height={TL.bar} fill={`url(#${hatch})`}>
              <title>{`침묵 대기: 말은 오디오 ${spoke.toFixed(2)}s에 끝났고, VAD가 ${v.t!.toFixed(2)}s에 알렸어요`}</title>
            </rect>,
          )
        }
        for (const [k, bar] of row.spans.entries()) {
          parts.push(
            <rect
              key={`s${k}`}
              x={x(bar.t)}
              y={laneY(1 + lanes.indexOf(laneOf(bar)))}
              width={Math.max(0.8, x(bar.t + bar.dur) - x(bar.t))}
              height={TL.bar}
              rx="1.5"
              fill={laneColor(laneOf(bar))}
            >
              <title>{`${laneOf(bar)} ${bar.dur.toFixed(3)}s${bar.reason ? ` (${bar.reason})` : ""} · ${bar.t.toFixed(2)}s 시작`}</title>
            </rect>,
          )
        }
        if (commit) {
          const top = laneY(0) - 1
          const bottom = laneY(lanes.length) + TL.bar + 1
          const heardX = x(model.wallAt(commit.audio))
          const during = row.spans.find((s) => s.t <= commit.t && commit.t <= s.t + s.dur)
          const markY = during ? laneY(1 + lanes.indexOf(laneOf(during))) + TL.bar / 2 : top + 4
          const cx = x(commit.t)
          parts.push(
            <line key="heard" x1={heardX} y1={top} x2={heardX} y2={bottom} className="stroke-foreground" opacity="0.6">
              <title>{`commit이 들은 오디오: ${commit.audio.toFixed(2)}s`}</title>
            </line>,
            <path key="mark" d={`M ${cx} ${markY - 4} l 4 4 l -4 4 l -4 -4 z`} className="fill-card stroke-foreground" strokeWidth="1.2">
              <title>{`commit ${commit.t.toFixed(2)}s (${reason})${commit.original ? ` -- ${commit.original}` : ""}`}</title>
            </path>,
            <text key="reason" x={x(row.start) + 3} y={y0 + 10} fill={reasonColor(reason)} className="font-mono text-[10px]">
              {reason}
            </text>,
          )
          if (commit.committed_elapsed_sec != null && commit.decision_audio_sec != null) {
            const fsl = commit.committed_elapsed_sec - commit.decision_audio_sec
            const anchor = x(Math.max(commit.t, ...row.spans.map((s) => s.t + s.dur))) + 6
            const past = anchor > TL.padL + plot - 62
            parts.push(
              <text
                key="fsl"
                x={past ? TL.padL + plot : anchor}
                y={y0 + 10}
                textAnchor={past ? "end" : "start"}
                className="fill-foreground text-[10px] font-semibold tabular-nums"
              >
                FSL {fsl.toFixed(2)}s
              </text>,
            )
          }
        }
        return (
          <g key={i}>
            <rect x={TL.padL} y={y0} width={plot} height={rowH - TL.gap + 2} rx="3" className={i % 2 ? "fill-transparent" : "fill-muted"} />
            <text x={TL.padL - 8} y={laneY(1) + 6} textAnchor="end" className="fill-muted-foreground text-[11px] tabular-nums">
              {commit ? `commit ${i + 1}` : "나머지"}
            </text>
            {parts}
          </g>
        )
      })}
    </>
  )
}

export function LayerTiming({ item, playback }: { item: PlayedItem | null; playback: Playback }) {
  const [body, measured] = useWidth()
  const hatch = useId().replace(/:/g, "")
  const model = useMemo(() => (item ? timingModel(item) : null), [item])
  const lanes = useMemo(() => (item ? laneOrder(item) : []), [item])
  const { now, playing, jump } = playback

  const width = Math.max(560, measured - 2)
  const rowH = TL.label + (lanes.length + 1) * TL.lane + TL.gap
  const height = model ? TL.padT + model.rows.length * rowH + TL.axis : 0
  const plotted = useMemo(
    () => (model && model.rows.length ? <Plot model={model} lanes={lanes} width={width} hatch={hatch} /> : null),
    [model, lanes, width, hatch],
  )

  const headX = model ? TL.padL + (width - TL.padL - TL.padR) * Math.min(1, Math.max(0, (now + model.feedLag) / model.end)) : 0
  // The row being heard: the first commit that had not yet been decided at `now`.
  let heardRow = model ? model.rows.findIndex((row) => row.commit && now <= row.commit.audio + 1e-6) : -1
  if (model && heardRow < 0) heardRow = model.rows.length - 1
  useFollow(heardRow, playing, jump, () => {
    const box = body.current
    if (!box || heardRow < 0) return
    const top = TL.padT + heardRow * rowH
    if (top >= box.scrollTop && top + rowH <= box.scrollTop + box.clientHeight) return
    box.scrollTop = Math.max(0, top - (box.clientHeight - rowH) * 0.4)
  })

  const commits = item?.events.filter((e) => e.type === "translated").length ?? 0
  const legend = [
    { label: "audio in", color: AUDIO_COLOR, className: "opacity-45" },
    ...lanes.map((lane, i) => ({ label: lane, color: LANE_COLORS[i % LANE_COLORS.length] })),
  ]

  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <p className="text-sm text-muted-foreground tabular-nums">
          {item ? `${item.id} · 오디오 ${item.span.toFixed(1)}s · commit ${commits}개 · ` : ""}commit 하나가 한 줄이에요. 가로축은 첫 chunk부터 흐른 실제 시간이에요.
        </p>
        <span className="flex-1" />
        {item && (
          <HelpButton label="타이밍 차트 범례">
            <Legend entries={legend} />
            <span className="text-xs text-muted-foreground">◆ commit을 결정한 시점</span>
            <Legend entries={[{ label: "commit이 들은 오디오 끝", color: "var(--foreground)", className: "w-0.5" }]} />
            <Legend entries={[{ label: "침묵을 기다린 구간", color: AUDIO_COLOR, className: "opacity-45" }]} />
          </HelpButton>
        )}
      </div>
      <div ref={body} className="max-h-[440px] overflow-auto pt-1 pb-2">
        {!item ? null : plotted ? (
          <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="block">
            {plotted}
            <line
              x1={headX}
              x2={headX}
              y1={TL.padT - 4}
              y2={height - TL.axis}
              className="stroke-foreground"
              strokeWidth="1.5"
              opacity="0.75"
            />
          </svg>
        ) : (
          <p className="pb-4 text-sm text-muted-foreground">
            이 항목에는 타이밍이 없어요. 부품별 타이밍 기록 이전의 실행이거나, 한 번도 commit하지 않은 항목이에요.
          </p>
        )}
      </div>
    </div>
  )
}
