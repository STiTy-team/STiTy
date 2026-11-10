import { useMemo, useRef, useState } from "react"

import { DeleteRecordingButton } from "@/components/logs/delete-recording-button"
import { formatBytes, RecordingStatePill } from "@/components/logs/recording-table"
import { ColumnHead } from "@/components/replay/column-head"
import { EventStream } from "@/components/replay/event-stream"
import { PipelineConfig } from "@/components/replay/kv"
import { Phone } from "@/components/replay/phone"
import { PlayerCard } from "@/components/replay/player-card"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useAudioSync, usePlayback, useSpaceToToggle } from "@/hooks/use-playback"
import { formatClock, formatDateTime } from "@/lib/format"
import { logUrls } from "@/lib/logs/queries"
import type { RecordingReplay as Replay } from "@/lib/logs/types"
import { prepareItem } from "@/lib/replay/model"

export function RecordingReplay({ data, onDeleted }: { data: Replay; onDeleted: () => void }) {
  const { recording, item } = data
  const [speed, setSpeed] = useState(1)
  const [soundOn, setSoundOn] = useState(true)
  const played = useMemo(() => prepareItem(item), [item])
  const playback = usePlayback(played.span, speed)
  useSpaceToToggle(playback.toggle)

  const audio = useRef<HTMLAudioElement>(null)
  useAudioSync(audio, {
    url: soundOn && item.audio_url ? logUrls.audio(recording.id) : null,
    now: playback.now,
    playing: playback.playing,
    speed,
  })

  return (
    <div className="flex flex-col gap-6">
      <audio ref={audio} preload="auto" />
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
        <span className="font-mono break-all">{recording.id}</span>
        <RecordingStatePill state={recording.state} />
        <span className="text-muted-foreground">
          {formatDateTime(recording.started_at)} · {formatClock(recording.audio_seconds)} ·{" "}
          {formatBytes(recording.bytes)}
        </span>
        <span className="flex-1" />
        <DeleteRecordingButton recording={recording} onDeleted={onDeleted} />
      </div>
      <PlayerCard
        item={played}
        turns={[]}
        playback={playback}
        speed={speed}
        onSpeed={setSpeed}
        soundOn={soundOn}
        onSound={setSoundOn}
      />
      <div className="grid items-start gap-6 xl:grid-cols-[384px_minmax(0,1fr)]">
        <section>
          <ColumnHead
            title="휴대폰 화면"
            description={
              <>
                서버가 보낸 <code>final</code> 중 <code>{item.target_lang || "?"}</code> 번역
              </>
            }
          />
          <Phone
            item={played}
            now={playback.now}
            playing={playback.playing}
            srcLang={item.src_lang}
            targetLang={item.target_lang}
          />
        </section>
        <section>
          <ColumnHead title="이벤트 흐름" description="주고받은 WebSocket 메시지 전부, 오디오 순서" />
          <EventStream events={played.played} playback={playback} />
        </section>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>서버 설정</CardTitle>
        </CardHeader>
        <CardContent>
          <PipelineConfig config={data.server_config} />
        </CardContent>
      </Card>
    </div>
  )
}
