import { useNavigate } from "react-router"

import { DeleteRecordingButton } from "@/components/logs/delete-recording-button"
import { StatusPill, type Tone } from "@/components/queue/status-pill"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { formatClock, formatDateTime, timeAgo } from "@/lib/format"
import type { Recording, RecordingState } from "@/lib/logs/types"

const STATE: Record<RecordingState, { label: string; tone: Tone }> = {
  done: { label: "완료", tone: "done" },
  live: { label: "녹음 중", tone: "running" },
  unfinished: { label: "중단됨", tone: "failed" },
}

export const formatBytes = (bytes: number) =>
  bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`

export function RecordingStatePill({ state }: { state: RecordingState }) {
  return (
    <StatusPill tone={STATE[state].tone} dot={state === "live"}>
      {STATE[state].label}
    </StatusPill>
  )
}

export function RecordingTable({ recordings }: { recordings: Recording[] }) {
  const navigate = useNavigate()
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>시작</TableHead>
          <TableHead>참가자</TableHead>
          <TableHead>파이프라인</TableHead>
          <TableHead className="text-right">길이</TableHead>
          <TableHead className="text-right">크기</TableHead>
          <TableHead>상태</TableHead>
          <TableHead />
        </TableRow>
      </TableHeader>
      <TableBody>
        {recordings.map((recording) => (
          <TableRow
            key={recording.id}
            className="cursor-pointer"
            onClick={() => navigate(`/logs?${new URLSearchParams({ id: recording.id })}`)}
          >
            <TableCell title={recording.started_at ?? ""}>
              {formatDateTime(recording.started_at)}
              <span className="ml-2 text-xs text-muted-foreground">{timeAgo(recording.started_at)}</span>
            </TableCell>
            <TableCell className="font-mono text-xs">{recording.participant_id}</TableCell>
            <TableCell className="font-mono text-xs">{recording.pipeline}</TableCell>
            <TableCell className="text-right tabular-nums">{formatClock(recording.audio_seconds)}</TableCell>
            <TableCell className="text-right text-muted-foreground tabular-nums">
              {formatBytes(recording.bytes)}
            </TableCell>
            <TableCell>
              <RecordingStatePill state={recording.state} />
            </TableCell>
            <TableCell className="w-10" onClick={(ev) => ev.stopPropagation()}>
              <DeleteRecordingButton recording={recording} />
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}
