import { useNavigate, useSearchParams } from "react-router"

import { Notice, PageBody, PageHeader } from "@/components/page-header"
import { RecordingReplay } from "@/components/logs/recording-replay"
import { RecordingTable } from "@/components/logs/recording-table"
import { Skeleton } from "@/components/ui/skeleton"
import { useRecordingReplay, useRecordings } from "@/lib/logs/queries"

function RecordingList() {
  const recordings = useRecordings()
  if (recordings.isPending) return <Skeleton className="h-40 w-full" />
  if (recordings.error) return <Notice>녹음 목록을 불러오지 못했어요: {recordings.error.message}</Notice>
  const { dir, recordings: rows } = recordings.data
  return (
    <>
      <p className="text-sm text-muted-foreground">
        이 머신의 <code className="break-all">{dir}</code> 에 서버가 남긴 녹음이에요.
      </p>
      {rows.length ? <RecordingTable recordings={rows} /> : <Notice>아직 녹음이 없어요.</Notice>}
    </>
  )
}

function RecordingView({ id }: { id: string }) {
  const navigate = useNavigate()
  const replay = useRecordingReplay(id)
  if (replay.isPending) return <Skeleton className="h-40 w-full" />
  if (replay.error) return <Notice>녹음을 불러오지 못했어요: {replay.error.message}</Notice>
  return <RecordingReplay key={id} data={replay.data} onDeleted={() => navigate("/logs", { replace: true })} />
}

export function LogsPage() {
  const [params] = useSearchParams()
  const id = params.get("id")
  return (
    <>
      <PageHeader title="Logs" description="대화 서버 녹음 — 다시 보기와 삭제" />
      <PageBody className="max-w-[1380px]">{id ? <RecordingView id={id} /> : <RecordingList />}</PageBody>
    </>
  )
}
