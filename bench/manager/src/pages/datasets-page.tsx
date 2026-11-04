import { useNavigate, useParams, useSearchParams } from "react-router"

import { BrowsePanel } from "@/components/datasets/browse-panel"
import { DatasetPicker } from "@/components/datasets/dataset-picker"
import { NewDatasetDialog } from "@/components/datasets/new-dataset-dialog"
import { RecorderPanel } from "@/components/datasets/recorder-panel"
import { Notice, PageBody, PageHeader } from "@/components/page-header"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useDatasetsList } from "@/lib/datasets/queries"

const TABS = ["browse", "record"] as const
type Tab = (typeof TABS)[number]
const TAB_LABEL: Record<Tab, string> = { browse: "살펴보기", record: "녹음" }

export function DatasetsPage() {
  const { tab: tabParam } = useParams()
  const navigate = useNavigate()
  const [search, setSearch] = useSearchParams()
  const list = useDatasetsList()

  const tab = TABS.find((t) => t === tabParam) ?? "browse"
  const datasets = list.data?.datasets ?? []
  const ds = datasets.some((d) => d.name === search.get("ds")) ? search.get("ds") : datasets[0]?.name ?? null

  const setDs = (name: string) => setSearch((prev) => ({ ...Object.fromEntries(prev), ds: name }), { replace: true })

  return (
    <>
      <PageHeader title="Datasets" description="데이터셋을 살펴보고, 새 항목을 녹음해 넣어요" />
      <PageBody>
        {list.isPending ? (
          <Skeleton className="h-9 w-80" />
        ) : list.error ? (
          <Notice>데이터셋 목록을 불러오지 못했어요: {list.error.message}</Notice>
        ) : datasets.length === 0 ? (
          <div className="flex items-center gap-3">
            <Notice>데이터셋이 아직 없어요.</Notice>
            <NewDatasetDialog onCreated={setDs} />
          </div>
        ) : (
          <Tabs value={tab} onValueChange={(next) => navigate(`/datasets/${next}?${search.toString()}`)}>
            <div className="flex flex-wrap items-center gap-3">
              <TabsList>
                {TABS.map((t) => (
                  <TabsTrigger key={t} value={t}>
                    {TAB_LABEL[t]}
                  </TabsTrigger>
                ))}
              </TabsList>
              <DatasetPicker datasets={datasets} value={ds} onChange={setDs} />
              <span className="flex-1" />
              <NewDatasetDialog onCreated={setDs} />
            </div>
            {ds && (
              <>
                <TabsContent value="browse" className="pt-4">
                  <BrowsePanel key={ds} ds={ds} />
                </TabsContent>
                <TabsContent value="record" className="pt-4">
                  <RecorderPanel key={ds} ds={ds} />
                </TabsContent>
              </>
            )}
          </Tabs>
        )}
      </PageBody>
    </>
  )
}
