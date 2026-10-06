import { useState } from "react"
import { useNavigate, useParams } from "react-router"

import { CONFIG_ROW_ATTR, ConfigDetailsSheet } from "@/components/configs/config-details-sheet"
import { NewConfigDialog } from "@/components/configs/new-config-dialog"
import { Notice, PageBody, PageHeader } from "@/components/page-header"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type { ConfigFile, ConfigKind, Configs } from "@/lib/api"
import { KIND_LABEL } from "@/lib/configs"
import { formatDateTime, timeAgo } from "@/lib/format"
import { useConfigs } from "@/lib/queries"

const KINDS: ConfigKind[] = ["pipeline", "dataset"]
const SHOWN_TAGS = 2

const matches = (name: string, config: ConfigFile, query: string) =>
  !query || [name, config.description, ...config.tags].some((text) => text.toLowerCase().includes(query))

function Tags({ tags }: { tags: string[] }) {
  const hidden = tags.length - SHOWN_TAGS
  return (
    <div className="flex items-center gap-1" title={hidden > 0 ? tags.join(", ") : undefined}>
      {tags.slice(0, SHOWN_TAGS).map((tag) => (
        <Badge key={tag} variant="secondary">
          {tag}
        </Badge>
      ))}
      {hidden > 0 && <span className="text-xs text-muted-foreground">+{hidden}</span>}
    </div>
  )
}

function ConfigTable({
  kind,
  configs,
  query,
  opened,
  onOpen,
}: {
  kind: ConfigKind
  configs: Configs
  query: string
  opened: string | null
  onOpen: (name: string) => void
}) {
  const rows = Object.entries(configs[kind])
    .filter(([name, config]) => matches(name, config, query))
    .sort(([, a], [, b]) => (b.modified_at ?? "").localeCompare(a.modified_at ?? ""))
  if (!rows.length)
    return (
      <Notice>
        {query ? `찾는 ${KIND_LABEL[kind]} 설정이 없어요.` : `${KIND_LABEL[kind]} 설정이 아직 없어요. 오른쪽 위 버튼으로 추가해 주세요.`}
      </Notice>
    )
  return (
    <div className="rounded-xl border">
      <Table className="table-fixed">
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className="w-[32%] pl-4">이름</TableHead>
            <TableHead>설명</TableHead>
            <TableHead className="w-48">태그</TableHead>
            <TableHead className="w-24 pr-4 text-right">수정</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map(([name, config]) => (
            <TableRow
              key={name}
              {...{ [CONFIG_ROW_ATTR]: "" }}
              data-state={name === opened ? "selected" : undefined}
              className="cursor-pointer data-[state=selected]:bg-accent"
              tabIndex={0}
              onClick={() => onOpen(name)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault()
                  onOpen(name)
                }
              }}
            >
              <TableCell className="pl-4">
                <div className="flex min-w-0 items-center gap-1.5">
                  <span className="truncate font-mono text-[13px] font-medium">{name}</span>
                  {config.version > 1 && (
                    <Badge variant="outline" className="h-4.5 px-1.5 text-[11px]">
                      v{config.version}
                    </Badge>
                  )}
                </div>
              </TableCell>
              <TableCell className="truncate text-foreground/80">{config.description}</TableCell>
              <TableCell>
                <Tags tags={config.tags} />
              </TableCell>
              <TableCell className="pr-4 text-right text-muted-foreground" title={formatDateTime(config.modified_at)}>
                {timeAgo(config.modified_at)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

export function ConfigsPage() {
  const { kind: kindParam } = useParams()
  const navigate = useNavigate()
  const configs = useConfigs()
  const [query, setQuery] = useState("")
  const [opened, setOpened] = useState<string | null>(null)
  const kind = KINDS.find((k) => k === kindParam) ?? "pipeline"

  return (
    <>
      <PageHeader title="Configs" description="실행에 쓰는 파이프라인과 데이터셋 설정" />
      <PageBody>
        {configs.isPending ? (
          <Skeleton className="h-9 w-80" />
        ) : configs.error ? (
          <Notice>설정을 불러오지 못했어요: {configs.error.message}</Notice>
        ) : (
          <Tabs
            value={kind}
            onValueChange={(next) => {
              setOpened(null)
              navigate(`/configs/${next}`)
            }}
          >
            <div className="flex flex-wrap items-center gap-3">
              <TabsList>
                {KINDS.map((k) => (
                  <TabsTrigger key={k} value={k}>
                    {KIND_LABEL[k]}
                    <span className="text-muted-foreground tabular-nums">{Object.keys(configs.data[k]).length}</span>
                  </TabsTrigger>
                ))}
              </TabsList>
              <Input
                type="search"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="이름, 설명, 태그로 찾기"
                aria-label="설정 찾기"
                className="max-w-xs"
              />
              <span className="flex-1" />
              <NewConfigDialog kind={kind} configs={configs.data} />
            </div>
            {KINDS.map((k) => (
              <TabsContent key={k} value={k} className="pt-3">
                <ConfigTable kind={k} configs={configs.data} query={query.trim().toLowerCase()} opened={opened} onOpen={setOpened} />
              </TabsContent>
            ))}
            {opened && configs.data[kind][opened] && (
              <ConfigDetailsSheet
                key={`${kind}/${opened}`}
                kind={kind}
                name={opened}
                config={configs.data[kind][opened]}
                onClose={() => setOpened(null)}
                onRenamed={setOpened}
              />
            )}
          </Tabs>
        )}
      </PageBody>
    </>
  )
}
