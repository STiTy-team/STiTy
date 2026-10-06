import { ChartColumnIcon, PencilIcon } from "lucide-react"
import { useState } from "react"
import { Link } from "react-router"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Sheet, SheetContent, SheetDescription, SheetFooter, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Textarea } from "@/components/ui/textarea"
import { ApiError, type ConfigFile, type ConfigKind } from "@/lib/api"
import { useOverview } from "@/lib/compare/queries"
import { compareLink, configRef, KIND_LABEL } from "@/lib/configs"
import { formatDateTime, timeAgo } from "@/lib/format"
import { useEditConfigMeta } from "@/lib/queries"

const NAME_PATTERN = /^[a-z0-9][a-z0-9.+-]*(_[a-z0-9][a-z0-9.+-]*)*$/

const splitTags = (text: string) => [...new Set(text.split(",").map((tag) => tag.trim()).filter(Boolean))]

export const CONFIG_ROW_ATTR = "data-config-row"

function CompareButton({ kind, name, config }: { kind: ConfigKind; name: string; config: ConfigFile }) {
  const overview = useOverview()
  const link = overview.data && compareLink(kind, configRef(name, config.version), overview.data.runs)
  if (!link)
    return (
      <Button variant="outline" disabled>
        <ChartColumnIcon />
        {overview.isPending ? "실행을 찾는 중…" : "비교할 실행이 아직 없어요"}
      </Button>
    )
  return (
    <Button variant="outline" asChild>
      <Link to={link}>
        <ChartColumnIcon />
        실행 비교
      </Link>
    </Button>
  )
}

function Details({ kind, name, config, onEdit }: { kind: ConfigKind; name: string; config: ConfigFile; onEdit: () => void }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 px-4 pb-4">
      {config.tags.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {config.tags.map((tag) => (
            <Badge key={tag} variant="secondary">
              {tag}
            </Badge>
          ))}
        </div>
      )}
      <p className="text-sm text-muted-foreground" title={formatDateTime(config.modified_at)}>
        v{config.version} · {timeAgo(config.modified_at)} 수정
      </p>
      <div className="flex flex-wrap gap-2">
        <CompareButton kind={kind} name={name} config={config} />
        <Button variant="outline" onClick={onEdit}>
          <PencilIcon />
          수정
        </Button>
      </div>
      <div className="flex min-h-0 flex-1 flex-col gap-1.5">
        <span className="text-xs text-muted-foreground">YAML</span>
        <pre className="min-h-0 flex-1 overflow-auto rounded-lg border bg-muted/40 p-3 font-mono text-xs leading-relaxed">{config.yaml}</pre>
      </div>
    </div>
  )
}

function EditForm({
  kind,
  name,
  config,
  onCancel,
  onSaved,
}: {
  kind: ConfigKind
  name: string
  config: ConfigFile
  onCancel: () => void
  onSaved: (name: string) => void
}) {
  const [newName, setNewName] = useState(name)
  const [description, setDescription] = useState(config.description)
  const [tags, setTags] = useState(config.tags.join(", "))
  const [version, setVersion] = useState(String(config.version))
  const edit = useEditConfigMeta(kind)

  const trimmed = newName.trim()
  const versionNumber = Number(version)
  const problem = !NAME_PATTERN.test(trimmed)
    ? "이름은 '_'로 이은 소문자 조각이어야 해요. 각 조각에는 a-z, 0-9, . + - 만 쓸 수 있어요 (configs/README.md 참고)."
    : !Number.isInteger(versionNumber) || versionNumber < 1
      ? "버전은 1 이상의 정수여야 해요."
      : null

  const save = () =>
    edit.mutate(
      {
        name,
        edit: {
          etag: config.etag,
          name: trimmed,
          meta: { description: description.trim(), tags: splitTags(tags), ...(versionNumber > 1 && { version: versionNumber }) },
        },
      },
      { onSuccess: (saved) => onSaved(saved.name) },
    )
  const error =
    edit.error instanceof ApiError && edit.error.status === 409
      ? "수정하는 사이에 다른 사람이 이 설정을 바꿨어요. 최신 내용을 다시 불러왔으니 수정을 다시 열어 주세요."
      : edit.error?.message

  return (
    <>
      <div className="flex min-w-0 flex-col gap-4 overflow-auto px-4">
        <p className="text-sm text-muted-foreground">여기서는 이름과 meta 블록만 바꿀 수 있어요. 나머지 YAML은 그대로예요.</p>
        <div className="flex flex-col gap-2">
          <Label htmlFor="edit-name">이름</Label>
          <Input
            id="edit-name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            className="font-mono"
            autoComplete="off"
            spellCheck={false}
          />
          {trimmed !== name && (
            <p className="text-xs text-muted-foreground">이름을 바꾸면 S3에서 설정 파일이 옮겨져요. 지난 실행은 원래 이름을 그대로 써요.</p>
          )}
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="edit-description">설명</Label>
          <Textarea id="edit-description" value={description} onChange={(e) => setDescription(e.target.value)} className="min-h-20" />
        </div>
        <div className="grid grid-cols-[1fr_6rem] gap-4">
          <div className="flex flex-col gap-2">
            <Label htmlFor="edit-tags">태그</Label>
            <Input id="edit-tags" value={tags} onChange={(e) => setTags(e.target.value)} placeholder="fleurs, sentence, clean" />
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="edit-version">버전</Label>
            <Input id="edit-version" type="number" min={1} value={version} onChange={(e) => setVersion(e.target.value)} />
          </div>
        </div>
        {(problem || error) && <p className="text-sm text-destructive">{problem ?? error}</p>}
      </div>
      <SheetFooter className="flex-row justify-end">
        <Button variant="outline" onClick={onCancel}>
          취소
        </Button>
        <Button onClick={save} disabled={Boolean(problem) || edit.isPending}>
          {edit.isPending ? "저장 중…" : "S3에 저장"}
        </Button>
      </SheetFooter>
    </>
  )
}

export function ConfigDetailsSheet({
  kind,
  name,
  config,
  onClose,
  onRenamed,
}: {
  kind: ConfigKind
  name: string
  config: ConfigFile
  onClose: () => void
  onRenamed: (name: string) => void
}) {
  const [editing, setEditing] = useState(false)
  return (
    <Sheet open modal={false} onOpenChange={(open) => !open && onClose()}>
      <SheetContent
        className="w-full data-[side=right]:sm:max-w-lg"
        onInteractOutside={(event) => {
          if ((event.target as Element | null)?.closest(`[${CONFIG_ROW_ATTR}]`)) event.preventDefault()
        }}
      >
        <SheetHeader className="pr-12">
          <SheetDescription>
            {KIND_LABEL[kind]}
            {editing && " 수정"}
          </SheetDescription>
          <SheetTitle className="font-mono text-base font-medium break-all">{name}</SheetTitle>
          {!editing && config.description && <p className="pt-1 text-sm">{config.description}</p>}
        </SheetHeader>
        {editing ? (
          <EditForm
            kind={kind}
            name={name}
            config={config}
            onCancel={() => setEditing(false)}
            onSaved={(saved) => {
              setEditing(false)
              onRenamed(saved)
            }}
          />
        ) : (
          <Details kind={kind} name={name} config={config} onEdit={() => setEditing(true)} />
        )}
      </SheetContent>
    </Sheet>
  )
}
