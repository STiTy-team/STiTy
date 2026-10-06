import { CircleAlertIcon, CircleCheckIcon, LoaderCircleIcon, PlusIcon } from "lucide-react"
import { useState, type ReactNode } from "react"

import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { useDebounced } from "@/hooks/use-debounced"
import { ApiError, type ConfigKind, type Configs } from "@/lib/api"
import { KIND_LABEL, NAME_EXAMPLE, TEMPLATES, withVersion } from "@/lib/configs"
import { formatDateTime } from "@/lib/format"
import { useCheckConfig, useCreateConfig } from "@/lib/queries"
import { cn } from "@/lib/utils"

const CHECK_DELAY_MS = 400

function Verdict({ tone, children }: { tone: "ok" | "bad" | "wait"; children: ReactNode }) {
  const Icon = tone === "ok" ? CircleCheckIcon : tone === "bad" ? CircleAlertIcon : LoaderCircleIcon
  return (
    <div
      className={cn(
        "flex items-start gap-2 rounded-md border px-3 py-2 text-sm",
        tone === "ok" && "border-status-done/30 text-status-done",
        tone === "bad" && "border-destructive/30 text-destructive",
        tone === "wait" && "text-muted-foreground",
      )}
    >
      <Icon className={cn("mt-0.5 size-4 shrink-0", tone === "wait" && "animate-spin")} />
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  )
}

function ConfigForm({ kind, configs, onDone }: { kind: ConfigKind; configs: Configs | undefined; onDone: () => void }) {
  const [name, setName] = useState("")
  const [yaml, setYaml] = useState(TEMPLATES[kind])
  const trimmed = name.trim()
  const settledName = useDebounced(trimmed, CHECK_DELAY_MS)
  const settledYaml = useDebounced(yaml, CHECK_DELAY_MS)
  const exists = Boolean(configs?.[kind][trimmed])
  const check = useCheckConfig(kind, exists ? "" : settledName, settledYaml)
  const create = useCreateConfig(kind)

  const settled = settledName === trimmed && settledYaml === yaml
  const result = settled ? check.data : undefined
  const savable = !exists && result?.valid === true && !result.taken && !create.isPending

  let verdict: ReactNode
  if (!trimmed) verdict = <Verdict tone="bad">이름을 정해 주세요. 예: {NAME_EXAMPLE[kind]}. 규칙은 configs/README.md에 있어요.</Verdict>
  else if (exists)
    verdict = <Verdict tone="bad">같은 이름이 이미 있어요: {trimmed}. 다른 이름을 골라 주세요.</Verdict>
  else if (!settled || check.isFetching) verdict = <Verdict tone="wait">확인 중…</Verdict>
  else if (check.error) verdict = <Verdict tone="bad">{check.error.message}</Verdict>
  else if (result && !result.valid)
    verdict = (
      <Verdict tone="bad">
        이대로는 저장할 수 없어요.
        <pre className="mt-1 font-mono text-xs whitespace-pre-wrap">{result.error}</pre>
      </Verdict>
    )
  else if (result?.taken)
    verdict = (
      <Verdict tone="bad">
        {result.ref} 이름은 {result.taken.by}님이 {formatDateTime(result.taken.at)}에 다른 내용으로 이미 썼어요. 새 버전이
        필요해요.
        <Button size="sm" variant="outline" className="mt-2 flex" onClick={() => setYaml(withVersion(yaml, result.free_version))}>
          버전 올리기 (v{result.free_version})
        </Button>
      </Verdict>
    )
  else if (result) verdict = <Verdict tone="ok">저장할 수 있어요: {result.ref}</Verdict>

  const save = () =>
    create.mutate(
      { name: trimmed, yaml },
      {
        onSuccess: onDone,
      },
    )
  const saveError =
    create.error instanceof ApiError && create.error.status === 409
      ? "방금 다른 사람이 같은 이름으로 만들었어요. 다른 이름을 골라 주세요."
      : create.error?.message

  return (
    <>
      <div className="flex min-w-0 flex-col gap-4">
        <div className="flex flex-col gap-2">
          <Label htmlFor="config-name">이름</Label>
          <Input
            id="config-name"
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder={NAME_EXAMPLE[kind]}
            className="font-mono"
            autoComplete="off"
            spellCheck={false}
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="config-yaml">YAML</Label>
          <Textarea
            id="config-yaml"
            value={yaml}
            onChange={(e) => setYaml(e.target.value)}
            className="h-80 resize-y font-mono text-xs"
            spellCheck={false}
          />
        </div>
        {verdict}
        {saveError && <Verdict tone="bad">{saveError}</Verdict>}
      </div>
      <DialogFooter>
        <Button variant="outline" onClick={onDone}>
          취소
        </Button>
        <Button onClick={save} disabled={!savable}>
          {create.isPending ? "저장 중…" : "S3에 저장"}
        </Button>
      </DialogFooter>
    </>
  )
}

export function NewConfigDialog({ kind, configs }: { kind: ConfigKind; configs: Configs | undefined }) {
  const [open, setOpen] = useState(false)
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>
          <PlusIcon />
          새 {KIND_LABEL[kind]}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>새 {KIND_LABEL[kind]} 설정</DialogTitle>
          <DialogDescription>S3에 저장돼서 팀 전체와 모든 worker가 바로 쓸 수 있어요.</DialogDescription>
        </DialogHeader>
        {open && <ConfigForm kind={kind} configs={configs} onDone={() => setOpen(false)} />}
      </DialogContent>
    </Dialog>
  )
}
