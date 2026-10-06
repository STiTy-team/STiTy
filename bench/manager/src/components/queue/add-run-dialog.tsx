import { ChevronRightIcon, PlusIcon } from "lucide-react"
import { useState } from "react"

import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import type { Branches, ConfigKind, Configs } from "@/lib/api"
import { KIND_LABEL } from "@/lib/configs"
import { useAddRun, useBranches, useConfigs } from "@/lib/queries"


function defaultBranch(branches: Branches | undefined) {
  const names = branches?.branches.map((branch) => branch.name) ?? []
  if (branches?.current && names.includes(branches.current)) return branches.current
  return names.includes("main") ? "main" : names[0]
}

function ConfigPicker({
  kind,
  configs,
  value,
  onChange,
}: {
  kind: ConfigKind
  configs: Configs | undefined
  value: string | undefined
  onChange: (name: string) => void
}) {
  const names = Object.keys(configs?.[kind] ?? {}).sort()
  const chosen = value ? configs?.[kind][value] : undefined
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <Label htmlFor={`run-${kind}`}>{KIND_LABEL[kind]}</Label>
      <Select value={value} onValueChange={onChange} disabled={!configs}>
        <SelectTrigger id={`run-${kind}`} className="w-full font-mono text-xs">
          <SelectValue placeholder={configs ? `${KIND_LABEL[kind]} 선택` : "불러오는 중…"} />
        </SelectTrigger>
        <SelectContent>
          {names.map((name) => (
            <SelectItem key={name} value={name} className="font-mono text-xs">
              {name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="line-clamp-2 min-h-10 text-sm text-muted-foreground">
        {chosen ? chosen.description || "설명 없음" : ""}
      </p>
      {chosen && (
        <Collapsible className="group/yaml">
          <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground">
            <ChevronRightIcon className="size-3.5 transition-transform group-data-[state=open]/yaml:rotate-90" />
            YAML 보기
          </CollapsibleTrigger>
          <CollapsibleContent>
            <pre className="mt-2 max-h-64 overflow-auto rounded-md border bg-muted/40 p-3 font-mono text-xs">{chosen.yaml}</pre>
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  )
}

export function AddRunDialog({ host }: { host: string }) {
  const [open, setOpen] = useState(false)
  const [pipeline, setPipeline] = useState<string>()
  const [dataset, setDataset] = useState<string>()
  const [branch, setBranch] = useState<string>()
  const configs = useConfigs(open)
  const branches = useBranches(open)
  const addRun = useAddRun(host)

  const chosenBranch = branch ?? defaultBranch(branches.data)
  const ready = Boolean(configs.data && pipeline && dataset && chosenBranch)

  function changeOpen(next: boolean) {
    setOpen(next)
    if (!next) addRun.reset()
  }

  function submit() {
    if (!configs.data || !pipeline || !dataset || !chosenBranch) return
    addRun.mutate(
      { run: { branch: chosenBranch, pipeline, dataset }, configs: configs.data },
      { onSuccess: () => changeOpen(false) },
    )
  }

  const loadError = configs.error ?? branches.error
  return (
    <Dialog open={open} onOpenChange={changeOpen}>
      <DialogTrigger asChild>
        <Button>
          <PlusIcon />
          실행 추가
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>
            실행 추가 · <span className="font-mono">{host}</span>
          </DialogTitle>
          <DialogDescription>
            만들어 둔 파이프라인과 데이터셋 설정을 고르면 이 머신의 대기열에 들어가요.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-6 md:grid-cols-2">
          <ConfigPicker kind="pipeline" configs={configs.data} value={pipeline} onChange={setPipeline} />
          <ConfigPicker kind="dataset" configs={configs.data} value={dataset} onChange={setDataset} />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="run-branch">브랜치</Label>
          <Select value={chosenBranch} onValueChange={setBranch} disabled={!branches.data}>
            <SelectTrigger id="run-branch" className="w-full font-mono text-xs md:w-96">
              <SelectValue placeholder="불러오는 중…" />
            </SelectTrigger>
            <SelectContent>
              {branches.data?.branches.map((item) => (
                <SelectItem key={item.name} value={item.name} className="font-mono text-xs">
                  {item.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="text-sm text-muted-foreground">
            실행이 시작될 때 origin에 있는 이 브랜치의 최신 커밋을 써요.
          </p>
        </div>
        {(loadError || addRun.error) && (
          <p className="text-sm text-destructive">{(addRun.error ?? loadError)?.message}</p>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={() => changeOpen(false)}>
            취소
          </Button>
          <Button onClick={submit} disabled={!ready || addRun.isPending}>
            {addRun.isPending ? "추가 중…" : "실행 추가"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
