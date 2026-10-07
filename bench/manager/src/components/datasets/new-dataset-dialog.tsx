import { PlusIcon } from "lucide-react"
import { useState } from "react"

import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useCreateDataset } from "@/lib/datasets/queries"

const NAME_PATTERN = /^[A-Za-z0-9_-]{1,64}$/

function NewDatasetForm({ onDone }: { onDone: (name: string) => void }) {
  const [name, setName] = useState("")
  const [languages, setLanguages] = useState("")
  const create = useCreateDataset()
  const trimmed = name.trim()
  const problem = !trimmed ? null : !NAME_PATTERN.test(trimmed) ? "이름은 영문·숫자·_·- 만, 64자 이내로 써 주세요." : null

  const save = () => {
    if (!trimmed || problem) return
    create.mutate({ name: trimmed, languages }, { onSuccess: (saved) => onDone(saved.name) })
  }

  return (
    <>
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-2">
          <Label htmlFor="dataset-name">이름</Label>
          <Input
            id="dataset-name"
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="overlap_ko"
            maxLength={64}
            className="font-mono"
            autoComplete="off"
            spellCheck={false}
            onKeyDown={(e) => e.key === "Enter" && save()}
          />
        </div>
        <div className="flex flex-col gap-2">
          <Label htmlFor="dataset-langs">언어 <span className="text-muted-foreground">쉼표로 여러 개</span></Label>
          <Input
            id="dataset-langs"
            value={languages}
            onChange={(e) => setLanguages(e.target.value)}
            placeholder="ko, en"
            autoComplete="off"
            onKeyDown={(e) => e.key === "Enter" && save()}
          />
        </div>
        {(problem || create.error) && <p className="text-sm text-destructive">{problem ?? create.error?.message}</p>}
      </div>
      <DialogFooter>
        <Button variant="outline" onClick={() => onDone("")}>
          취소
        </Button>
        <Button onClick={save} disabled={!trimmed || Boolean(problem) || create.isPending}>
          {create.isPending ? "만드는 중…" : "만들기"}
        </Button>
      </DialogFooter>
    </>
  )
}

export function NewDatasetDialog({ onCreated }: { onCreated: (name: string) => void }) {
  const [open, setOpen] = useState(false)
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline">
          <PlusIcon />새 데이터셋
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>새 데이터셋</DialogTitle>
          <DialogDescription>
            데이터셋 루트에 디렉토리와 dataset.yml, README 초안이 만들어져요. 녹음 탭에서 바로 녹음할 수 있어요.
          </DialogDescription>
        </DialogHeader>
        {open && (
          <NewDatasetForm
            onDone={(name) => {
              setOpen(false)
              if (name) onCreated(name)
            }}
          />
        )}
      </DialogContent>
    </Dialog>
  )
}
