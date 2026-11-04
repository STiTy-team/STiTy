import { useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { formatClock } from "@/lib/format"
import { useDatasetShape, useVerifyDataset } from "@/lib/datasets/queries"
import { cn } from "@/lib/utils"

const FILES: { key: "manifest" | "alignment" | "readme" | "convert"; label: string }[] = [
  { key: "manifest", label: "manifest.jsonl" },
  { key: "alignment", label: "alignment.jsonl" },
  { key: "readme", label: "README.md" },
  { key: "convert", label: "convert.py" },
]

function num(n: number | undefined | null, digits = 1): string {
  return n == null ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: digits })
}

function pct(a: number, b: number): string {
  return b ? `${Math.round((100 * a) / b)}%` : "—"
}

function Stat({ label, value, note, flag }: { label: string; value: string; note?: string; flag?: boolean }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-lg font-medium tabular-nums">{value}</span>
      {note && <span className={cn("text-xs", flag ? "text-destructive" : "text-muted-foreground")}>{note}</span>}
    </div>
  )
}

function Bars({ entries, total }: { entries: [string, number][]; total: number }) {
  const max = Math.max(1, ...entries.map(([, v]) => v))
  if (!entries.length) return <p className="text-sm text-muted-foreground">없음</p>
  return (
    <div className="flex flex-col gap-1.5">
      {entries.map(([label, v]) => (
        <div key={label} className="grid grid-cols-[8rem_1fr_3.5rem_3rem] items-center gap-2 text-xs">
          <span className="truncate text-muted-foreground" title={label}>
            {label}
          </span>
          <div className="h-1.5 rounded-full bg-muted">
            <div className="h-full rounded-full bg-foreground/60" style={{ width: `${(100 * v) / max}%` }} />
          </div>
          <span className="text-right tabular-nums">{v.toLocaleString()}</span>
          <span className="text-right text-muted-foreground tabular-nums">{pct(v, total)}</span>
        </div>
      ))}
    </div>
  )
}

function binLabel(lo: number, hi: number | null): string {
  return hi === null ? `${lo}s+` : `${lo}–${hi}s`
}

export function ShapePanel({ ds }: { ds: string }) {
  const shape = useDatasetShape(ds)
  const verify = useVerifyDataset()
  const [verified, setVerified] = useState<{ ds: string; ok: boolean; report: string } | null>(null)

  if (shape.isPending) return <Skeleton className="h-64 w-full" />
  if (shape.error) return <p className="text-sm text-destructive">모양을 불러오지 못했어요: {shape.error.message}</p>

  const s = shape.data
  const spec = s.spec ?? {}
  const provides = spec.provides ?? {}
  const declared = provides.translations ?? []
  const translations: [string, number][] = [...new Set([...declared, ...Object.keys(s.translations)])].map((lang) => [
    lang,
    s.translations[lang] ?? 0,
  ])
  const shortTranslations = translations.some(([lang, n]) => declared.includes(lang) && n < s.items)

  return (
    <div className="flex flex-col gap-6 rounded-xl border p-4 md:p-6">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge variant={Object.keys(spec).length > 0 ? "secondary" : "outline"} className="font-mono">
          dataset.yml
        </Badge>
        {FILES.map(({ key, label }) => (
          <Badge key={key} variant={s.files[key] ? "secondary" : "outline"} className={cn("font-mono", !s.files[key] && "text-muted-foreground/60")}>
            {label}
          </Badge>
        ))}
        <span className="flex-1" />
        <code className="truncate text-xs text-muted-foreground" title={s.path}>
          {s.path}
        </code>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="항목" value={num(s.items, 0)} />
        <Stat label="총 길이" value={formatClock(s.duration.total)} />
        <Stat
          label="세션 (group)"
          value={num(s.sessions, 0)}
          note={s.sessions_contiguous ? undefined : "연속이 아니에요"}
          flag={!s.sessions_contiguous}
        />
        <Stat label="화자" value={num(s.speakers, 0)} />
        <Stat
          label="오디오 파일"
          value={num(s.audio_files, 0)}
          note={s.audio_missing ? `${s.audio_missing}개 없음` : undefined}
          flag={s.audio_missing > 0}
        />
        <Stat label="offset 항목" value={num(s.offset_items, 0)} />
        <Stat label="partial 항목" value={num(s.partial_items, 0)} />
        <Stat label="primary_metric" value={spec.primary_metric ?? "—"} />
      </div>

      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-medium">길이 <span className="text-muted-foreground">초</span></h3>
        {s.items ? (
          <>
            <div className="flex flex-wrap gap-4 text-xs text-muted-foreground tabular-nums">
              <span>min <b className="text-foreground">{num(s.duration.min, 2)}</b></span>
              <span>p50 <b className="text-foreground">{num(s.duration.p50, 2)}</b></span>
              <span>mean <b className="text-foreground">{num(s.duration.mean, 2)}</b></span>
              <span>p90 <b className="text-foreground">{num(s.duration.p90, 2)}</b></span>
              <span>max <b className="text-foreground">{num(s.duration.max, 2)}</b></span>
            </div>
            <Bars entries={s.histogram.filter((b) => b.count).map((b) => [binLabel(b.lo, b.hi), b.count])} total={s.items} />
          </>
        ) : (
          <p className="text-sm text-muted-foreground">항목 없음</p>
        )}
      </div>

      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-medium">src_lang</h3>
        <Bars entries={Object.entries(s.src_lang)} total={s.items} />
      </div>

      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-medium">
          참조 {shortTranslations && <span className="text-destructive">선언한 번역이 빠진 항목이 있어요</span>}
        </h3>
        <Bars
          entries={[
            [`transcript${provides.transcript === false ? " (선언 안 함)" : ""}`, s.transcript.filled],
            ...translations.map(([lang, n]): [string, number] => [`→ ${lang}${declared.includes(lang) ? "" : " (선언 안 함)"}`, n]),
          ]}
          total={s.items}
        />
        {s.transcript.units && (
          <p className="text-xs text-muted-foreground">
            전사 길이 (단어, 중·일은 글자): min {s.transcript.units.min} · p50 {num(s.transcript.units.p50, 0)} · p90{" "}
            {s.transcript.units.p90} · max {s.transcript.units.max}
          </p>
        )}
      </div>

      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-medium">
          정렬{" "}
          {s.alignment.stale > 0 && <span className="text-destructive">{s.alignment.stale}개가 옛 전사 기준이에요</span>}{" "}
          {s.alignment.orphan > 0 && <span className="text-destructive">{s.alignment.orphan}개는 manifest에 없어요</span>}
        </h3>
        {s.files.alignment ? (
          <>
            <Bars entries={[["aligned", s.alignment.items]]} total={s.transcript.filled || s.items} />
            <p className="text-xs text-muted-foreground">
              {Object.entries(s.alignment.aligners).map(([k, v]) => `${k} × ${v}`).join(" · ") || "—"}
            </p>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">alignment.jsonl 없음</p>
        )}
      </div>

      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-medium">행의 필드</h3>
        <div className="rounded-lg border">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="pl-3">키</TableHead>
                <TableHead>타입</TableHead>
                <TableHead className="pr-3 text-right">개수</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {s.fields.map((f) => (
                <TableRow key={f.key}>
                  <TableCell className={cn("pl-3 font-mono text-xs", f.key.includes(".") && "pl-6 text-muted-foreground")}>{f.key}</TableCell>
                  <TableCell className="text-xs text-muted-foreground">{Object.keys(f.types).join(" | ")}</TableCell>
                  <TableCell
                    className={cn(
                      "pr-3 text-right text-xs tabular-nums",
                      f.count < s.items && !f.key.startsWith("reference.translations.") && "text-destructive",
                    )}
                  >
                    {f.count}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>

      {s.files.manifest_sha256 && (
        <p className="text-xs text-muted-foreground">
          manifest sha256 <code className="font-mono">{s.files.manifest_sha256}</code> · {num(s.files.manifest_bytes / 1024, 0)} KB
        </p>
      )}

      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-medium">dataset.yml</h3>
        <pre className="max-h-64 overflow-auto rounded-lg border bg-muted/40 p-3 font-mono text-xs leading-relaxed">
          {Object.keys(spec).length ? JSON.stringify(spec, null, 2) : "dataset.yml 없음"}
        </pre>
      </div>

      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium">검증</h3>
          <Button
            variant="outline"
            size="sm"
            disabled={verify.isPending}
            onClick={() =>
              verify.mutate(ds, {
                onSuccess: (result) => setVerified({ ds, ok: result.ok, report: result.report }),
                onError: (error) => setVerified({ ds, ok: false, report: error.message }),
              })
            }
          >
            {verify.isPending ? "검증 중…" : "verify() 돌리기"}
          </Button>
        </div>
        {verified?.ds === ds && (
          <pre
            className={cn(
              "max-h-64 overflow-auto rounded-lg p-3 font-mono text-xs leading-relaxed",
              verified.ok ? "bg-status-done-soft text-status-done" : "bg-status-failed-soft text-status-failed",
            )}
          >
            {verified.report}
          </pre>
        )}
      </div>
    </div>
  )
}
