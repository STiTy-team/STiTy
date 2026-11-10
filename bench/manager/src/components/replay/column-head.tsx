import type { ReactNode } from "react"

export function ColumnHead({
  title,
  description,
  children,
}: {
  title?: string
  description: ReactNode
  children?: ReactNode
}) {
  return (
    <div className="mb-3 flex min-h-9 flex-wrap items-center gap-x-3 gap-y-2">
      {title && <h2 className="font-semibold">{title}</h2>}
      <p className="text-sm text-muted-foreground">{description}</p>
      <span className="flex-1" />
      {children}
    </div>
  )
}
