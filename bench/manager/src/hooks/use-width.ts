import { useEffect, useRef, useState } from "react"

/** The element's client width, updated on resize; changes under 3px are ignored. */
export function useWidth<T extends HTMLElement = HTMLDivElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new ResizeObserver(() => setWidth((prev) => (Math.abs(el.clientWidth - prev) >= 3 ? el.clientWidth : prev)))
    observer.observe(el)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}
