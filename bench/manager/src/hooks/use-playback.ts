import { useCallback, useEffect, useRef, useState, type RefObject } from "react"

const AUDIO_MAX_SPEED = 4
const DRIFT_SEC = 0.25

export type Playback = {
  now: number
  playing: boolean
  jump: number
  seek: (to: number) => void
  toggle: () => void
  restart: () => void
}

export function usePlayback(span: number, speed: number): Playback {
  const [now, setNow] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [jump, setJump] = useState(0)
  const nowRef = useRef(now)
  useEffect(() => {
    nowRef.current = now
  }, [now])

  useEffect(() => {
    if (!playing) return
    let last: number | null = null
    let raf = requestAnimationFrame(function frame(ts) {
      if (last != null) {
        const step = ((ts - last) / 1000) * speed
        setNow((prev) => Math.min(span, prev + step))
        if (nowRef.current + step >= span) setPlaying(false)
      }
      last = ts
      raf = requestAnimationFrame(frame)
    })
    return () => cancelAnimationFrame(raf)
  }, [playing, speed, span])

  const seek = useCallback(
    (to: number) => {
      setNow(Math.max(0, Math.min(span, to)))
      setJump((j) => j + 1)
    },
    [span],
  )

  const toggle = useCallback(() => {
    if (!playing && now >= span) setNow(0)
    setPlaying(!playing)
  }, [playing, now, span])

  const restart = useCallback(() => {
    seek(0)
    setPlaying(true)
  }, [seek])

  return { now, playing, jump, seek, toggle, restart }
}

/** Keeps an <audio> element on the replay clock: plays along within the clip, pauses otherwise. */
export function useAudioSync(
  audio: RefObject<HTMLAudioElement | null>,
  { url, now, playing, speed }: { url: string | null; now: number; playing: boolean; speed: number },
) {
  const pendingTime = useRef<number | null>(null)

  useEffect(() => {
    const player = audio.current
    if (!player) return
    pendingTime.current = null
    if (url) player.src = url
    else player.removeAttribute("src")
    const onMeta = () => {
      if (pendingTime.current != null) player.currentTime = pendingTime.current
      pendingTime.current = null
    }
    player.addEventListener("loadedmetadata", onMeta)
    return () => {
      player.removeEventListener("loadedmetadata", onMeta)
      player.pause()
    }
  }, [audio, url])

  useEffect(() => {
    const player = audio.current
    if (!player) return
    if (!url) {
      if (!player.paused) player.pause()
      return
    }
    const seekTo = (to: number) => {
      if (player.readyState >= 1) player.currentTime = to
      else pendingTime.current = to
    }
    const duration = Number.isFinite(player.duration) ? player.duration : null
    const inClip = now >= 0 && (duration == null || now < duration - 0.02)
    player.playbackRate = Math.min(speed, AUDIO_MAX_SPEED)
    if (!playing || speed > AUDIO_MAX_SPEED || !inClip) {
      if (!player.paused) player.pause()
      if (!playing && inClip) seekTo(now)
      return
    }
    if (Math.abs(player.currentTime - now) > DRIFT_SEC) seekTo(now)
    if (player.paused) player.play().catch(() => {})
  }, [audio, url, now, playing, speed])
}

/** Scrolls a panel to keep the playhead in view while playing and right after a seek. */
export function useFollow(key: unknown, playing: boolean, jump: number, scroll: () => void) {
  const seenJump = useRef(jump)
  const scrollRef = useRef(scroll)
  useEffect(() => {
    scrollRef.current = scroll
  })
  useEffect(() => {
    const jumped = seenJump.current !== jump
    seenJump.current = jump
    if (playing || jumped) scrollRef.current()
  }, [key, playing, jump])
}

/** Scrolls `box` itself (never the page) so `el` sits a little above its middle. */
export function scrollWithin(box: HTMLElement | null, el: Element | null | undefined) {
  if (!box || !el) return
  const b = box.getBoundingClientRect()
  const r = el.getBoundingClientRect()
  if (r.top >= b.top && r.bottom <= b.bottom) return
  box.scrollTop += r.top - b.top - (box.clientHeight - r.height) * 0.6
}
