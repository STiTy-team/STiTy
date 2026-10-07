import { useCallback, useEffect, useRef, useState } from "react"

const RATE = 16000

export type RecorderStatus = "idle" | "arming" | "armed" | "recording" | "saving" | "error"

function toPCM(chunks: Float32Array[]): Int16Array {
  const pcm = new Int16Array(chunks.reduce((n, c) => n + c.length, 0))
  let i = 0
  for (const chunk of chunks) {
    for (const s of chunk) {
      const v = Math.max(-1, Math.min(1, s))
      pcm[i++] = v < 0 ? v * 0x8000 : v * 0x7fff
    }
  }
  return pcm
}

/** Arms a 16 kHz mono mic graph with every browser DSP step off (echo cancellation and
 * noise suppression are trained to erase an overlapping second voice, which is exactly
 * what an overlap-recording dataset needs to keep), captures float samples through an
 * AudioWorklet, and hands the finished take to `onStop` as s16le PCM. */
export function useRecorder(onStop: (pcm: Int16Array) => Promise<void> | void) {
  const [status, setStatus] = useState<RecorderStatus>("idle")
  const [error, setError] = useState<string | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const ctxRef = useRef<AudioContext | null>(null)
  const nodeRef = useRef<AudioWorkletNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Float32Array[]>([])
  const framesRef = useRef(0)
  const onStopRef = useRef(onStop)
  useEffect(() => {
    onStopRef.current = onStop
  }, [onStop])

  useEffect(
    () => () => {
      streamRef.current?.getTracks().forEach((t) => t.stop())
      void ctxRef.current?.close()
    },
    [],
  )

  const arm = useCallback(async () => {
    setStatus("arming")
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      })
      streamRef.current = stream

      const ctx = new AudioContext({ sampleRate: RATE })
      if (Math.round(ctx.sampleRate) !== RATE) {
        await ctx.close()
        throw new Error(`브라우저가 16 kHz 를 거부했어요 (${ctx.sampleRate} Hz)`)
      }
      await ctx.audioWorklet.addModule("/recorder-worklet.js")
      ctxRef.current = ctx

      const node = new AudioWorkletNode(ctx, "capture", { channelCount: 1 })
      node.port.onmessage = (e: MessageEvent<Float32Array>) => {
        chunksRef.current.push(e.data)
        framesRef.current += e.data.length
        setElapsed(framesRef.current / RATE)
      }
      nodeRef.current = node

      // A muted sink keeps the graph pulling without routing the mic back to the
      // speakers, which in a room with two people talking is a feedback loop.
      const sink = ctx.createGain()
      sink.gain.value = 0
      ctx.createMediaStreamSource(stream).connect(node)
      node.connect(sink).connect(ctx.destination)

      setStatus("armed")
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setStatus("error")
    }
  }, [])

  const start = useCallback(() => {
    chunksRef.current = []
    framesRef.current = 0
    setElapsed(0)
    nodeRef.current?.port.postMessage({ recording: true })
    setStatus("recording")
  }, [])

  const stop = useCallback(async () => {
    nodeRef.current?.port.postMessage({ recording: false })
    setStatus("saving")
    const pcm = toPCM(chunksRef.current)
    chunksRef.current = []
    try {
      await onStopRef.current(pcm)
      setStatus("armed")
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setStatus("error")
    }
  }, [])

  const toggle = useCallback(() => {
    if (!nodeRef.current) return void arm()
    if (status === "recording") return void stop()
    if (status === "armed") start()
  }, [status, arm, start, stop])

  return { status, error, elapsed, toggle }
}
