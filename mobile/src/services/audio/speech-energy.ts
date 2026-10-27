const RMS_THRESHOLD = 0.018;
const PEAK_THRESHOLD = 0.08;
// Reads every 4th 16-bit sample: precise enough for a loudness gate and cheap on the JS thread.
const STRIDE_BYTES = 8;

/** True when a PCM s16le chunk is loud enough to be someone speaking rather than room noise. */
export const hasSpeechEnergy = (pcm: ArrayBuffer): boolean => {
  const view = new DataView(pcm);
  let sumSquares = 0;
  let peak = 0;
  let count = 0;

  for (let offset = 0; offset + 1 < view.byteLength; offset += STRIDE_BYTES) {
    const sample = view.getInt16(offset, true) / 32768;
    sumSquares += sample * sample;
    peak = Math.max(peak, Math.abs(sample));
    count += 1;
  }

  if (count === 0) return false;
  return Math.sqrt(sumSquares / count) >= RMS_THRESHOLD && peak >= PEAK_THRESHOLD;
};
