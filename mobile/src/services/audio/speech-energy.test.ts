import { hasSpeechEnergy } from './speech-energy';

const pcm = (amplitude: number, samples = 1_600) => {
  const view = new DataView(new ArrayBuffer(samples * 2));
  for (let i = 0; i < samples; i += 1) {
    view.setInt16(i * 2, Math.round(Math.sin(i / 4) * amplitude * 32767), true);
  }
  return view.buffer;
};

describe('hasSpeechEnergy', () => {
  it('ignores silence and room noise', () => {
    expect(hasSpeechEnergy(pcm(0))).toBe(false);
    expect(hasSpeechEnergy(pcm(0.01))).toBe(false);
  });

  it('detects a speaking level signal', () => {
    expect(hasSpeechEnergy(pcm(0.3))).toBe(true);
  });

  it('handles an empty chunk', () => {
    expect(hasSpeechEnergy(new ArrayBuffer(0))).toBe(false);
  });
});
