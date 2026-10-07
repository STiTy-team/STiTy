// Float samples off the audio graph. The AudioContext runs at the contract's
// 16 kHz, so a quantum arrives at the target rate and leaves as a copy.

class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.recording = false;
    this.port.onmessage = (e) => { this.recording = !!e.data.recording; };
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && this.recording) {
      const pcm = new Float32Array(channel);
      this.port.postMessage(pcm, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor("capture", Capture);
