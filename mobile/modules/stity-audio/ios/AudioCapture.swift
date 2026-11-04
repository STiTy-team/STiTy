import AVFoundation

final class AudioCapture {
  private let engine = AVAudioEngine()
  private let onChunk: (String) -> Void
  private var pending = Data()
  private var chunkBytes = 0

  private(set) var isRunning = false

  init(onChunk: @escaping (String) -> Void) {
    self.onChunk = onChunk
  }

  func start(sampleRate: Double, chunkSamples: Int) throws {
    guard !isRunning else { return }

    let input = engine.inputNode
    try input.setVoiceProcessingEnabled(true)

    let inputFormat = input.outputFormat(forBus: 0)
    guard
      let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatInt16,
        sampleRate: sampleRate,
        channels: 1,
        interleaved: true
      ),
      let converter = AVAudioConverter(from: inputFormat, to: targetFormat)
    else {
      throw CaptureError.unsupportedFormat
    }

    chunkBytes = chunkSamples * MemoryLayout<Int16>.size
    pending.removeAll(keepingCapacity: true)

    let tapFrames = AVAudioFrameCount(inputFormat.sampleRate / 10)
    input.installTap(onBus: 0, bufferSize: tapFrames, format: inputFormat) { [weak self] buffer, _ in
      self?.process(buffer, converter: converter, targetFormat: targetFormat)
    }

    engine.prepare()
    do {
      try engine.start()
    } catch {
      input.removeTap(onBus: 0)
      throw error
    }
    isRunning = true
  }

  func stop() {
    guard isRunning else { return }
    engine.inputNode.removeTap(onBus: 0)
    engine.stop()
    pending.removeAll()
    isRunning = false
  }

  private func process(
    _ buffer: AVAudioPCMBuffer,
    converter: AVAudioConverter,
    targetFormat: AVAudioFormat
  ) {
    let ratio = targetFormat.sampleRate / buffer.format.sampleRate
    let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1
    guard let output = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else {
      return
    }

    var consumed = false
    var error: NSError?
    converter.convert(to: output, error: &error) { _, status in
      if consumed {
        status.pointee = .noDataNow
        return nil
      }
      consumed = true
      status.pointee = .haveData
      return buffer
    }
    guard error == nil, let samples = output.int16ChannelData else { return }

    pending.append(
      Data(bytes: samples[0], count: Int(output.frameLength) * MemoryLayout<Int16>.size))
    while pending.count >= chunkBytes {
      let chunk = pending.prefix(chunkBytes)
      pending.removeFirst(chunkBytes)
      onChunk(chunk.base64EncodedString())
    }
  }

  enum CaptureError: Error {
    case unsupportedFormat
  }
}
