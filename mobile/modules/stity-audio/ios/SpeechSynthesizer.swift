import AVFoundation

final class SpeechSynthesizer: NSObject, AVSpeechSynthesizerDelegate {
  private let synthesizer = AVSpeechSynthesizer()
  private let lock = NSLock()
  private var utteranceIds: [ObjectIdentifier: String] = [:]
  private let onFinished: (String) -> Void
  private let onFailed: (String, String) -> Void

  init(onFinished: @escaping (String) -> Void, onFailed: @escaping (String, String) -> Void) {
    self.onFinished = onFinished
    self.onFailed = onFailed
    super.init()
    synthesizer.usesApplicationAudioSession = true
    synthesizer.delegate = self
  }

  func speak(utteranceId: String, text: String, languageTag: String, rate: Float) {
    guard let voice = AVSpeechSynthesisVoice(language: languageTag) else {
      onFailed(utteranceId, "unsupported-language")
      return
    }
    let utterance = AVSpeechUtterance(string: text)
    utterance.voice = voice
    utterance.rate = min(
      AVSpeechUtteranceMaximumSpeechRate,
      max(AVSpeechUtteranceMinimumSpeechRate, AVSpeechUtteranceDefaultSpeechRate * rate)
    )
    lock.withLock { utteranceIds[ObjectIdentifier(utterance)] = utteranceId }
    synthesizer.speak(utterance)
  }

  func stop() {
    synthesizer.stopSpeaking(at: .immediate)
  }

  func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
    if let id = takeId(of: utterance) {
      onFinished(id)
    }
  }

  func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
    if let id = takeId(of: utterance) {
      onFailed(id, "cancelled")
    }
  }

  private func takeId(of utterance: AVSpeechUtterance) -> String? {
    lock.withLock { utteranceIds.removeValue(forKey: ObjectIdentifier(utterance)) }
  }
}
