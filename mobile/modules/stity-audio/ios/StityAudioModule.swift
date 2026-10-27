import AVFoundation
import ExpoModulesCore

private let eventAudioChunk = "onAudioChunk"
private let eventCaptureError = "onCaptureError"
private let eventSpeechFinished = "onSpeechFinished"
private let eventSpeechFailed = "onSpeechFailed"

enum OutputRouteArgument: String, Enumerable {
  case speaker
  case earpiece
}

final class CaptureStartException: GenericException<String> {
  override var code: String { "ERR_CAPTURE_START" }
  override var reason: String { "Could not start the microphone: \(param)" }
}

public class StityAudioModule: Module {
  private let session = AudioSessionController()
  private lazy var capture = AudioCapture { [weak self] data in
    self?.sendEvent(eventAudioChunk, ["data": data])
  }
  private lazy var speech = SpeechSynthesizer(
    onFinished: { [weak self] id in
      self?.sendEvent(eventSpeechFinished, ["utteranceId": id])
    },
    onFailed: { [weak self] id, reason in
      self?.sendEvent(eventSpeechFailed, ["utteranceId": id, "reason": reason])
    }
  )
  private var interruptionObserver: NSObjectProtocol?

  public func definition() -> ModuleDefinition {
    Name("StityAudio")

    Events(eventAudioChunk, eventCaptureError, eventSpeechFinished, eventSpeechFailed)

    OnCreate {
      self.interruptionObserver = NotificationCenter.default.addObserver(
        forName: AVAudioSession.interruptionNotification,
        object: nil,
        queue: .main
      ) { [weak self] notification in
        self?.handleInterruption(notification)
      }
    }

    OnDestroy {
      if let observer = self.interruptionObserver {
        NotificationCenter.default.removeObserver(observer)
      }
      self.capture.stop()
      self.speech.stop()
      self.session.release()
    }

    AsyncFunction("getMicrophonePermissionAsync") { () -> [String: Any] in
      Self.permissionResponse(Self.currentPermission())
    }

    AsyncFunction("requestMicrophonePermissionAsync") { (promise: Promise) in
      let resolve: (Bool) -> Void = { _ in
        promise.resolve(Self.permissionResponse(Self.currentPermission()))
      }
      if #available(iOS 17.0, *) {
        AVAudioApplication.requestRecordPermission(completionHandler: resolve)
      } else {
        AVAudioSession.sharedInstance().requestRecordPermission(resolve)
      }
    }

    AsyncFunction("startCaptureAsync") { (sampleRate: Double, chunkSamples: Int) in
      do {
        try self.session.activate()
        try self.capture.start(sampleRate: sampleRate, chunkSamples: chunkSamples)
      } catch {
        throw CaptureStartException(error.localizedDescription)
      }
    }

    Function("stopCapture") {
      self.capture.stop()
    }

    Function("setOutputRoute") { (route: OutputRouteArgument) in
      try self.session.route(to: route == .speaker ? .speaker : .earpiece)
    }

    Function("isHeadsetConnected") { () -> Bool in
      self.session.isHeadsetConnected()
    }

    Function("releaseAudioSession") {
      self.session.release()
    }

    Function("speak") { (utteranceId: String, text: String, languageTag: String, rate: Float) in
      self.speech.speak(utteranceId: utteranceId, text: text, languageTag: languageTag, rate: rate)
    }

    Function("stopSpeaking") {
      self.speech.stop()
    }
  }

  private func handleInterruption(_ notification: Notification) {
    guard
      let raw = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
      AVAudioSession.InterruptionType(rawValue: raw) == .began,
      capture.isRunning
    else { return }
    capture.stop()
    sendEvent(eventCaptureError, ["message": "interrupted"])
  }

  private enum PermissionStatus: String {
    case granted
    case denied
    case undetermined
  }

  private static func currentPermission() -> PermissionStatus {
    if #available(iOS 17.0, *) {
      switch AVAudioApplication.shared.recordPermission {
      case .granted: return .granted
      case .denied: return .denied
      default: return .undetermined
      }
    }
    switch AVAudioSession.sharedInstance().recordPermission {
    case .granted: return .granted
    case .denied: return .denied
    default: return .undetermined
    }
  }

  private static func permissionResponse(_ status: PermissionStatus) -> [String: Any] {
    [
      "status": status.rawValue,
      "granted": status == .granted,
      "canAskAgain": status == .undetermined,
      "expires": "never",
    ]
  }
}
