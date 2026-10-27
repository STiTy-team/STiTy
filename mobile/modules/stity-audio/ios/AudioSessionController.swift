import AVFoundation

enum OutputRoute: String {
  case speaker
  case earpiece
}

final class AudioSessionController {
  private let session = AVAudioSession.sharedInstance()

  private static let headsetPorts: Set<AVAudioSession.Port> = [
    .headphones, .bluetoothA2DP, .bluetoothHFP, .bluetoothLE, .usbAudio,
  ]

  // voiceChat turns on the system echo canceller, so translations played through the speaker
  // are not transcribed again.
  func activate() throws {
    try session.setCategory(
      .playAndRecord,
      mode: .voiceChat,
      options: [.allowBluetooth, .allowBluetoothA2DP]
    )
    try session.setActive(true)
    try route(to: .speaker)
  }

  func route(to route: OutputRoute) throws {
    switch route {
    case .speaker:
      try session.overrideOutputAudioPort(.speaker)
    case .earpiece:
      try session.overrideOutputAudioPort(.none)
    }
  }

  func release() {
    try? session.setActive(false, options: .notifyOthersOnDeactivation)
  }

  func isHeadsetConnected() -> Bool {
    session.currentRoute.outputs.contains { Self.headsetPorts.contains($0.portType) }
  }
}
