jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock'),
);

jest.mock('@modules/stity-audio', () => ({
  __esModule: true,
  default: {
    addListener: jest.fn(() => ({ remove: jest.fn() })),
    getMicrophonePermissionAsync: jest.fn(async () => ({ granted: true, canAskAgain: true })),
    requestMicrophonePermissionAsync: jest.fn(async () => ({ granted: true, canAskAgain: true })),
    startCaptureAsync: jest.fn(async () => {}),
    stopCapture: jest.fn(),
    setOutputRoute: jest.fn(),
    isHeadsetConnected: jest.fn(() => false),
    releaseAudioSession: jest.fn(),
    speak: jest.fn(),
    stopSpeaking: jest.fn(),
  },
}));
