import StityAudio, { type OutputRoute } from '@modules/stity-audio';

export type { OutputRoute };

export const audioOutput = {
  route: (route: OutputRoute) => StityAudio.setOutputRoute(route),
  isHeadsetConnected: () => StityAudio.isHeadsetConnected(),
  /** Returns the phone to its normal audio mode; call when a conversation ends. */
  release: () => StityAudio.releaseAudioSession(),
};
