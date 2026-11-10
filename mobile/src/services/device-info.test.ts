import { Platform } from 'react-native';

import { deviceInfo } from '@/services/device-info';

jest.mock('@/services/audio/audio-output', () => ({
  audioOutput: { isHeadsetConnected: jest.fn(() => true) },
}));

const platform = (fields: object) => {
  for (const [key, value] of Object.entries(fields)) {
    Object.defineProperty(Platform, key, { value, configurable: true });
  }
};

describe('deviceInfo', () => {
  it('names the model when the platform has one', () => {
    platform({ OS: 'android', Version: 35, constants: { Model: 'Pixel 9' } });
    expect(deviceInfo()).toEqual({
      platform: 'android',
      version: '35',
      model: 'Pixel 9',
      headset: true,
    });
  });

  it('works on the web, which has no platform constants', () => {
    platform({ OS: 'web', Version: undefined, constants: undefined });
    expect(deviceInfo()).toMatchObject({ platform: 'web', version: undefined, model: undefined });
  });
});
