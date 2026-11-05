import { getConfig } from 'expo/config';

import { featureSwitchesSchema, resolveFeatures } from './feature-switches';

const SWITCHES = { speech: { ios: true, android: true, web: false } };

describe('resolveFeatures', () => {
  it('picks the switch for the platform', () => {
    expect(resolveFeatures(SWITCHES, 'ios')).toEqual({ speech: true });
    expect(resolveFeatures(SWITCHES, 'android')).toEqual({ speech: true });
    expect(resolveFeatures(SWITCHES, 'web')).toEqual({ speech: false });
  });

  it('treats any other platform as web', () => {
    expect(resolveFeatures(SWITCHES, 'windows')).toEqual({ speech: false });
  });

  it('turns everything on when the build carries no switches', () => {
    expect(resolveFeatures(undefined, 'web')).toEqual({ speech: true });
  });

  it('rejects a malformed config', () => {
    expect(() => resolveFeatures({ speech: { ios: 'yes' } }, 'ios')).toThrow();
  });
});

describe('configs/application.yml', () => {
  it('reaches the app config as valid switches', () => {
    const { exp } = getConfig(process.cwd(), { skipSDKVersionRequirement: true });
    expect(() => featureSwitchesSchema.parse(exp.extra?.features)).not.toThrow();
  });
});
