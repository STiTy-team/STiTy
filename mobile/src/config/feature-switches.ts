import { z } from 'zod';

const perPlatform = z.object({ ios: z.boolean(), android: z.boolean(), web: z.boolean() });

/** `features` in `configs/application.yml`: each feature switched on or off per platform. */
export const featureSwitchesSchema = z.object({ speech: perPlatform });

export type FeatureSwitches = z.infer<typeof featureSwitchesSchema>;

export type Features = Record<keyof FeatureSwitches, boolean>;

type SwitchedPlatform = keyof z.infer<typeof perPlatform>;

const ALL_ON: FeatureSwitches = { speech: { ios: true, android: true, web: true } };

const switchedPlatform = (platform: string): SwitchedPlatform =>
  platform === 'ios' || platform === 'android' ? platform : 'web';

/** The switches for one platform. Missing means a local run (Expo Go, tests): everything on. */
export const resolveFeatures = (value: unknown, platform: string): Features => {
  const switches = value === undefined ? ALL_ON : featureSwitchesSchema.parse(value);
  const key = switchedPlatform(platform);
  return Object.fromEntries(
    Object.entries(switches).map(([feature, platforms]) => [feature, platforms[key]]),
  ) as Features;
};
