const { defineConfig } = require('eslint/config');
const expoConfig = require('eslint-config-expo/flat');

module.exports = defineConfig([
  expoConfig,
  {
    ignores: ['dist/*', 'android/*', 'ios/*', '.expo/*'],
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            {
              group: ['../*'],
              message:
                'Import across folders with the "@/" alias instead of relative parent paths.',
            },
          ],
        },
      ],
    },
  },
  {
    files: ['src/app/**', 'src/features/**', 'src/components/**', 'src/stores/**'],
    ignores: ['**/*.test.ts', '**/*.test.tsx'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            {
              group: ['../*'],
              message:
                'Import across folders with the "@/" alias instead of relative parent paths.',
            },
            {
              group: ['@/api/*'],
              message:
                'Import the API from "@/api" only; adapters (*.live.ts, *.mock.ts) stay behind it.',
            },
          ],
        },
      ],
    },
  },
]);
