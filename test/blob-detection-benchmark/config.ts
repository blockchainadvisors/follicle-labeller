/**
 * Parameter grid definitions for blob detection benchmarking.
 */

import type { BlobDetectionOptions } from '../../src/types';

/**
 * Type for the partial detection options we test.
 */
export type TestConfig = Partial<BlobDetectionOptions>;

/**
 * Focused grid - tests the most impactful parameters.
 */
export const COARSE_GRID = {
  // Size constraints (most impactful on blob count)
  minSize: [3, 6, 10],
  maxSize: [100, 200, 400],

  // Soft-NMS - can significantly reduce blob count
  useSoftNMS: [true, false],

  // Circularity filter
  minCircularity: [0, 0.2],

  // CLAHE preprocessing
  claheClipLimit: [0, 2.5, 4.0],  // 0 means CLAHE off

  // Preprocessing flags (simplified)
  preprocessing: ['none', 'blur', 'morph', 'both'] as const,
};

/**
 * Generate coarse grid configurations (~200-300 configs).
 */
export function generateCoarseGrid(): TestConfig[] {
  const configs: TestConfig[] = [];

  for (const minSize of COARSE_GRID.minSize) {
    for (const maxSize of COARSE_GRID.maxSize) {
      if (minSize >= maxSize * 0.3) continue;

      for (const useSoftNMS of COARSE_GRID.useSoftNMS) {
        for (const minCircularity of COARSE_GRID.minCircularity) {
          for (const claheClipLimit of COARSE_GRID.claheClipLimit) {
            for (const prep of COARSE_GRID.preprocessing) {
              const config: TestConfig = {
                minCircularity,
                useSoftNMS,
                minWidth: minSize,
                maxWidth: maxSize,
                minHeight: minSize,
                maxHeight: maxSize,
                useCLAHE: claheClipLimit > 0,
              };

              if (claheClipLimit > 0) {
                config.claheClipLimit = claheClipLimit;
              }

              // Set preprocessing based on combo
              switch (prep) {
                case 'none':
                  config.useGaussianBlur = false;
                  config.useMorphOpen = false;
                  break;
                case 'blur':
                  config.useGaussianBlur = true;
                  config.gaussianKernelSize = 5;
                  config.useMorphOpen = false;
                  break;
                case 'morph':
                  config.useGaussianBlur = false;
                  config.useMorphOpen = true;
                  config.morphKernelSize = 3;
                  break;
                case 'both':
                  config.useGaussianBlur = true;
                  config.gaussianKernelSize = 5;
                  config.useMorphOpen = true;
                  config.morphKernelSize = 3;
                  break;
              }

              configs.push(config);
            }
          }
        }
      }
    }
  }

  return configs;
}

/**
 * Generate refinement grid around a base configuration.
 */
export function generateRefinementGrid(baseConfig: TestConfig): TestConfig[] {
  const configs: TestConfig[] = [];

  // Refine size constraints
  if (baseConfig.minWidth !== undefined) {
    const base = baseConfig.minWidth;
    for (const delta of [-2, 2]) {
      const value = base + delta;
      if (value >= 2) {
        configs.push({ ...baseConfig, minWidth: value, minHeight: value });
      }
    }
  }

  if (baseConfig.maxWidth !== undefined) {
    const base = baseConfig.maxWidth;
    for (const delta of [-50, 50]) {
      const value = base + delta;
      if (value > (baseConfig.minWidth ?? 5) * 3) {
        configs.push({ ...baseConfig, maxWidth: value, maxHeight: value });
      }
    }
  }

  // Refine circularity
  if (baseConfig.minCircularity !== undefined) {
    const base = baseConfig.minCircularity;
    for (const delta of [-0.1, 0.1]) {
      const value = Math.round((base + delta) * 100) / 100;
      if (value >= 0 && value <= 0.5) {
        configs.push({ ...baseConfig, minCircularity: value });
      }
    }
  }

  // Refine CLAHE
  if (baseConfig.useCLAHE && baseConfig.claheClipLimit) {
    const base = baseConfig.claheClipLimit;
    for (const delta of [-0.5, 0.5]) {
      const value = base + delta;
      if (value >= 1.5 && value <= 5) {
        configs.push({ ...baseConfig, claheClipLimit: value });
      }
    }
  }

  // Try toggling Soft-NMS
  configs.push({ ...baseConfig, useSoftNMS: !baseConfig.useSoftNMS });

  return configs;
}

/**
 * Generate fine-tuning variations around a configuration.
 */
export function generateFineTuneGrid(baseConfig: TestConfig): TestConfig[] {
  const configs: TestConfig[] = [];

  // Fine-tune min size
  if (baseConfig.minWidth !== undefined) {
    const base = baseConfig.minWidth;
    for (const value of [base - 1, base + 1].filter(v => v >= 2)) {
      configs.push({ ...baseConfig, minWidth: value, minHeight: value });
    }
  }

  // Fine-tune max size
  if (baseConfig.maxWidth !== undefined) {
    const base = baseConfig.maxWidth;
    for (const value of [base - 20, base + 20].filter(v => v > (baseConfig.minWidth ?? 5) * 3)) {
      configs.push({ ...baseConfig, maxWidth: value, maxHeight: value });
    }
  }

  // Fine-tune circularity
  if (baseConfig.minCircularity !== undefined) {
    const base = baseConfig.minCircularity;
    for (const delta of [-0.05, 0.05]) {
      const value = Math.round((base + delta) * 100) / 100;
      if (value >= 0 && value <= 0.5 && value !== base) {
        configs.push({ ...baseConfig, minCircularity: value });
      }
    }
  }

  return configs;
}

/**
 * Get a short description of a configuration.
 */
export function describeConfig(config: TestConfig): string {
  const parts: string[] = [];

  if (config.useGaussianBlur) {
    parts.push(`blur=${config.gaussianKernelSize}`);
  }

  if (config.useMorphOpen) {
    parts.push(`morph=${config.morphKernelSize}`);
  }

  parts.push(`circ=${(config.minCircularity ?? 0).toFixed(1)}`);

  if (config.useCLAHE) {
    parts.push(`clahe=${config.claheClipLimit?.toFixed(1)}`);
  }

  parts.push(`sz=${config.minWidth}-${config.maxWidth}`);

  if (config.useSoftNMS === false) {
    parts.push('noNMS');
  }

  return parts.join('+');
}
