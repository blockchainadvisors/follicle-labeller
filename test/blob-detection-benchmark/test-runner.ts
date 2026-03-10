/**
 * Parallel test execution for blob detection benchmark.
 */

import { detectBlobsSync } from './blob-detector-sync';
import { applyCLAHEToImageData } from './clahe-processor';
import type { BlobDetectionOptions, DetectedBlob } from '../../src/types';
import { describeConfig, type TestConfig } from './config';

/**
 * Result from a single test run.
 */
export interface TestResult {
  config: TestConfig;
  configDescription: string;
  blobCount: number;
  executionTimeMs: number;
  blobs?: DetectedBlob[]; // Optional, can be large
}

/**
 * Progress information for tracking test execution.
 */
export interface ProgressInfo {
  phase: string;
  completed: number;
  total: number;
  running: number;
  bestSoFar: number;
  currentConfigs: string[];
  elapsedMs: number;
  estimatedRemainingMs: number;
}

/**
 * Run a single test with the given configuration.
 */
export function runSingleTest(
  config: TestConfig,
  imageData: ImageData
): TestResult {
  let processedImage = imageData;

  // Apply CLAHE if enabled (must be done before detectBlobsSync)
  if (config.useCLAHE) {
    processedImage = applyCLAHEToImageData(imageData, {
      clipLimit: config.claheClipLimit ?? 3.0,
      tileGridSize: config.claheTileSize ?? 8,
    });
  }

  const startTime = performance.now();
  const blobs = detectBlobsSync(processedImage, config);
  const duration = performance.now() - startTime;

  return {
    config,
    configDescription: describeConfig(config),
    blobCount: blobs.length,
    executionTimeMs: duration,
  };
}

/**
 * Run tests in batches with progress reporting.
 */
export async function runTestsBatched(
  configs: TestConfig[],
  imageData: ImageData,
  phase: string,
  batchSize: number = 8,
  onProgress?: (progress: ProgressInfo) => void
): Promise<TestResult[]> {
  const results: TestResult[] = [];
  const startTime = performance.now();
  let bestSoFar = 0;

  // Process in batches
  for (let i = 0; i < configs.length; i += batchSize) {
    const batch = configs.slice(i, Math.min(i + batchSize, configs.length));
    const currentConfigs = batch.map(c => describeConfig(c));

    // Report progress before running batch
    if (onProgress) {
      const elapsedMs = performance.now() - startTime;
      const avgTimePerTest = results.length > 0 ? elapsedMs / results.length : 50;
      const remainingTests = configs.length - i;
      const estimatedRemainingMs = avgTimePerTest * remainingTests;

      onProgress({
        phase,
        completed: results.length,
        total: configs.length,
        running: batch.length,
        bestSoFar,
        currentConfigs,
        elapsedMs,
        estimatedRemainingMs,
      });
    }

    // Run batch in parallel using Promise.all
    const batchPromises = batch.map(config =>
      Promise.resolve(runSingleTest(config, imageData))
    );

    const batchResults = await Promise.all(batchPromises);

    // Update best result
    for (const result of batchResults) {
      if (result.blobCount > bestSoFar) {
        bestSoFar = result.blobCount;
      }
      results.push(result);
    }
  }

  // Final progress report
  if (onProgress) {
    const elapsedMs = performance.now() - startTime;
    onProgress({
      phase,
      completed: configs.length,
      total: configs.length,
      running: 0,
      bestSoFar,
      currentConfigs: [],
      elapsedMs,
      estimatedRemainingMs: 0,
    });
  }

  return results;
}

/**
 * Sort results by blob count (descending), then by execution time (ascending).
 */
export function sortResults(results: TestResult[]): TestResult[] {
  return [...results].sort((a, b) => {
    // Primary: more blobs is better
    if (b.blobCount !== a.blobCount) {
      return b.blobCount - a.blobCount;
    }
    // Tiebreaker: faster is better
    return a.executionTimeMs - b.executionTimeMs;
  });
}

/**
 * Get top N results.
 */
export function getTopResults(results: TestResult[], n: number): TestResult[] {
  return sortResults(results).slice(0, n);
}

/**
 * Deduplicate configs by their description.
 */
export function deduplicateConfigs(configs: TestConfig[]): TestConfig[] {
  const seen = new Set<string>();
  return configs.filter(config => {
    const desc = describeConfig(config);
    if (seen.has(desc)) {
      return false;
    }
    seen.add(desc);
    return true;
  });
}
