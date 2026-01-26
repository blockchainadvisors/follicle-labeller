/**
 * Three-phase adaptive search optimizer for blob detection parameters.
 */

import {
  generateCoarseGrid,
  generateRefinementGrid,
  generateFineTuneGrid,
  type TestConfig,
} from './config';
import {
  runTestsBatched,
  getTopResults,
  sortResults,
  deduplicateConfigs,
  type TestResult,
  type ProgressInfo,
} from './test-runner';

/**
 * Optimization result containing all test data and analysis.
 */
export interface OptimizationResult {
  phases: PhaseResult[];
  allResults: TestResult[];
  bestResult: TestResult;
  totalTests: number;
  totalTimeMs: number;
}

/**
 * Result for a single optimization phase.
 */
export interface PhaseResult {
  name: string;
  description: string;
  testCount: number;
  durationMs: number;
  bestBlobCount: number;
  topResults: TestResult[];
}

/**
 * Progress callback type.
 */
export type ProgressCallback = (progress: ProgressInfo) => void;

/**
 * Run the three-phase optimization process.
 */
export async function runOptimization(
  imageData: ImageData,
  onProgress?: ProgressCallback,
  batchSize: number = 8
): Promise<OptimizationResult> {
  const allResults: TestResult[] = [];
  const phases: PhaseResult[] = [];
  const overallStart = performance.now();

  // Phase 1: Coarse Grid Search
  console.log('\nPhase 1: Coarse Grid Search');
  const coarseConfigs = generateCoarseGrid();
  console.log(`  Testing ${coarseConfigs.length} configurations...`);

  const phase1Start = performance.now();
  const phase1Results = await runTestsBatched(
    coarseConfigs,
    imageData,
    'Phase 1: Coarse Grid',
    batchSize,
    onProgress
  );
  const phase1Duration = performance.now() - phase1Start;

  allResults.push(...phase1Results);
  const phase1Top = getTopResults(phase1Results, 5);

  phases.push({
    name: 'Phase 1',
    description: 'Coarse Grid Search',
    testCount: phase1Results.length,
    durationMs: phase1Duration,
    bestBlobCount: phase1Top[0]?.blobCount ?? 0,
    topResults: phase1Top,
  });

  console.log(`  Best: ${phase1Top[0]?.blobCount ?? 0} blobs | Duration: ${(phase1Duration / 1000).toFixed(1)}s`);

  // Phase 2: Refinement around top 5 configs
  console.log('\nPhase 2: Refinement');
  const refinementConfigs: TestConfig[] = [];
  for (const result of phase1Top) {
    refinementConfigs.push(...generateRefinementGrid(result.config));
  }
  const uniqueRefinementConfigs = deduplicateConfigs(refinementConfigs);
  console.log(`  Testing ${uniqueRefinementConfigs.length} configurations...`);

  const phase2Start = performance.now();
  const phase2Results = await runTestsBatched(
    uniqueRefinementConfigs,
    imageData,
    'Phase 2: Refinement',
    batchSize,
    onProgress
  );
  const phase2Duration = performance.now() - phase2Start;

  allResults.push(...phase2Results);
  const combinedTop = getTopResults([...phase1Results, ...phase2Results], 3);

  phases.push({
    name: 'Phase 2',
    description: 'Refinement',
    testCount: phase2Results.length,
    durationMs: phase2Duration,
    bestBlobCount: combinedTop[0]?.blobCount ?? 0,
    topResults: combinedTop,
  });

  console.log(`  Best: ${combinedTop[0]?.blobCount ?? 0} blobs | Duration: ${(phase2Duration / 1000).toFixed(1)}s`);

  // Phase 3: Fine-tuning around best config
  console.log('\nPhase 3: Fine-Tuning');
  const fineTuneConfigs: TestConfig[] = [];
  for (const result of combinedTop) {
    fineTuneConfigs.push(...generateFineTuneGrid(result.config));
  }
  const uniqueFineTuneConfigs = deduplicateConfigs(fineTuneConfigs);
  console.log(`  Testing ${uniqueFineTuneConfigs.length} configurations...`);

  const phase3Start = performance.now();
  const phase3Results = await runTestsBatched(
    uniqueFineTuneConfigs,
    imageData,
    'Phase 3: Fine-Tuning',
    batchSize,
    onProgress
  );
  const phase3Duration = performance.now() - phase3Start;

  allResults.push(...phase3Results);
  const finalSorted = sortResults(allResults);

  phases.push({
    name: 'Phase 3',
    description: 'Fine-Tuning',
    testCount: phase3Results.length,
    durationMs: phase3Duration,
    bestBlobCount: finalSorted[0]?.blobCount ?? 0,
    topResults: getTopResults(allResults, 3),
  });

  console.log(`  Best: ${finalSorted[0]?.blobCount ?? 0} blobs | Duration: ${(phase3Duration / 1000).toFixed(1)}s`);

  const totalTimeMs = performance.now() - overallStart;

  return {
    phases,
    allResults: finalSorted,
    bestResult: finalSorted[0],
    totalTests: allResults.length,
    totalTimeMs,
  };
}

/**
 * Analyze parameter impact across all results.
 */
export interface ParameterImpact {
  parameter: string;
  values: Array<{
    value: string;
    avgBlobs: number;
    count: number;
    isRecommended: boolean;
  }>;
}

export function analyzeParameterImpact(results: TestResult[]): ParameterImpact[] {
  const impacts: ParameterImpact[] = [];

  // Analyze Gaussian Blur
  const blurStats = new Map<string, { total: number; count: number }>();
  for (const result of results) {
    const key = result.config.useGaussianBlur
      ? `${result.config.gaussianKernelSize}x${result.config.gaussianKernelSize}`
      : 'OFF';
    const current = blurStats.get(key) ?? { total: 0, count: 0 };
    current.total += result.blobCount;
    current.count++;
    blurStats.set(key, current);
  }

  const blurValues = Array.from(blurStats.entries())
    .map(([value, stats]) => ({
      value,
      avgBlobs: Math.round(stats.total / stats.count),
      count: stats.count,
      isRecommended: false,
    }))
    .sort((a, b) => b.avgBlobs - a.avgBlobs);

  if (blurValues.length > 0) {
    blurValues[0].isRecommended = true;
  }

  impacts.push({
    parameter: 'Gaussian Blur',
    values: blurValues,
  });

  // Analyze Morphological Opening
  const morphStats = new Map<string, { total: number; count: number }>();
  for (const result of results) {
    const key = result.config.useMorphOpen
      ? `${result.config.morphKernelSize}x${result.config.morphKernelSize}`
      : 'OFF';
    const current = morphStats.get(key) ?? { total: 0, count: 0 };
    current.total += result.blobCount;
    current.count++;
    morphStats.set(key, current);
  }

  const morphValues = Array.from(morphStats.entries())
    .map(([value, stats]) => ({
      value,
      avgBlobs: Math.round(stats.total / stats.count),
      count: stats.count,
      isRecommended: false,
    }))
    .sort((a, b) => b.avgBlobs - a.avgBlobs);

  if (morphValues.length > 0) {
    morphValues[0].isRecommended = true;
  }

  impacts.push({
    parameter: 'Morphological Opening',
    values: morphValues,
  });

  // Analyze Circularity
  const circStats = new Map<string, { total: number; count: number }>();
  for (const result of results) {
    const key = result.config.minCircularity?.toFixed(2) ?? '0.00';
    const current = circStats.get(key) ?? { total: 0, count: 0 };
    current.total += result.blobCount;
    current.count++;
    circStats.set(key, current);
  }

  const circValues = Array.from(circStats.entries())
    .map(([value, stats]) => ({
      value,
      avgBlobs: Math.round(stats.total / stats.count),
      count: stats.count,
      isRecommended: false,
    }))
    .sort((a, b) => b.avgBlobs - a.avgBlobs);

  if (circValues.length > 0) {
    circValues[0].isRecommended = true;
  }

  impacts.push({
    parameter: 'Circularity',
    values: circValues,
  });

  // Analyze CLAHE
  const claheStats = new Map<string, { total: number; count: number }>();
  for (const result of results) {
    const key = result.config.useCLAHE
      ? result.config.claheClipLimit?.toFixed(1) ?? '3.0'
      : 'OFF';
    const current = claheStats.get(key) ?? { total: 0, count: 0 };
    current.total += result.blobCount;
    current.count++;
    claheStats.set(key, current);
  }

  const claheValues = Array.from(claheStats.entries())
    .map(([value, stats]) => ({
      value,
      avgBlobs: Math.round(stats.total / stats.count),
      count: stats.count,
      isRecommended: false,
    }))
    .sort((a, b) => b.avgBlobs - a.avgBlobs);

  if (claheValues.length > 0) {
    claheValues[0].isRecommended = true;
  }

  impacts.push({
    parameter: 'CLAHE',
    values: claheValues,
  });

  return impacts;
}
