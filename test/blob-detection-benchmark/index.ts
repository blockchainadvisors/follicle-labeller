/**
 * Blob Detection Benchmark - Main Entry Point
 *
 * This script runs an automated benchmark to find optimal blob detection parameters.
 * It tests different combinations through a three-phase optimization process:
 *
 * Phase 1: Coarse Grid Search - Tests sparse combinations across all parameters
 * Phase 2: Refinement - Focuses on promising regions identified in Phase 1
 * Phase 3: Fine-Tuning - Single-parameter variations around the best configuration
 *
 * Usage: npm run test:benchmark [imagePath] [maxDimension]
 *
 * Examples:
 *   npm run test:benchmark                     # Uses default image and maxDimension=2000
 *   npm run test:benchmark ./test/5.jpg 1500   # Custom image and dimension
 */

import * as path from 'path';
import { loadTestImage } from './image-loader';
import { runOptimization } from './optimizer';
import { generateReport, writeReports, printSummary } from './report-generator';
import type { ProgressInfo } from './test-runner';

// Default test image path
const DEFAULT_IMAGE_PATH = './test/5.jpg';

// Default maximum dimension for image downscaling
const DEFAULT_MAX_DIMENSION = 2000;

// Output directory for reports
const OUTPUT_DIR = './test/blob-detection-benchmark/output';

/**
 * Format a progress bar string.
 */
function formatProgressBar(completed: number, total: number, width: number = 20): string {
  const progress = total > 0 ? completed / total : 0;
  const filled = Math.round(progress * width);
  const empty = width - filled;
  const bar = '\u2588'.repeat(filled) + '\u2591'.repeat(empty);
  const percentage = Math.round(progress * 100);
  return `[${bar}] ${completed}/${total} (${percentage}%)`;
}

/**
 * Progress display callback.
 */
function displayProgress(progress: ProgressInfo): void {
  const bar = formatProgressBar(progress.completed, progress.total);
  const eta = progress.estimatedRemainingMs > 0
    ? `ETA: ~${Math.ceil(progress.estimatedRemainingMs / 1000)}s`
    : '';
  const elapsed = `Elapsed: ${(progress.elapsedMs / 1000).toFixed(1)}s`;

  // Clear previous lines and write new progress
  process.stdout.write('\x1b[2K'); // Clear line
  process.stdout.write(`\r${progress.phase}\n`);
  process.stdout.write('\x1b[2K');
  process.stdout.write(`${bar}\n`);

  if (progress.running > 0 && progress.currentConfigs.length > 0) {
    const configPreview = progress.currentConfigs.slice(0, 3).join(', ');
    const more = progress.currentConfigs.length > 3 ? '...' : '';
    process.stdout.write('\x1b[2K');
    process.stdout.write(`Running ${progress.running} tests: ${configPreview}${more}\n`);
  }

  process.stdout.write('\x1b[2K');
  process.stdout.write(`Best so far: ${progress.bestSoFar} blobs | ${elapsed} | ${eta}\n`);

  // Move cursor back up for next update
  if (progress.completed < progress.total) {
    process.stdout.write('\x1b[4A'); // Move up 4 lines
  }
}

/**
 * Main entry point.
 */
async function main(): Promise<void> {
  console.log('=== Blob Detection Benchmark ===\n');

  // Get image path and max dimension from command line or use defaults
  const imagePath = process.argv[2] || DEFAULT_IMAGE_PATH;
  const maxDimension = parseInt(process.argv[3], 10) || DEFAULT_MAX_DIMENSION;

  // 1. Load test image
  console.log(`Loading image: ${imagePath}`);
  console.log(`Max dimension: ${maxDimension}`);
  let image;
  try {
    image = await loadTestImage(imagePath, { maxDimension });

    if (image.wasResized) {
      console.log(`Original size: ${image.originalWidth}x${image.originalHeight}`);
      console.log(`Resized to: ${image.width}x${image.height} (for faster benchmarking)`);
    } else {
      console.log(`Image size: ${image.width}x${image.height}`);
    }
    console.log('');
  } catch (error) {
    console.error(`Failed to load image: ${error}`);
    process.exit(1);
  }

  // 2. Run three-phase optimization
  const result = await runOptimization(image.imageData, displayProgress);

  // 3. Generate and save reports
  console.log('\nSaving reports...');
  const report = generateReport(result, image);
  const outputPath = path.resolve(OUTPUT_DIR);
  writeReports(report, result.allResults, outputPath);

  // 4. Print summary to console
  printSummary(report);

  console.log(`\nReport saved to: ${outputPath}`);
}

// Run main
main().catch(error => {
  console.error('Benchmark failed:', error);
  process.exit(1);
});
