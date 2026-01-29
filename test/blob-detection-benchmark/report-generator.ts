/**
 * Report generator for blob detection benchmark results.
 */

import * as fs from 'fs';
import * as path from 'path';
import type { LoadedImage } from './image-loader';
import type { OptimizationResult, ParameterImpact } from './optimizer';
import { analyzeParameterImpact } from './optimizer';
import type { TestResult } from './test-runner';

/**
 * Full benchmark report structure.
 */
export interface BenchmarkReport {
  generatedAt: string;
  image: {
    filePath: string;
    fileName: string;
    width: number;
    height: number;
    originalWidth: number;
    originalHeight: number;
    wasResized: boolean;
  };
  summary: {
    totalTests: number;
    totalTimeMs: number;
    totalTimeFormatted: string;
    bestBlobCount: number;
  };
  bestConfiguration: {
    config: Record<string, unknown>;
    blobCount: number;
    executionTimeMs: number;
  };
  topConfigurations: Array<{
    rank: number;
    blobCount: number;
    executionTimeMs: number;
    config: Record<string, unknown>;
    description: string;
  }>;
  parameterImpact: ParameterImpact[];
  phases: Array<{
    name: string;
    description: string;
    testCount: number;
    durationMs: number;
    bestBlobCount: number;
  }>;
}

/**
 * Generate the complete benchmark report.
 */
export function generateReport(
  result: OptimizationResult,
  image: LoadedImage
): BenchmarkReport {
  const parameterImpact = analyzeParameterImpact(result.allResults);

  return {
    generatedAt: new Date().toISOString(),
    image: {
      filePath: image.filePath,
      fileName: path.basename(image.filePath),
      width: image.width,
      height: image.height,
      originalWidth: image.originalWidth,
      originalHeight: image.originalHeight,
      wasResized: image.wasResized,
    },
    summary: {
      totalTests: result.totalTests,
      totalTimeMs: result.totalTimeMs,
      totalTimeFormatted: formatDuration(result.totalTimeMs),
      bestBlobCount: result.bestResult?.blobCount ?? 0,
    },
    bestConfiguration: {
      config: result.bestResult?.config ?? {},
      blobCount: result.bestResult?.blobCount ?? 0,
      executionTimeMs: result.bestResult?.executionTimeMs ?? 0,
    },
    topConfigurations: result.allResults.slice(0, 10).map((r, i) => ({
      rank: i + 1,
      blobCount: r.blobCount,
      executionTimeMs: r.executionTimeMs,
      config: r.config,
      description: r.configDescription,
    })),
    parameterImpact,
    phases: result.phases.map(p => ({
      name: p.name,
      description: p.description,
      testCount: p.testCount,
      durationMs: p.durationMs,
      bestBlobCount: p.bestBlobCount,
    })),
  };
}

/**
 * Format duration in milliseconds to human readable string.
 */
function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  }
  return `${(ms / 1000).toFixed(1)}s`;
}

/**
 * Generate Markdown report content.
 */
export function generateMarkdownReport(report: BenchmarkReport): string {
  const lines: string[] = [];

  lines.push('# Blob Detection Benchmark Report');
  lines.push('');
  lines.push(`**Generated:** ${report.generatedAt}`);

  if (report.image.wasResized) {
    lines.push(`**Test Image:** ${report.image.fileName} (${report.image.originalWidth}x${report.image.originalHeight} -> ${report.image.width}x${report.image.height})`);
  } else {
    lines.push(`**Test Image:** ${report.image.fileName} (${report.image.width}x${report.image.height})`);
  }
  lines.push('');

  // Summary
  lines.push('## Summary');
  lines.push('');
  lines.push('| Metric | Value |');
  lines.push('|--------|-------|');
  lines.push(`| Total Tests | ${report.summary.totalTests} |`);
  lines.push(`| Duration | ${report.summary.totalTimeFormatted} |`);
  lines.push(`| Best Blob Count | ${report.summary.bestBlobCount} |`);
  lines.push('');

  // Best Configuration
  lines.push('## Best Configuration');
  lines.push('');
  lines.push('| Parameter | Value |');
  lines.push('|-----------|-------|');

  const best = report.bestConfiguration.config;
  if (best.useGaussianBlur) {
    lines.push(`| Gaussian Blur | ON (${best.gaussianKernelSize}x${best.gaussianKernelSize}) |`);
  } else {
    lines.push('| Gaussian Blur | OFF |');
  }

  if (best.useMorphOpen) {
    lines.push(`| Morphological Opening | ON (${best.morphKernelSize}x${best.morphKernelSize}) |`);
  } else {
    lines.push('| Morphological Opening | OFF |');
  }

  lines.push(`| Circularity Filter | ${(best.minCircularity as number)?.toFixed(2) ?? '0.00'} |`);

  if (best.useCLAHE) {
    lines.push(`| CLAHE | ON (${(best.claheClipLimit as number)?.toFixed(1) ?? '3.0'}) |`);
  } else {
    lines.push('| CLAHE | OFF |');
  }
  lines.push('');

  // Parameter Impact Analysis
  lines.push('## Parameter Impact Analysis');
  lines.push('');

  for (const impact of report.parameterImpact) {
    lines.push(`### ${impact.parameter}`);
    lines.push('');
    lines.push('| Setting | Avg Blobs | Samples | Recommendation |');
    lines.push('|---------|-----------|---------|----------------|');

    for (const value of impact.values) {
      const rec = value.isRecommended ? 'RECOMMENDED' : '';
      lines.push(`| ${value.value} | ${value.avgBlobs} | ${value.count} | ${rec} |`);
    }
    lines.push('');
  }

  // Top 10 Configurations
  lines.push('## Top 10 Configurations');
  lines.push('');
  lines.push('| Rank | Blobs | Time (ms) | Configuration |');
  lines.push('|------|-------|-----------|---------------|');

  for (const top of report.topConfigurations) {
    lines.push(`| ${top.rank} | ${top.blobCount} | ${top.executionTimeMs.toFixed(1)} | ${top.description} |`);
  }
  lines.push('');

  // Phase Progress
  lines.push('## Phase Progress');
  lines.push('');
  lines.push('| Phase | Tests | Duration | Best Blobs |');
  lines.push('|-------|-------|----------|------------|');

  for (const phase of report.phases) {
    lines.push(`| ${phase.name}: ${phase.description} | ${phase.testCount} | ${formatDuration(phase.durationMs)} | ${phase.bestBlobCount} |`);
  }
  lines.push('');

  return lines.join('\n');
}

/**
 * Generate CSV report content.
 */
export function generateCSVReport(results: TestResult[]): string {
  const headers = [
    'rank',
    'blob_count',
    'execution_time_ms',
    'use_gaussian_blur',
    'gaussian_kernel_size',
    'use_morph_open',
    'morph_kernel_size',
    'min_circularity',
    'use_clahe',
    'clahe_clip_limit',
    'description',
  ];

  const lines: string[] = [headers.join(',')];

  results.forEach((result, index) => {
    const row = [
      index + 1,
      result.blobCount,
      result.executionTimeMs.toFixed(2),
      result.config.useGaussianBlur ? 'true' : 'false',
      result.config.gaussianKernelSize ?? '',
      result.config.useMorphOpen ? 'true' : 'false',
      result.config.morphKernelSize ?? '',
      result.config.minCircularity?.toFixed(2) ?? '',
      result.config.useCLAHE ? 'true' : 'false',
      result.config.claheClipLimit?.toFixed(1) ?? '',
      `"${result.configDescription}"`,
    ];
    lines.push(row.join(','));
  });

  return lines.join('\n');
}

/**
 * Write all reports to the output directory.
 */
export function writeReports(
  report: BenchmarkReport,
  allResults: TestResult[],
  outputDir: string
): void {
  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Write JSON report
  const jsonPath = path.join(outputDir, 'benchmark-report.json');
  fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2), 'utf-8');
  console.log(`  JSON report: ${jsonPath}`);

  // Write Markdown report
  const mdPath = path.join(outputDir, 'benchmark-summary.md');
  fs.writeFileSync(mdPath, generateMarkdownReport(report), 'utf-8');
  console.log(`  Markdown report: ${mdPath}`);

  // Write CSV report
  const csvPath = path.join(outputDir, 'benchmark-results.csv');
  fs.writeFileSync(csvPath, generateCSVReport(allResults), 'utf-8');
  console.log(`  CSV report: ${csvPath}`);
}

/**
 * Print summary to console.
 */
export function printSummary(report: BenchmarkReport): void {
  console.log('\n=== Results ===');
  console.log(`Total tests: ${report.summary.totalTests}`);
  console.log(`Total time: ${report.summary.totalTimeFormatted}`);
  console.log(`Best blob count: ${report.summary.bestBlobCount}`);
  console.log('\nBest configuration:');
  console.log(JSON.stringify(report.bestConfiguration.config, null, 2));
}
