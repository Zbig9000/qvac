'use strict'

/**
 * Desktop harness for the ACE-Step Real-Time Factor benchmark.
 *
 * The measurement itself lives in test/utils/benchmark-runner.js, shared with
 * the on-device lane (test/mobile/test.cjs). This file only wires that runner
 * into brittle: it writes the on-disk artifact the desktop CI job uploads,
 * emits the canonical log record, and asserts the result is usable.
 *
 * One invocation benchmarks ONE (ditVariant, useGPU) combination; sweeping
 * several is scripts/run-rtf-benchmark-matrix.js's job. See the runner module
 * for the full list of QVAC_AUDIOGEN_GGML_BENCHMARK_* environment variables.
 */

const test = require('brittle')
const {
  readBenchmarkSettings,
  runRtfBenchmark,
  writeRtfArtifact,
  emitCanonicalReport
} = require('../utils/benchmark-runner')

// A hung generation must not hold the whole matrix; a single ACE-Step render is
// minutes at worst, so this is a generous ceiling rather than a target.
const SUITE_TIMEOUT_MS = 3600000

function assertResults(t, settings, summary, runs) {
  t.is(runs.length, settings.numRuns, `completed ${settings.numRuns} measured run(s)`)
  t.ok(summary.rtf.mean > 0, 'mean RTF is positive')
  t.ok(
    runs.every((run) => run.sampleCount > 0),
    'every run rendered audio'
  )
  t.ok(summary.memory.peakRssMb > 0, 'peak RSS is positive')
  t.ok(summary.memory.peakRssMb >= summary.memory.avgRssMb, 'peak RSS is at least average RSS')

  if (settings.rtfUpperBound !== null) {
    t.ok(
      summary.rtf.mean <= settings.rtfUpperBound,
      `mean RTF ${summary.rtf.mean.toFixed(4)} is within ${settings.rtfUpperBound}`
    )
  }
}

test('RTF benchmark: ACE-Step music generation', { timeout: SUITE_TIMEOUT_MS }, async (t) => {
  const settings = readBenchmarkSettings()
  const result = await runRtfBenchmark(settings)
  t.teardown(() => result.destroy())

  writeRtfArtifact(settings, result.report)
  emitCanonicalReport(settings, result.summary, result.backend)
  assertResults(t, settings, result.summary, result.runs)
})
