'use strict'

// Vulkan-desktop accuracy regression for bci-whispercpp (QVAC-21702).
//
// The QVAC-21702 optimizations — day-projection matmul reorder + threading,
// gaussian-smooth reorder, and the dummy-audio mel skip — are all designed to
// be output-preserving. The C++ unit tests (test_core.cpp) lock the
// preprocessing math bit-for-bit against naive references; this integration
// test is the end-to-end backstop: it runs the real GGML model on the GPU path
// (Vulkan on Linux/Windows desktop) and asserts every fixture still transcribes
// AT LEAST AS WELL as the WER recorded in the fixture manifest — i.e. the
// optimizations did not regress transcription quality.
//
// The existing "[BCI] WER measurement across all test samples" check only
// guards avgWER < 0.5 (a liveness bound). This one is the strict quality gate.
//
// use_gpu=true selects Vulkan on desktop; on a runner with no GPU it falls back
// to CPU (the addon logs the active backend; we also print backendId here). The
// preprocessing under test runs on the CPU either way and is identical, so the
// WER bound holds regardless of backend. Set QVAC_BCI_WER_RELAX=1 to downgrade a
// WER-bound miss to a warning (e.g. an exotic GPU whose kernels round
// differently than the reference machine).

const fs = require('bare-fs')
const os = require('bare-os')
const path = require('bare-path')
const test = require('brittle')
const BCIWhispercpp = require('../../index')
const { getTestPaths, getModelPath, computeWER, detectPlatform } = require('./helpers')
const { flattenSegments } = require('@qvac/bci-whispercpp/util')

const { platform, label } = detectPlatform()
const { manifest, getSamplePath } = getTestPaths()

const MODEL_PATH = (os.hasEnv('WHISPER_MODEL_PATH') ? os.getEnv('WHISPER_MODEL_PATH') : null) ||
  getModelPath('ggml-bci-windowed.bin')
const EMBEDDER_PATH = path.join(path.dirname(MODEL_PATH), 'bci-embedder.bin')
const hasModel = fs.existsSync(MODEL_PATH)

const RELAX = os.hasEnv('QVAC_BCI_WER_RELAX') && os.getEnv('QVAC_BCI_WER_RELAX') === '1'

// The manifest stores bci_wer rounded to 4 decimals, so allow a tiny tolerance
// on the per-sample bound. It is still far below the ~0.1–0.2 WER cost of a
// single extra word error on these short utterances, so any real transcription
// regression trips the guard.
const WER_TOL = 1e-4

function backendIdToName (id) {
  return ({ 0: 'CPU', 1: 'Metal', 2: 'CUDA', 3: 'Vulkan', 4: 'OpenCL', 99: 'other-GPU' })[id] ||
    ('unknown(' + id + ')')
}

function assertNoRegression (t, name, wer, bound) {
  const msg = name + ': WER ' + (wer * 100).toFixed(2) + '% must be <= ' +
    (bound * 100).toFixed(2) + '% (recorded good result)'
  if (wer <= bound + WER_TOL) {
    t.pass(msg)
  } else if (RELAX) {
    t.comment('WARNING (relaxed): ' + msg)
    t.pass(name + ': relaxed')
  } else {
    t.fail(msg)
  }
}

test('[BCI][Vulkan-desktop] transcription accuracy has not regressed (QVAC-21702)',
  { skip: !hasModel, timeout: 180000 }, async (t) => {
    t.ok(manifest.samples.length > 0, 'Manifest must contain at least one sample')
    t.comment('Platform: ' + label + '   Model: ' + MODEL_PATH)

    // Group by day_idx so one loaded context serves all its samples (the day
    // projection is materialized/cached per day). Mirrors the addon WER test.
    const byDay = new Map()
    for (const sample of manifest.samples) {
      const key = typeof sample.day_idx === 'number' ? sample.day_idx : -1
      if (!byDay.has(key)) byDay.set(key, [])
      byDay.get(key).push(sample)
    }

    const results = []
    let backendId = null

    for (const [day, samples] of byDay) {
      const bci = new BCIWhispercpp({
        files: { model: MODEL_PATH, embedder: EMBEDDER_PATH },
        opts: { stats: true }
      }, {
        whisperConfig: { language: 'en', temperature: 0.0 },
        miscConfig: { caption_enabled: false },
        contextParams: { use_gpu: true },
        bciConfig: day >= 0 ? { day_idx: day } : undefined
      })

      try {
        await bci.load()
        for (const sample of samples) {
          const samplePath = getSamplePath(sample.file)
          if (!fs.existsSync(samplePath)) {
            t.fail('Fixture ' + sample.file + ' is missing')
            continue
          }
          const response = await bci.transcribeFile(samplePath)
          const output = await response.await()
          if (backendId === null && response.stats) backendId = response.stats.backendId

          const text = flattenSegments(output).map(s => s.text).join('').trim()
          const wer = computeWER(text, sample.expected_text)
          const bound = typeof sample.bci_wer === 'number' ? sample.bci_wer : 0
          results.push({ file: sample.file, wer, bound })

          t.comment('[' + sample.file + '] expected=' + JSON.stringify(sample.expected_text) +
            ' got=' + JSON.stringify(text) +
            ' WER=' + (wer * 100).toFixed(2) + '% (bound ' + (bound * 100).toFixed(2) + '%)')
        }
      } finally {
        await bci.destroy()
      }
    }

    t.comment('Active backend: backendId=' + backendId + ' (' + backendIdToName(backendId) + ')')
    // When the GPU path actually engaged on desktop, confirm it is the Vulkan
    // (or CUDA) backend this ticket targets — not a silent CPU fallback dressed
    // up as GPU. A genuine CPU fallback (backendId 0, e.g. NO_GPU CI) is allowed
    // here; the dedicated gpu-smoke test owns the "must engage GPU" gate.
    if ((platform === 'linux' || platform === 'win32') && backendId !== null && backendId !== 0) {
      t.ok(backendId === 3 || backendId === 2,
        'desktop GPU path should be Vulkan(3) or CUDA(2), got ' + backendIdToName(backendId))
    }

    t.is(results.length, manifest.samples.length, 'All manifest samples were evaluated')

    for (const r of results) assertNoRegression(t, r.file, r.wer, r.bound)

    const avg = results.reduce((s, r) => s + r.wer, 0) / results.length
    const refAvg = results.reduce((s, r) => s + r.bound, 0) / results.length
    t.comment('Average WER: ' + (avg * 100).toFixed(2) + '%  (recorded reference avg ' +
      (refAvg * 100).toFixed(2) + '%)')
    assertNoRegression(t, 'average', avg, refAvg)
  })
