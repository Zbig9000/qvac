'use strict'

// LavaSR enhancer integration test (QVAC-16579): a Supertonic model loaded
// with an `enhancer` block reports 48 kHz output (vs the native 44.1 kHz),
// exercising the addon -> tts_cpp::lavasr::Enhancer path end-to-end on the
// real ggml backend.
//
// Gated on the converted enhancer GGUF being present. It is a converted
// artifact (not yet in the model registry), so when missing the test skips
// cleanly. Produce it with:
//   python qvac-ext-lib-whisper.cpp/tts-cpp/scripts/convert-lavasr-enhancer-to-gguf.py \
//     --backbone enhancer_backbone.onnx --spec-head enhancer_spec_head.onnx \
//     --out models/lavasr/lavasr-enhancer.gguf --ftype f16
// or set LAVASR_ENHANCER_GGUF to its path.

const os = require('bare-os')
const fs = require('bare-fs')
const path = require('bare-path')
const proc = require('bare-process')
const test = require('brittle')

const { ensureSupertonicModel } = require('../utils/downloadModel')

const platform = os.platform()
const isMobile = platform === 'ios' || platform === 'android'

function getBaseDir () {
  return isMobile && global.testDir ? global.testDir : '.'
}

function findEnhancerGguf (baseDir) {
  const candidates = [
    proc.env && proc.env.LAVASR_ENHANCER_GGUF,
    path.join(baseDir, 'models', 'lavasr', 'lavasr-enhancer.gguf'),
    path.join(baseDir, 'models', 'lavasr-enhancer.gguf')
  ].filter(Boolean)
  for (const c of candidates) {
    try { if (fs.existsSync(c)) return c } catch (_e) {}
  }
  return null
}

async function runAndCollect (model, text) {
  let samples = 0
  let sampleRate = null
  const response = await model.run({ input: text, type: 'text' })
  await response
    .onUpdate(d => {
      if (d && d.outputArray) samples += d.outputArray.length
      if (d && d.sampleRate) sampleRate = d.sampleRate
    })
    .await()
  return { samples, sampleRate, stats: response.stats || null }
}

test('Supertonic + LavaSR enhancer reports 48 kHz enhanced output', { timeout: 600000 }, async (t) => {
  const baseDir = getBaseDir()
  const enhancerPath = findEnhancerGguf(baseDir)
  if (!enhancerPath) {
    t.comment('LavaSR enhancer GGUF not found (set LAVASR_ENHANCER_GGUF or stage models/lavasr/lavasr-enhancer.gguf). Skipping.')
    t.pass('skipped — no enhancer GGUF')
    return
  }
  const download = await ensureSupertonicModel({ targetDir: path.join(baseDir, 'models') })
  if (!download.success) { t.fail('Supertonic GGUF not available — registry fetch failed.'); return }

  const TTSGgml = require('@qvac/tts-ggml')
  const model = new TTSGgml({
    engine: TTSGgml.ENGINE_SUPERTONIC,
    files: { supertonicModel: download.path, lavasrEnhancer: enhancerPath },
    voice: 'F1',
    enhancer: { type: 'lavasr', enhance: true },
    config: { language: 'en', useGPU: false },
    opts: { stats: true }
  })
  await model.load()
  try {
    const r = await runAndCollect(model, 'LavaSR neural enhancement upsamples this to forty-eight kilohertz.')
    t.is(r.sampleRate, 48000, 'enhanced supertonic output reports 48 kHz')
    t.ok(r.samples > 0, 'enhanced synthesis produced audio')
  } finally {
    try { await model.unload() } catch (_e) {}
  }
})

test('Supertonic without enhancer still reports native 44.1 kHz (backward compat)', { timeout: 600000 }, async (t) => {
  const baseDir = getBaseDir()
  const download = await ensureSupertonicModel({ targetDir: path.join(baseDir, 'models') })
  if (!download.success) { t.fail('Supertonic GGUF not available — registry fetch failed.'); return }

  const TTSGgml = require('@qvac/tts-ggml')
  const model = new TTSGgml({
    engine: TTSGgml.ENGINE_SUPERTONIC,
    files: { supertonicModel: download.path },
    voice: 'F1',
    config: { language: 'en', useGPU: false },
    opts: { stats: true }
  })
  await model.load()
  try {
    const r = await runAndCollect(model, 'No enhancement here, just the native engine output.')
    t.is(r.sampleRate, 44100, 'un-enhanced supertonic reports 44.1 kHz')
    t.ok(r.samples > 0, 'synthesis produced audio')
  } finally {
    try { await model.unload() } catch (_e) {}
  }
})
