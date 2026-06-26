'use strict'

// LavaSR enhancer integration + regression tests.
//
// The construct-time tests need no models and always run in CI. They pin the
// behaviours review flagged: enhancer + native chunk streaming is rejected (it
// would otherwise emit un-enhanced 24 kHz mislabeled as 48 kHz), and a
// misconfigured enhancer can't silently become a no-op (an unknown
// enhancer.type throws). The model-backed tests assert the enhanced output is
// reported as 48 kHz for both engines; they are gated on the converted enhancer
// GGUF being staged, and skip cleanly otherwise.
//
// Stage the enhancer GGUF via scripts/convert-lavasr-enhancer-to-gguf.py (from
// the public LavaSRcpp ONNX release) into models/lavasr/lavasr-enhancer.gguf,
// or set LAVASR_ENHANCER_GGUF.

const os = require('bare-os')
const path = require('bare-path')
const test = require('brittle')
const TTSGgml = require('@qvac/tts-ggml')

const {
  ensureLavaSREnhancerGguf,
  ensureSupertonicModel,
  ensureChatterboxModels
} = require('../utils/downloadModel')
const { resolveRefWavPath } = require('../utils/runChatterboxTTS')

const platform = os.platform()
const isMobile = platform === 'ios' || platform === 'android'

function getBaseDir () {
  return isMobile && global.testDir ? global.testDir : '.'
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

// ---- Construct-time regression tests (no models, always run) ----

test('Chatterbox: enhancer + streamChunkTokens is rejected at construction', (t) => {
  t.exception(
    () => new TTSGgml({
      engine: TTSGgml.ENGINE_CHATTERBOX,
      files: {
        t3Model: './models/chatterbox-t3-turbo.gguf',
        s3genModel: './models/chatterbox-s3gen.gguf',
        lavasrEnhancer: './models/lavasr/lavasr-enhancer.gguf'
      },
      streamChunkTokens: 25,
      config: { language: 'en' }
    }),
    /streamChunkTokens/,
    'enhancer + native chunk streaming is rejected (it needs the full utterance)'
  )
})

test('enhancer with an unknown type is rejected at construction', (t) => {
  t.exception(
    () => new TTSGgml({
      engine: TTSGgml.ENGINE_SUPERTONIC,
      files: {
        supertonicModel: './models/supertonic.gguf',
        lavasrEnhancer: './models/lavasr/lavasr-enhancer.gguf'
      },
      enhancer: { type: 'lavasr-typo' },
      config: { language: 'en' }
    }),
    /unknown enhancer\.type/,
    'a typo in enhancer.type throws instead of silently disabling enhancement'
  )
})

test('enhancer block with no GGUF path leaves enhancement off (no throw)', (t) => {
  const model = new TTSGgml({
    engine: TTSGgml.ENGINE_SUPERTONIC,
    files: { supertonicModel: './models/supertonic.gguf' },
    enhancer: { type: 'lavasr' },
    config: { language: 'en' }
  })
  t.absent(
    model._buildTtsParams().lavasrEnhancerPath,
    'no path resolved -> enhancement stays off (the path is the on switch)'
  )
})

// ---- Model-backed tests (gated on staged models) ----

test('Supertonic + LavaSR enhancer reports 48 kHz enhanced output', { timeout: 600000 }, async (t) => {
  const baseDir = getBaseDir()
  const enh = await ensureLavaSREnhancerGguf({ targetDir: path.join(baseDir, 'models', 'lavasr') })
  if (!enh.success) { t.comment('LavaSR enhancer GGUF not staged; skipping.'); t.pass('skipped — no enhancer GGUF'); return }
  const dl = await ensureSupertonicModel({ targetDir: path.join(baseDir, 'models') })
  if (!dl.success) { t.fail('Supertonic GGUF not available — registry fetch failed.'); return }

  const model = new TTSGgml({
    engine: TTSGgml.ENGINE_SUPERTONIC,
    files: { supertonicModel: dl.path, lavasrEnhancer: enh.path },
    voice: 'F1',
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

test('Supertonic without enhancer reports native 44.1 kHz (backward compat)', { timeout: 600000 }, async (t) => {
  const baseDir = getBaseDir()
  const dl = await ensureSupertonicModel({ targetDir: path.join(baseDir, 'models') })
  if (!dl.success) { t.fail('Supertonic GGUF not available — registry fetch failed.'); return }

  const model = new TTSGgml({
    engine: TTSGgml.ENGINE_SUPERTONIC,
    files: { supertonicModel: dl.path },
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

test('Chatterbox + LavaSR enhancer (batch) reports 48 kHz enhanced output', { timeout: 900000 }, async (t) => {
  const baseDir = getBaseDir()
  const enh = await ensureLavaSREnhancerGguf({ targetDir: path.join(baseDir, 'models', 'lavasr') })
  if (!enh.success) { t.comment('LavaSR enhancer GGUF not staged; skipping.'); t.pass('skipped — no enhancer GGUF'); return }
  const modelsDir = path.join(baseDir, 'models')
  const dl = await ensureChatterboxModels({ targetDir: modelsDir })
  if (!dl.success) { t.fail('Chatterbox GGUFs not available — registry fetch failed.'); return }
  const dir = dl.targetDir || modelsDir

  const model = new TTSGgml({
    engine: TTSGgml.ENGINE_CHATTERBOX,
    files: {
      modelDir: dir,
      t3Model: path.join(dir, 'chatterbox-t3-turbo.gguf'),
      s3genModel: path.join(dir, 'chatterbox-s3gen.gguf'),
      lavasrEnhancer: enh.path
    },
    referenceAudio: resolveRefWavPath({}),
    config: { language: 'en', useGPU: false },
    opts: { stats: true }
  })
  await model.load()
  try {
    const r = await runAndCollect(model, 'Chatterbox output neurally upsampled to forty-eight kilohertz.')
    t.is(r.sampleRate, 48000, 'enhanced chatterbox output reports 48 kHz')
    t.ok(r.samples > 0, 'enhanced synthesis produced audio')
  } finally {
    try { await model.unload() } catch (_e) {}
  }
})
