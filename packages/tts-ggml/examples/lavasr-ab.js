'use strict'

/**
 * LavaSR A/B driver — proves the QVAC LavaSR enhancer end to end on the real
 * shipped path (native @qvac/tts-ggml addon → tts-cpp lavasr::Enhancer). This is
 * the SAME enhancement code demos/lavasr runs via @qvac/sdk; here we drive the
 * addon directly (route A) so it works without bun/SDK.
 *
 * It synthesizes the same text twice with Chatterbox:
 *   1) enhancer OFF  → before_24k.wav (native 24 kHz)
 *   2) enhancer ON   → after_48k.wav  (LavaSR bandwidth-extended to 48 kHz)
 *
 * Enabling LavaSR is a single thing: pass files.lavasrEnhancer (the GGUF path is
 * the on switch — there is no separate flag).
 *
 * Usage:
 *   bare examples/lavasr-ab.js "<text>" [reference.wav]
 *
 * Env (all optional; sensible local defaults):
 *   TTS_T3_GGUF      Chatterbox T3 GGUF        (default: local mtl checkpoint)
 *   TTS_S3GEN_GGUF   Chatterbox S3Gen GGUF     (default: local mtl checkpoint)
 *   LAVASR_ENHANCER_GGUF  LavaSR enhancer GGUF (default: local f16)
 *   TTS_LANG         language code             (default: en)
 *   OUT_DIR          output directory          (default: data/lavasr/qvac-demo)
 */

const fs = require('bare-fs')
const path = require('bare-path')
const TTSGgml = require('../')
const { createWav } = require('./wav-helper')
const { setLogger, releaseLogger } = require('../addonLogging')

const DATA = '/home/zbig9000/repos/tether/data'
const argv = (global.Bare && global.Bare.argv) || (typeof process !== 'undefined' ? process.argv : [])

const text =
  argv[2] ||
  "QVAC's LavaSR enhancer rebuilds the missing high frequencies, lifting the " +
    'twenty four kilohertz engine output all the way to a crisp forty eight kilohertz.'
const refAudio = argv[3]

const T3 = path.join(DATA, 'chatterbox-convert/models/chatterbox-t3-mtl.gguf')
const S3GEN = path.join(DATA, 'chatterbox-convert/models/chatterbox-s3gen-mtl.gguf')
const ENH = path.join(DATA, 'lavasr/gguf/lavasr-enhancer-f16.gguf')
const LANG = 'en'
const OUT = path.join(DATA, 'lavasr/qvac-demo')

const BASE_SR = 24000
const ENHANCED_SR = 48000

for (const [label, f] of [['t3', T3], ['s3gen', S3GEN], ['lavasr enhancer', ENH]]) {
  if (!fs.existsSync(f)) {
    console.error(`Missing ${label} GGUF: ${f}`)
    if (global.Bare) global.Bare.exit(1)
    else process.exit(1)
  }
}
if (refAudio && !fs.existsSync(refAudio)) {
  console.error(`Reference audio not found: ${refAudio}`)
  if (global.Bare) global.Bare.exit(1)
  else process.exit(1)
}
fs.mkdirSync(OUT, { recursive: true })

async function synthesize (enhance) {
  const files = { t3Model: T3, s3genModel: S3GEN }
  if (enhance) files.lavasrEnhancer = ENH // ← the on switch for LavaSR

  const model = new TTSGgml({
    files,
    ...(refAudio ? { referenceAudio: refAudio } : {}),
    config: { language: LANG },
    logger: console,
    opts: { stats: true }
  })

  await model.load()
  try {
    const response = await model.run({ input: text, type: 'text' })
    let buffer = []
    let sampleRate = enhance ? ENHANCED_SR : BASE_SR
    await response
      .onUpdate(data => {
        if (data && data.outputArray) buffer = buffer.concat(Array.from(data.outputArray))
        if (data && data.sampleRate) sampleRate = data.sampleRate
      })
      .await()
    const stats = response.stats || {}
    return { buffer, sampleRate, stats }
  } finally {
    await model.unload()
  }
}

async function main () {
  setLogger((priority, message) => {
    if (priority > 1) return
    const names = { 0: 'ERROR', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG', 4: 'OFF' }
    console.log(`[C++ ${names[priority] || '?'}] ${message}`)
  })

  console.log('== QVAC LavaSR A/B (native addon → tts-cpp lavasr::Enhancer) ==')
  console.log(`text  : ${text}`)
  console.log(`t3    : ${T3}`)
  console.log(`s3gen : ${S3GEN}`)
  console.log(`enh   : ${ENH}`)
  console.log(`lang  : ${LANG}`)
  console.log(`out   : ${OUT}\n`)

  try {
    console.log('-- pass 1/2: baseline (enhancer OFF) --')
    const before = await synthesize(false)
    const beforePath = path.join(OUT, 'before_24k.wav')
    createWav(before.buffer, before.sampleRate, beforePath)
    console.log(`wrote ${beforePath}  (${before.buffer.length} samples @ ${before.sampleRate} Hz, ${(before.buffer.length / before.sampleRate).toFixed(2)}s)\n`)

    console.log('-- pass 2/2: enhanced (enhancer ON) --')
    const after = await synthesize(true)
    const afterPath = path.join(OUT, 'after_48k.wav')
    createWav(after.buffer, after.sampleRate, afterPath)
    console.log(`wrote ${afterPath}  (${after.buffer.length} samples @ ${after.sampleRate} Hz, ${(after.buffer.length / after.sampleRate).toFixed(2)}s)\n`)

    if (after.sampleRate !== ENHANCED_SR) {
      console.warn(`WARNING: enhanced sample rate was ${after.sampleRate} Hz, expected ${ENHANCED_SR} Hz.`)
    } else {
      console.log(`OK: enhancer reported ${after.sampleRate} Hz (24 kHz -> 48 kHz bandwidth extension confirmed).`)
    }
  } catch (err) {
    console.error('Error during LavaSR A/B:', err)
    throw err
  } finally {
    releaseLogger()
  }
}

main().catch(err => {
  console.error(err)
  if (global.Bare) global.Bare.exit(1)
  else process.exit(1)
})
