'use strict'

// Resident-set-size (RSS) sampling shared by the desktop RTF benchmark
// (test/benchmark/rtf-benchmark.test.js) and the mobile perf runner
// (test/integration/mobile-perf-runner.js). RSS is read from the Bare runtime
// so average / peak / reclaimed figures stay comparable across models,
// platforms and quantizations, and it sidesteps the getrusage maxRSS unit
// mismatch between linux (kilobytes) and darwin (bytes).

const BYTES_PER_MB = 1024 * 1024
const DEFAULT_SAMPLE_INTERVAL_MS = 25

function isPositiveNumber (value) {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
}

function readRssFromBareOs () {
  try {
    const usage = require('bare-os').memoryUsage()
    return usage && isPositiveNumber(usage.rss) ? usage.rss : 0
  } catch {
    return 0
  }
}

function resolveProcess () {
  if (globalThis.process && typeof globalThis.process.memoryUsage === 'function') {
    return globalThis.process
  }
  try {
    const proc = require('bare-process')
    return typeof proc.memoryUsage === 'function' ? proc : null
  } catch {
    return null
  }
}

function readRssFromProcess () {
  try {
    const proc = resolveProcess()
    const usage = proc && proc.memoryUsage()
    return usage && isPositiveNumber(usage.rss) ? usage.rss : 0
  } catch {
    return 0
  }
}

function readRssBytes () {
  return readRssFromBareOs() || readRssFromProcess()
}

function filterPositive (samples) {
  return (Array.isArray(samples) ? samples : []).filter(isPositiveNumber)
}

function sumValues (values) {
  return values.reduce((total, value) => total + value, 0)
}

function maxValue (values) {
  return values.reduce((current, value) => (value > current ? value : current), values[0])
}

function minValue (values) {
  return values.reduce((current, value) => (value < current ? value : current), values[0])
}

function summarizeSamples (samples) {
  const values = filterPositive(samples)
  if (values.length === 0) {
    return { count: 0, avgBytes: 0, peakBytes: 0, minBytes: 0 }
  }
  return {
    count: values.length,
    avgBytes: sumValues(values) / values.length,
    peakBytes: maxValue(values),
    minBytes: minValue(values)
  }
}

function scheduleSample (state) {
  state.timer = setTimeout(() => {
    if (!state.running) return
    collectSample(state)
    scheduleSample(state)
  }, state.intervalMs)
  if (state.timer && typeof state.timer.unref === 'function') {
    state.timer.unref()
  }
}

function collectSample (state) {
  const rss = readRssBytes()
  if (rss > 0) state.samples.push(rss)
  return rss
}

function startSampler (state) {
  if (state.running) return
  state.running = true
  collectSample(state)
  scheduleSample(state)
}

function stopSampler (state) {
  state.running = false
  if (state.timer) {
    clearTimeout(state.timer)
    state.timer = null
  }
  collectSample(state)
  return summarizeSamples(state.samples)
}

function createMemorySampler (options) {
  const state = {
    intervalMs: (options && options.intervalMs) || DEFAULT_SAMPLE_INTERVAL_MS,
    samples: [],
    timer: null,
    running: false
  }
  return {
    start: () => startSampler(state),
    stop: () => stopSampler(state),
    sampleOnce: () => collectSample(state),
    get samples () { return state.samples }
  }
}

function toBytes (value) {
  return isPositiveNumber(value) ? value : 0
}

function computeReclaim (input) {
  const afterLoad = toBytes(input && input.rssAfterLoadBytes)
  const peak = toBytes(input && input.peakRssBytes)
  const afterUnload = toBytes(input && input.rssAfterUnloadBytes)
  const baseline = peak > 0 ? peak : afterLoad
  return {
    reclaimedBytes: afterLoad - afterUnload,
    reclaimedFromPeakBytes: baseline - afterUnload,
    reclaimedRatio: afterLoad > 0 ? (afterLoad - afterUnload) / afterLoad : 0
  }
}

function roundTo (value, digits) {
  const factor = 10 ** digits
  return Math.round(value * factor) / factor
}

function bytesToMb (bytes, digits) {
  if (!isPositiveNumber(bytes)) return 0
  const mb = bytes / BYTES_PER_MB
  return typeof digits === 'number' ? roundTo(mb, digits) : mb
}

function buildMemorySummary (input) {
  const source = input || {}
  const rssAfterLoadBytes = toBytes(source.rssAfterLoadBytes)
  const peakRssBytes = Math.max(toBytes(source.peakRssBytes), rssAfterLoadBytes)
  const rssAfterUnloadBytes = toBytes(source.rssAfterUnloadBytes)
  const reclaim = computeReclaim({ rssAfterLoadBytes, peakRssBytes, rssAfterUnloadBytes })
  return {
    avgRssMb: bytesToMb(source.avgRssBytes, 2),
    peakRssMb: bytesToMb(peakRssBytes, 2),
    rssBeforeLoadMb: bytesToMb(source.rssBeforeLoadBytes, 2),
    rssAfterLoadMb: bytesToMb(rssAfterLoadBytes, 2),
    rssAfterUnloadMb: bytesToMb(rssAfterUnloadBytes, 2),
    reclaimedMb: bytesToMb(reclaim.reclaimedBytes, 2),
    reclaimedFromPeakMb: bytesToMb(reclaim.reclaimedFromPeakBytes, 2),
    sampleCount: isPositiveNumber(source.sampleCount) ? source.sampleCount : 0
  }
}

module.exports = {
  BYTES_PER_MB,
  DEFAULT_SAMPLE_INTERVAL_MS,
  readRssBytes,
  createMemorySampler,
  summarizeSamples,
  computeReclaim,
  bytesToMb,
  buildMemorySummary
}
