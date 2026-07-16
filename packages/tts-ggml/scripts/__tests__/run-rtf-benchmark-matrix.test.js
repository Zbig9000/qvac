'use strict'

/**
 * Unit tests for the pure label/env helpers in
 * scripts/run-rtf-benchmark-matrix.js.
 *
 * Guards the byte-stability of the enhancer axis on the *producing* side: the
 * matrix run label (and therefore the artifact filename segment it feeds) must
 * only grow a `-<enhancer>` tag when the enhancer is enabled, so enhancer=none
 * runs stay byte-for-byte identical to pre-axis runs.
 *
 * Pure-function code paths only — requiring the module does not spawn any
 * benchmark (main() is guarded by require.main === module).
 *
 * Run locally:
 *   node --test scripts/__tests__/run-rtf-benchmark-matrix.test.js
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const { buildLabel } = require('../run-rtf-benchmark-matrix')

test('buildLabel omits the enhancer tag for the default (enhancer=none)', () => {
  assert.equal(buildLabel({ engine: 'supertonic', useGPU: true }, 0), '1-supertonic-gpu')
  assert.equal(
    buildLabel({ engine: 'chatterbox', enhancer: 'none', useGPU: false }, 1),
    '2-chatterbox-cpu'
  )
})

test('buildLabel inserts the enhancer tag only when the enhancer is enabled', () => {
  assert.equal(
    buildLabel({ engine: 'supertonic', enhancer: 'lavasr', useGPU: true }, 2),
    '3-supertonic-lavasr-gpu'
  )
  assert.equal(
    buildLabel({ engine: 'chatterbox', enhancer: 'lavasr', useGPU: false }, 0),
    '1-chatterbox-lavasr-cpu'
  )
})

test('buildLabel lowercases the enhancer tag so the segment is stable', () => {
  assert.equal(
    buildLabel({ engine: 'supertonic', enhancer: 'LavaSR', useGPU: false }, 0),
    '1-supertonic-lavasr-cpu'
  )
})

test('buildLabel honours an explicit label verbatim', () => {
  assert.equal(buildLabel({ label: 'custom-label', enhancer: 'lavasr' }, 4), 'custom-label')
})

test('buildLabel falls back to the tts engine name when none is given', () => {
  assert.equal(buildLabel({ useGPU: false }, 0), '1-tts-cpu')
})
