'use strict'

const test = require('brittle')
const { normalizeEnhancer, VALID_ENHANCERS, DEFAULT_ENHANCER } = require('../utils/downloadModel')

test('normalizeEnhancer accepts the known enhancer axis values', (t) => {
  t.is(normalizeEnhancer('none'), 'none', 'none -> none')
  t.is(normalizeEnhancer('lavasr'), 'lavasr', 'lavasr -> lavasr')
  t.is(normalizeEnhancer('LavaSR'), 'lavasr', 'case-insensitive')
})

test('normalizeEnhancer defaults empty / unset input to the shared default', (t) => {
  t.is(DEFAULT_ENHANCER, 'none', 'default enhancer is none (engine as-is)')
  t.ok(VALID_ENHANCERS.includes('lavasr'), 'lavasr is a valid enhancer')
  t.is(normalizeEnhancer(''), DEFAULT_ENHANCER, "'' -> default")
  t.is(normalizeEnhancer(undefined), DEFAULT_ENHANCER, 'undefined -> default')
  t.is(normalizeEnhancer(null), DEFAULT_ENHANCER, 'null -> default')
})

test('normalizeEnhancer throws on an unknown enhancer so a typo fails loudly', (t) => {
  t.exception(
    () => normalizeEnhancer('lavasr-typo'),
    /Invalid benchmark enhancer/,
    'unknown enhancer is rejected instead of silently disabling enhancement'
  )
})
