'use strict'

const test = require('brittle')
const process = require('bare-process')
const { unsetEnvVar, restoreEnvVar } = require('../integration/env.js')

const KEY = 'QVAC_TEST_ENV_HELPER_PROBE'

test('unsetEnvVar removes a set variable without aborting on the Bare env proxy', (t) => {
  process.env[KEY] = 'sentinel'
  t.execution(() => unsetEnvVar(KEY), 'unsetEnvVar does not throw on the Bare process.env proxy')
  t.is(process.env[KEY], undefined, 'variable is unset')
})

test('unsetEnvVar is a no-op when the variable is already absent', (t) => {
  unsetEnvVar(KEY)
  t.execution(() => unsetEnvVar(KEY), 'unsetEnvVar does not throw for an absent key')
  t.is(process.env[KEY], undefined, 'variable stays absent')
})

test('restoreEnvVar reinstates the previous value', (t) => {
  process.env[KEY] = 'original'
  const previous = process.env[KEY]
  process.env[KEY] = 'overwritten'
  restoreEnvVar(KEY, previous)
  t.is(process.env[KEY], 'original', 'previous value is restored')
  unsetEnvVar(KEY)
})

test('restoreEnvVar unsets the variable when there was no previous value', (t) => {
  unsetEnvVar(KEY)
  const previous = process.env[KEY]
  process.env[KEY] = 'temporary'
  restoreEnvVar(KEY, previous)
  t.is(process.env[KEY], undefined, 'variable is unset when previous was undefined')
})
