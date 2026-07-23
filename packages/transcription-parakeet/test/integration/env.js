'use strict'

const process = require('bare-process')

function unsetEnvVar(key) {
  try {
    delete process.env[key]
  } catch {}
}

function restoreEnvVar(key, previousValue) {
  if (previousValue === undefined) {
    unsetEnvVar(key)
    return
  }
  process.env[key] = previousValue
}

module.exports = {
  unsetEnvVar,
  restoreEnvVar
}
