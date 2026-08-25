'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const packageRoot = path.resolve(__dirname, '..', '..')
const packageJson = JSON.parse(fs.readFileSync(path.join(packageRoot, 'package.json'), 'utf8'))
const vcpkgManifest = JSON.parse(fs.readFileSync(path.join(packageRoot, 'vcpkg.json'), 'utf8'))
const cmakeSource = fs.readFileSync(path.join(packageRoot, 'CMakeLists.txt'), 'utf8')

const CUDA_CMAKE_OPTION = 'ASR_CUDA'
const CUDA_MANIFEST_FEATURE = 'cuda'
const SPEECH_PORT = 'speech-cpp'
const DESKTOP_PLATFORM = '!(osx | ios | android)'

function speechDependencies(dependencies) {
  return dependencies.filter((dependency) => dependency.name === SPEECH_PORT)
}

function desktopSpeechDependency() {
  return speechDependencies(vcpkgManifest.dependencies).find(
    (dependency) => dependency.platform === DESKTOP_PLATFORM
  )
}

function cudaFeature() {
  return vcpkgManifest.features[CUDA_MANIFEST_FEATURE]
}

function cudaSpeechDependency() {
  return speechDependencies(cudaFeature().dependencies)[0]
}

function versionFloorOf(dependency) {
  return dependency['version>=']
}

test('the CUDA build is opt-in behind a dedicated CMake option', () => {
  assert.match(cmakeSource, new RegExp(`option\\(${CUDA_CMAKE_OPTION} "[^"]+" OFF\\)`))
  assert.equal(packageJson.scripts['build:native'].includes(CUDA_CMAKE_OPTION), false)
})

test('the CUDA CMake option selects the cuda vcpkg manifest feature', () => {
  assert.match(
    cmakeSource,
    new RegExp(
      `if\\(${CUDA_CMAKE_OPTION}\\)\\s*\\n\\s*list\\(APPEND VCPKG_MANIFEST_FEATURES "${CUDA_MANIFEST_FEATURE}"\\)`
    )
  )
})

test('the CUDA build scripts turn the CMake option on', () => {
  assert.match(packageJson.scripts['build:native:cuda'], /bare-make generate -D ASR_CUDA=ON/)
  assert.equal(packageJson.scripts['build:cuda'], 'npm run build:ts && npm run build:native:cuda')
})

test('the cuda feature forwards to the speech-cpp CUDA backend', () => {
  assert.deepEqual(cudaSpeechDependency().features, [CUDA_MANIFEST_FEATURE])
  assert.equal(cudaSpeechDependency()['default-features'], false)
})

test('the cuda feature is confined to the platforms that have NVIDIA GPUs', () => {
  assert.equal(cudaFeature().supports, DESKTOP_PLATFORM)
  assert.equal(cudaSpeechDependency().platform, DESKTOP_PLATFORM)
})

test('the cuda feature requires a speech-cpp that declares it', () => {
  assert.ok(versionFloorOf(cudaSpeechDependency()) >= versionFloorOf(desktopSpeechDependency()))
})
