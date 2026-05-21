# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.3]

### Note
- Closes [QVAC-19071](https://tetherapp.atlassian.net/browse/QVAC-19071) ("[Whisper] Update qvac-registry-vcpkg and addon with new port versions"): this release is the consumer-side half of that meta task — it picks up the `whisper-cpp 1.8.4.3#4` port version from `qvac-registry-vcpkg` PR #152 (the registry-side half), and bumps the `transcription-whispercpp` addon itself. The QVAC-18992 `ggml-speech` migration (PR #13 + the `ggml-speech` port bump) stays deferred for this release; it will land as a follow-up port bump under the same QVAC-19071 umbrella.

### Fixed
- Android E2E tests crashed inside `whisper_init_from_file_with_params` with `SIGABRT` (`ggml_abort` → `ggml_backend_dev_backend_reg+48` → `whisper_init_with_params_no_state+480`). The integration-mobile-test workflow always failed at `runAccuracyMultilangTest` on both AWS Device Farm devices (Samsung Galaxy S25 Ultra + Pixel 9 Pro), with the app exiting state `4`→`1` 132 ms after `Downloaded model: ggml-tiny.bin`. Root cause: the addon never called `ggml_backend_load_all*()`. With `GGML_BACKEND_DL=ON`, the bundled ggml-base no longer defines `GGML_USE_CPU`, so the static `ggml_backend_registry` ctor registers zero backends — the per-arch `libggml-cpu-android_armv*_*.so` plus `libggml-vulkan.so` / `libggml-opencl.so` must be explicitly loaded before `whisper_init`, otherwise whisper's `ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)` returns NULL → passes NULL to `ggml_backend_dev_backend_reg()` → trips `GGML_ASSERT(device)` → abort. Mirrored the pattern used by every other ggml-based addon in the monorepo (`packages/{diffusion-cpp,llm-llamacpp,classification-ggml,…}`): JS `index.js` passes `path.join(__dirname, 'prebuilds')` as `configurationParams.backendsDir`; the C++ side combines it with the compile-time `BACKENDS_SUBDIR` (`<bare_target>/<module_name>`) and calls `ggml_backend_load_all_from_path()` exactly once per process (`std::once_flag`) on `__ANDROID__` only — desktop platforms still rely on ggml's static-ctor registration. [QVAC-18993]
- `bare-make generate` failed on `android-arm64` (CMake error: `get_target_property() called with non-existent target "ggml::ggml-cpu-android_armv8.0_1"` etc.) after enabling `GGML_BACKEND_DL=ON` on the `whisper-cpp` port. With dynamic-backend mode, ggml builds the per-arch CPU and GPU backends as standalone `MODULE` libraries that `dlopen` at runtime; upstream ggml's `install(TARGETS … EXPORT)` deliberately skips them, so the consumer's `BACKEND_DL_LIBS` loop in `CMakeLists.txt` referenced targets that don't exist. Materialise an imported `SHARED IMPORTED` target locally from each `.so` installed under vcpkg's `bin/` directory, then bundle via the existing `INSTALL TARGET` path — mirrors the pattern used by `packages/diffusion-cpp`. [QVAC-18993]
- `whisper-cpp[vulkan]` failed to build on `x64-windows` with `c1xx: fatal error C1083: Cannot open source file: '.../x64-windows/include'`. Root cause was the new spirv-headers include-shim in the `whisper-cpp` port (and the parallel `ggml-speech` port) emitting `-isystem <path>` into `CMAKE_CXX_FLAGS` — MSVC's `cl.exe` does not understand `-isystem`, treats the flag as a positional source file argument, and then tries to open the path as a `.cpp`. Fixed in `whisper-cpp 1.8.4.3#4` + `ggml-speech 2026-05-19#4` by emitting `/I<path>` on MSVC and keeping `-isystem <path>` on GCC/Clang. [QVAC-18992]
- Android prebuild shipped only CPU + Vulkan `.so` backends — `libggml-opencl.so` was missing despite the `whisper-cpp[opencl]` recipe being green and the `opencl` feature being declared in `vcpkg.json`. Two layered root causes: (1) `CMakeLists.txt` never appended `"opencl"` to `VCPKG_MANIFEST_FEATURES` (only `"tests"` and `"vulkan"` were wired through), so vcpkg resolved the addon without the `opencl` feature; (2) the obvious "gate on `CMAKE_SYSTEM_NAME STREQUAL "Android"`" detection used by `ENABLE_VULKAN` doesn't work at top-level CMakeLists.txt time because the bare-make toolchain file is loaded by `project()` (which runs *after* the option block), so the check always falls through to the Linux host default. The existing `ENABLE_VULKAN` block coincidentally works because Vulkan is also default-ON on Linux. Added an `ENABLE_OPENCL` option defaulted ON unconditionally that appends `"opencl"` to `VCPKG_MANIFEST_FEATURES`; the actual platform gating is delegated to the `platform: android` clause on the `whisper-cpp[opencl]` dep in `vcpkg.json` + the `VCPKG_TARGET_IS_ANDROID` check in the `whisper-cpp` portfile (mirrors the layered platform-gate pattern that `whisper-cpp[vulkan]` already uses with `!osx & !ios`). [QVAC-18300]

### Changed
- Bumped `whisper-cpp` to `1.8.4.3#4`: the two Android dynamic-backend ggml fixes (`GGML_BACKEND_DL` + static-core CMake guards; per-arch CPU dlopen fallback) and the tts-cpp `<atomic>` include fix are now upstreamed as commits on the whisper.cpp fork (`tetherto/qvac-ext-lib-whisper.cpp` PRs #25 + #27 + #28) instead of as port-level patches. Build output is bit-identical to `1.8.4.3#2` (modulo the MSVC `/I` fix above), but the registry no longer maintains the patch tree.
- Added `spirv-headers` to the `microsoft/vcpkg` registry routing in `vcpkg-configuration.json` — required because upstream whisper.cpp v1.8.4.3 unconditionally `#include`s `spirv/unified1/spirv.hpp` in `ggml-vulkan.cpp` (no `find_package(SpirvHeaders)` call in ggml's CMake, so the standalone `SPIRV-Headers` tree is needed on the include path).
- Re-pinned the `Zbig9000/qvac-registry-vcpkg` default-registry baseline to `ee71ecb5b286224377313e5a50558d11adbef3ac` ([qvac-registry-vcpkg PR #152](https://github.com/tetherto/qvac-registry-vcpkg/pull/152) HEAD). Picks up `whisper-cpp 1.8.4.3#0` — port REPO + REF flipped from the temporary `Zbig9000/qvac-ext-lib-whisper.cpp@14620c8857` branch pin to `tetherto/qvac-ext-lib-whisper.cpp@f3102199`, the merge commit for [whisper-cpp PR #28](https://github.com/tetherto/qvac-ext-lib-whisper.cpp/pull/28) (QVAC-18993 bundled-ggml Android dynamic backend) which closes the Group-1/2 merge chain on `tetherto/master` (PR #25 QVAC-18991 upstream-sync + PR #27 QVAC-18966 tts-cpp atomic + PR #28 QVAC-18993 ggml-backend). Port-version was reset to `#0` (per PR #152 review) and the historical `1.8.4.3#3..#5` entries were collapsed into a single canonical `#0` against the new upstream. Source tarball is byte-identical to the previous `1.8.4.3#4` build outside the `parakeet-cpp/` and `tts-cpp/` subdirs (separate vcpkg ports), so the bump is REF-only — no recipe change, no patch change, no build-output change.
- Earlier in this release the same `vcpkg-configuration.json` baseline was rolled through `9f4e8e2…` (MSVC `/I` shim fix) and `f287037…` (initial tetherto REF repoint at `1.8.4.3#5`); the bump above supersedes both.

## [0.7.2]

### Fixed
- Android APK consumers silently lost CPU init when the addon was packaged with `useLegacyPackaging=false` (the AGP ≥ 3.6 default). ggml's `ggml_backend_load_best()` directory iterator finds nothing inside compressed APK libs, and its on-disk filename fallback never composes the per-arch `libggml-cpu-android_armv*_*.so` names that `GGML_CPU_ALL_VARIANTS=ON` produces — so the CPU backend never registered and `init_cpu_backend()` returned null. Bumped `whisper-cpp` to `1.8.4.3#2`, which carries a port-level patch (`0002-ggml-android-cpu-variant-dlopen-fallback.patch`) mirroring [`qvac-ext-ggml@speech 9562ed04`](https://github.com/tetherto/qvac-ext-ggml/commit/9562ed04) to the bundled-ggml tree: on `__ANDROID__` the loader now tries the bare backend name as well as all seven known `cpu-android_armv*_*` variants, then picks the highest-scoring one the device's HWCAP supports. [QVAC-18993]

### Changed
- Re-pinned the `Zbig9000/qvac-registry-vcpkg` default-registry baseline to `86257dc376ca043c67cc4805ab8d1e74a94b7eda` so the `whisper-cpp 1.8.4.3#2` port-version + the matching `ggml-speech 2026-05-19#0` port-version are reachable.

## [0.7.1]

### Changed
- Bumped `whisper-cpp` to `1.8.4.3#1`, which:
  - Syncs the upstream `ggml-org/whisper.cpp` master series up to v1.8.4.3, including the bundled-ggml bump to v0.10.2 and the upstream PR #3677 VAD streaming API (`whisper_vad_detect_speech_no_reset`, `whisper_vad_reset_state`). [QVAC-18991]
  - Enables the `[opencl]` feature on Android, exposing the Adreno OpenCL backend (auto-selected at runtime when the device reports an Adreno 700+ GPU). [QVAC-18300]
  - Switches the Android build to full dynamic-backend mode (`GGML_BACKEND_DL=ON` + `GGML_CPU_ALL_VARIANTS=ON`): the addon `.bare` prebuild now ships one `libggml-cpu-android_armv*_*.so` per microarchitecture plus dynamically-loaded `libggml-vulkan.so` / `libggml-opencl.so`, and ggml's loader picks the best CPU variant + GPU backend per device at runtime. [QVAC-18993]
- Picks up the new `ggml-speech 2026-05-18#1` from the registry baseline (no direct dependency in this package, but consumed transitively by `qvac-lib-infer-parakeet` / `tts-cpp` consumers that share the same Android process). [QVAC-18992]

## [0.7.0]

### Fixed
- iOS bare-kit hard-crash on `transcribe()` after `unload()` (Mach exception 309 / EXC_BAD_ACCESS / PAC failure inside `js_delete_reference` / `js_open_handle_scope`) caused by `qvac-lib-inference-addon-cpp` 1.1.6+ deferring `js_delete_reference()` into a `uv_close` close-callback that races worklet `js_env_t*` invalidation:
  - Added a whisper-local `WhisperOutputCallBackJs` (`addon/src/addon/WhisperOutputCallbackJs.hpp`) that subclasses `OutputCallBackInterface` and synchronously deletes the JS references in its destructor (1.1.5-style ordering), keeping only the no-op `uv_async_t` teardown deferred. Wired into `createInstance` in `addon/src/addon/AddonJs.hpp` instead of the upstream `OutputCallBackJs`.
  - Defense-in-depth: `WhisperInterface.destroyInstance()` (`whisper.js`) now yields twice via `setImmediate` after the native `destroyInstance` returns, guaranteeing a full libuv iteration boundary (and therefore the close phase) elapses before `unload()` resolves to the SDK.

### Changed
- Bumped `qvac-lib-inference-addon-cpp` to `1.1.7#1`.
- Bumped `whisper-cpp` to `1.8.4.2#1`.

## [0.6.8]

### Changed
- Reverted `qvac-lib-inference-addon-cpp` to `1.1.5#1` due to iOS crash. It will be updated again in 0.7.0.

## [0.6.7]

### Changed
- Bumped `qvac-lib-inference-addon-cpp` to `1.1.6`.

## [0.6.6]

### Removed
- Removed redundant `path` (`npm:bare-path`) and `process` (`npm:bare-process@^4.2.2`) entries from `dependencies` in `package.json`. The `bare-path` package is already declared directly as `bare-path: "^3.0.0"`, and `process` was unused.

## [0.6.5]

### Added
- Added opt-in conversation streaming events to `runStreaming()`. Callers can pass `emitVadEvents`, `endOfTurnSilenceMs`, and `vadRunIntervalMs` to receive `{ type: "vad" }` state updates and `{ type: "endOfTurn" }` silence boundary events alongside transcript segments.
- Added native `VadStateUpdate` and `EndOfTurnEvent` output handlers so VAD and end-of-turn events flow through the existing addon output queue without changing the default transcript-only streaming behavior.
- Added `examples/example.mic-conversation.js`, a microphone streaming example that logs VAD state, end-of-turn signals, and transcript output from live audio.
- Added C++ unit coverage for `StreamingProcessor` conversation events, JS unit coverage for event forwarding, and a live-stream integration variant that verifies VAD events are emitted with transcript output.

### Changed
- Extended `runStreaming(audioStream, opts?)` TypeScript declarations to include the new conversation streaming options and output event types.

## [0.6.4]

### Changed
- Fixed bug that prevented Vulkan from being turned on by default on linux and windows

## [0.6.3]

### Added
- Vulkan GPU acceleration enabled by default in CMakeLists.txt for Linux, Android, and Windows (macOS/iOS use Metal)
- Dynamic ggml backend library installation in CMakeLists.txt for Android/Linux (matching the LLM addon pattern)
- Vulkan SDK installation on Windows integration test runner so `vulkan-1.dll` is available at runtime
- `atexit` cleanup handler in `binding.cpp` that clears streaming sessions before C++ static destructors run
- Vulkan GPU smoke test in integration test workflow for Linux GPU runners
- RTF performance benchmark workflow with multi-model/multi-audio matrix support

### Changed
- GPU usage is now opt-in: `use_gpu` defaults to `false` in `toWhisperContextParams` instead of inheriting the upstream default (`true`). Callers must explicitly set `use_gpu: true` to enable GPU acceleration.

### Fixed
- Fixed SIGSEGV (exit code 139) at process exit on Linux GPU runners caused by ggml Vulkan backend static destructor ordering (upstream whisper.cpp#2373)
- Fixed "The specified module could not be found" error on Windows integration tests by installing the Vulkan runtime
- Fixed `t.skip()` calls in GPU smoke test (brittle does not support `t.skip`, replaced with `t.pass`)

## [0.6.2]

### Changed
- Fixed chunking issue re-introduced in 0.6.0 in which the inference output was not streamed but instead returned as a single batched result of the end.

## [0.6.1]

### Changed

- Changed `@qvac/transcription-whispercpp` package visibility on NPM from private to public

## [0.6.0]

This release is a significant interface modernisation. The constructor switches to a local-files map, model download is removed from the load path, concurrent inference runs are serialised instead of rejected, and the class no longer extends `BaseInference`.

## Breaking Changes

### Constructor now takes a `files` map instead of loader + model name

The old API accepted a `loader`, `modelName`, `vadModelName`, and `diskPath`. Those are all removed. Pass local file paths directly:

```typescript
// Before
new TranscriptionWhispercpp({ loader, modelName: 'ggml-tiny.bin', diskPath: '/models' }, config)

// After
new TranscriptionWhispercpp({ files: { model: '/models/ggml-tiny.bin', vadModel: '/models/silero-vad.bin' } }, config)
```

`files.model` is required; `files.vadModel` is optional. No download step occurs — files must already exist on disk before calling `load()`.

### `TranscriptionWhispercpp` no longer extends `BaseInference`

The class is now standalone. `instanceof BaseInference` checks and any BaseInference-only APIs (`getApiDefinition`, `downloadWeights`, loader helpers) are no longer available on this class.

### Weight download removed from `_load`

`_load` previously triggered a `WeightsProvider` download when a loader was supplied. That path is gone. Load preparation is now the caller's responsibility.

## New APIs

### `runStreaming(audioStream)` is now part of the public API

The VAD-based live streaming path was previously internal. It is now a documented public method with its own TypeScript declaration, accepting the same audio stream types as `run()`.

```typescript
const response = await model.runStreaming(audioStream)
for await (const segment of response) { /* ... */ }
```

### Concurrent runs serialise instead of throwing

When `exclusiveRun` is enabled (the default), a second call to `run()` or `runStreaming()` while a transcription is in progress will **wait** for the first to complete rather than throwing a `JOB_ALREADY_RUNNING` error. This makes it safe to call `run()` from concurrent contexts.

### New typed exports

`TranscriptionWhispercppFiles` and `InferenceClientState` are now exported from the `TranscriptionWhispercpp` namespace. Lifecycle methods (`load`, `unload`, `destroy`, `cancel`, `pause`, `unpause`, `stop`, `status`, `getState`) are now explicitly declared in `index.d.ts`.

## [0.5.6]

### Changed
- Fixed chunking issue introduced in 0.5.0 in which the inference output was not streamed but instead returned as a single batched result of the end.

## [0.5.5]

### Changed
- Bumped `inference-addon-cpp` to `1.1.5`.
- Restored JS-owned job ID routing after addon-cpp reverted the accidental `1.1.3` native callback `jobId` contract and `cancel(jobId)` API break.

### Added
- Regression coverage for JS-owned cancel handling of active, buffered, and stale wrapper job IDs.

### Removed
- References of s3 bucket throughout documentation and helper scripts

## [0.5.4]

### Changed

- README: removed outdated npm Personal Access Token / `.npmrc` setup instructions for installing `@qvac/transcription-whispercpp`.

## [0.5.3]

### Changed
- Bumped `inference-addon-cpp` to `1.1.3`.
- Updated the JS wrapper to consume the shared addon-cpp native job-id callback contract so late cancel/error events remain attached to the cancelled job instead of a newer accepted run.

### Added
- Regression coverage for rejected runs and stale cancel callbacks in the addon inference tests.

## [0.5.2]

Security hardening release from comprehensive security audit.

### Fixed
- Replace global streaming state with per-instance map to eliminate race condition and dangling pointer risk (#1079)
- Add 500 MB buffer limit to audio accumulation to prevent OOM from unbounded buffering (#1080)
- Add SHA-256 integrity verification to model download scripts using HuggingFace LFS checksums (#1081)
- Validate `suppress_regex` parameter — ban grouping constructs (parentheses) and enforce 512-char length limit to prevent ReDoS (#1083)
- Sanitize error messages to remove filesystem paths from thrown errors (#1084)
- Wrap job ID counter at `Number.MAX_SAFE_INTEGER` to prevent precision loss (#1085)
- Harden benchmark server: add library allowlist, restrict file paths to allowed directories, remove dynamic `npm install`, add body size limit, restrict CORS to localhost (#1086)

## [0.5.1]

This release documents runtime statistics and transcription output shapes in TypeScript so consumers can type `response.stats` and `run()` results against the native addon.

## New APIs

### `RuntimeStats` and related types in `index.d.ts`

The `TranscriptionWhispercpp` namespace now exports **`RuntimeStats`**, aligned with `WhisperModel::runtimeStats()` (`totalTime`, `realTimeFactor`, `tokensPerSecond`, `audioDurationMs`, `totalSamples`, `totalTokens`, `totalSegments`, `processCalls`, and Whisper-internal timing fields through `totalWallMs`). **`WhisperTranscriptionSegment`** and **`WhisperRunOutput`** describe transcription payloads passed to `onUpdate`. **`run()`** is typed to return **`Promise<QvacResponse<WhisperRunOutput>>`**, with a note that **`response.stats`** matches **`RuntimeStats`** when stats collection is enabled via `opts.stats`.

## [0.5.0]

### Changed
- Migrated the native addon implementation to `inference-addon-cpp` 1.x (`IModel` + `AddonJs`/`AddonCpp`), replacing the removed legacy templated addon and jobs-handler path
- Updated the JS/native execution path to `createInstance` + `runJob` with parity-focused cancel/output lifecycle handling

### Added
- Expanded C++/JS parity coverage for addon-cpp runtime behavior, including dedicated `AddonCpp` tests

## [0.4.2]

### Changed
- Logger type in `TranscriptionWhispercppArgs` now uses `LoggerInterface` from `@qvac/logging` instead of a package-specific type, aligning with the shared logging interface used across all addons

## [0.4.1]

### Added
- HuggingFace model download support for standard Whisper and Silero VAD models
- Download script `scripts/download-models.sh` for interactive model downloads
- Auto-download of models in test helpers (`ensureWhisperModel`, `ensureVADModel`)
- Architecture documentation

### Removed
- Legacy P2P data loader peer dependency and dev dependency
- Legacy examples (`transcription.hd.js`, `exampleVad.hd.js`)

## [0.4.0]

### Removed
- `TranscriptionFfmpegAddon` module (`transcription-ffmpeg.js`, `transcription-ffmpeg.d.ts`, `examples/example.ffmpeg.js`)
- `@qvac/util-transcription` dependency

## [0.3.18]

### Added
- Windows platform support with PowerShell-specific CI configurations
- Prebuild package renaming from `tetherto__*` to `qvac__*` format

### Fixed
- Whisper.cpp API compatibility updated to new 4-parameter `whisper_full()` API

### Changed
- Integration tests now use `bare@1.26.0` for build consistency

## [0.3.17]

### Fixed
- Spurious linux-x64 prebuild compilation issue

## [0.3.16]

### Changed
- Audio decoder dependency updated to use FFmpeg (`@qvac/decoder-audio` v0.3.3) instead of GStreamer
- `@qvac/util-transcription` updated to v0.1.4, replacing all GStreamer references with FFmpeg

## [0.3.15]

### Changed
- Linux x64 builds switched to Ubuntu 22.04 for wider glibc compatibility
- Integration test matrix expanded to include Ubuntu 22.04 and 24.04
- Vulkan SDK installation improved for x64 and arm64 Linux architectures

### Removed
- Unnecessary Vulkan SDK installation from integration tests
- Custom vcpkg installation step no longer needed with standard Ubuntu runners

## [0.3.14]

### Changed
- Debug symbols stripped from native addon binaries on Linux and macOS for smaller prebuilt artifacts

### Removed
- Redundant Android artifact replication step

## [0.3.13]

### Fixed
- Type declarations: `Loader` and `QvacResponse` now correctly imported from `@qvac/infer-base`
- `test:dts` now passes

## [0.3.12]

### Added
- TypeScript type declarations for `addonLogging` subpath export

### Fixed
- `test:dts` script now references `transcription-ffmpeg.d.ts` instead of deleted `transcription-addon/index.d.ts`

## [0.3.11]

### Added
- Runtime statistics support for Whisper model performance tracking
  - New `runtimeStats()` method exposing detailed metrics (totalTime, realTimeFactor, tokensPerSecond, audioDurationMs, etc.)
  - Integration test validating stats are populated when `opts.stats=true`

## [0.3.10]

### Added
- Linux ARM64 prebuild support using `ubuntu-24.04-arm` runner (#386)
- Linux ARM64 integration tests (#390)

### Changed
- Updated CODEOWNERS (#380)
- Updated PR description template with team practices (#391)

## [0.3.9]

### Added
- darwin-x64 (macOS Intel) prebuild support (#378)
- Windows x64 integration tests (#371)
- Full benchmark scripts (#372)
- vcpkg and ccache caching in prebuilds workflow for ~35% faster builds (#383)

### Fixed
- Eliminated cold start delay - first transcription now runs 3x faster (#385)
- CI workflow fixes for linux-x64 prebuild on GPU runner (#375)
- Permission fix for workflows (#376)

### Changed
- Freeze vcpkg version on macOS for build reproducibility (#377)

## [0.3.8]

### Added
- AraDiaWER metric for Arabic dialect speech recognition benchmarking (#358)

### Fixed
- FFmpeg example to correctly pass audio format (#363)

## [0.3.7]

### Changed
- Updated util-transcription dependency version (#360)

## [0.3.6]

### Changed
- Updated decoder dependency version (#359)

## [0.3.5]

### Added
- Unit tests for Whisper model file validation (#352)
- Model file and VAD path validation logic (#352)

## [0.3.4]

### Fixed
- Job ID return value (#353)

### Changed
- Reorganized examples and cleaned up unnecessary files (#356)

## [0.3.3]

### Added
- Addon logging JS interface export (#357)

## [0.3.2]

### Added
- Enhanced C++ logging for WhisperModel and job handlers (#349)
- DEBUG-level logs for job queue and audio input handling (#349)

### Fixed
- Configuration errors in examples (#341)
- Updated Bare runtime version requirement to >= 1.24.2 (#354)

### Changed
- Reworked integration tests to use TranscriptionWhispercpp (#345)
- Updated documentation to reflect current codebase structure (#354)
