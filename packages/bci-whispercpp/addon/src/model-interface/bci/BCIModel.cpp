#include "BCIModel.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <ranges>
#include <string>
#include <utility>

#include <ggml-backend.h>

#include "BCIConfig.hpp"
#include "addon/BCIErrors.hpp"
#include "inference-addon-cpp/Errors.hpp"
#include "inference-addon-cpp/Logger.hpp"
#include "model-interface/BCITypes.hpp"

namespace qvac_lib_inference_addon_bci {

namespace {
constexpr float K_SEGMENT_TIMESTAMP_SCALE = 0.01F;
constexpr int K_WARMUP_SAMPLE_COUNT = 8000;
constexpr int K_DUMMY_AUDIO_30S = 16000 * 30;
} // namespace

static bool shouldAbortWhisper(void* userData) {
  const auto* cancelRequested = static_cast<const std::atomic_bool*>(userData);
  return cancelRequested != nullptr &&
         cancelRequested->load(std::memory_order_relaxed);
}

// Called right before the encoder runs. Replaces the mel spectrogram
// (computed from dummy silence) with our neural-signal-derived features.
static bool onEncoderBegin(
    whisper_context* ctx, whisper_state* state, void* userData) {
  auto* cbData = static_cast<BCIModel::EncoderCallbackData*>(userData);
  if (cbData == nullptr || cbData->melData == nullptr) {
    return true;
  }

  int result = whisper_set_mel_with_state(
      cbData->ctx, state,
      cbData->melData, cbData->melFrames, cbData->melBins);

  if (result != 0) {
    QLOG(qvac_lib_inference_addon_cpp::logger::Priority::ERROR,
         "whisper_set_mel_with_state failed: " + std::to_string(result));
    return false;
  }

  QLOG(qvac_lib_inference_addon_cpp::logger::Priority::DEBUG,
       "Injected neural mel features: " +
           std::to_string(cbData->melFrames) + " frames x " +
           std::to_string(cbData->melBins) + " bins");
  return true;
}

#if defined(__ANDROID__)
namespace {
// Android ships ggml with `GGML_BACKEND_DL=ON`, so no backend is
// statically registered. dlopen the per-arch CPU + GPU `.so` modules
// once per process; otherwise whisper_init aborts on a NULL CPU
// device. Mirrors transcription-whispercpp / llm-llamacpp /
// diffusion-cpp. See `aiDocs/15-android-mobile-test-crash-fix.md`
// for the post-mortem of the same crash on transcription-whispercpp.
//
// Today (PR 1) this is dormant because bci-whispercpp still pins
// `whisper-cpp@1.8.4.2` whose port does NOT set `GGML_BACKEND_DL=ON`
// on Android. The static CPU backend registers via ggml's ctor, the
// loader call below finds no MODULE `.so` files and is a no-op. The
// follow-up PR (QVAC-19009) bumps to `whisper-cpp@1.8.5` which
// inherits the `ggml-speech` port's unconditional Android
// `GGML_BACKEND_DL=ON` flag; from that point on this function is
// the only thing standing between us and the SIGABRT we already
// hit on transcription's PR #2124.
void ensureBackendsLoadedAndroid(const std::string& backendsDir) {
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    if (backendsDir.empty()) {
      QLOG(
          qvac_lib_inference_addon_cpp::logger::Priority::WARNING,
          "Android: configurationParams.backendsDir not set; falling back to "
          "ggml_backend_load_all() (default search path). CPU / Vulkan / "
          "OpenCL registration may fail inside an APK with default "
          "compressed-native-libs packaging.");
      ggml_backend_load_all();
      return;
    }
#ifdef BACKENDS_SUBDIR
    const std::filesystem::path variantsDir =
        (std::filesystem::path(backendsDir) /
         std::filesystem::path(BACKENDS_SUBDIR))
            .lexically_normal();
#else
    const std::filesystem::path variantsDir = backendsDir;
#endif
    QLOG(
        qvac_lib_inference_addon_cpp::logger::Priority::INFO,
        std::string("Android: loading ggml backends from: ") +
            variantsDir.string());
    ggml_backend_load_all_from_path(variantsDir.string().c_str());
  });
}
} // namespace
#endif // __ANDROID__

BCIModel::BCIModel(BCIConfig config)
    : cfg_(std::move(config)), neuralProcessor_() {}

BCIModel::~BCIModel() noexcept {
  try {
    unload();
  } catch (...) {
    is_loaded_ = false;
  }
}

void BCIModel::loadEmbedderIfNeeded() {
  if (neuralProcessor_.hasWeights()) {
    return;
  }

  // Look for embedder weights next to the model file
  auto modelPathIt = cfg_.whisperContextCfg.find("model");
  if (modelPathIt == cfg_.whisperContextCfg.end()) {
    return;
  }
  const auto modelPath = std::get<std::string>(modelPathIt->second);

  auto lastSep = modelPath.find_last_of("/\\");
  auto dir = (lastSep != std::string::npos)
                 ? modelPath.substr(0, lastSep)
                 : ".";
  auto embedderPath = dir + "/bci-embedder.bin";

  if (neuralProcessor_.loadEmbedderWeights(embedderPath)) {
    QLOG(qvac_lib_inference_addon_cpp::logger::Priority::INFO,
         "Loaded BCI embedder weights from: " + embedderPath);
  } else {
    throw qvac_errors::bci_error::makeStatus(
        qvac_errors::bci_error::Code::EmbedderWeightsNotFound,
        "BCI embedder weights not found at: " + embedderPath +
        ". This file is required for neural signal preprocessing. "
        "Generate it with: python3 scripts/convert-model.py --checkpoint <ckpt>");
  }
}

void BCIModel::load() {
  if (ctx_) return;

#if defined(__ANDROID__)
  ensureBackendsLoadedAndroid(cfg_.backendsDir);
#endif

  whisper_context_params contextParams = toWhisperContextParams(cfg_);

  const auto modelPathIt = cfg_.whisperContextCfg.find("model");
  if (modelPathIt == cfg_.whisperContextCfg.end()) {
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        "Model path not specified in contextParams");
  }
  const auto modelPath = std::get<std::string>(modelPathIt->second);

  QLOG(qvac_lib_inference_addon_cpp::logger::Priority::INFO,
       "Loading BCI model from: " + modelPath);

  auto* rawCtx = whisper_init_from_file_with_params(modelPath.c_str(), contextParams);
  if (rawCtx == nullptr) {
    throw qvac_errors::bci_error::makeStatus(
        qvac_errors::bci_error::Code::FailedToLoadModel,
        "Failed to initialize Whisper context from: " + modelPath);
  }

  try {
    ctx_.reset(rawCtx);
    captureActiveBackendInfo();
    loadEmbedderIfNeeded();
    if (!is_warmed_up_) {
      warmup();
      is_warmed_up_ = true;
    }
    is_loaded_ = true;
  } catch (...) {
    ctx_.reset();
    is_loaded_ = false;
    throw;
  }
}

void BCIModel::unload() {
  resetContext();
  is_loaded_ = false;
  is_warmed_up_ = false;
}

void BCIModel::reload() {
  unload();
  load();
}

void BCIModel::reset() {
  output_.clear();
  totalTokens_ = 0;
  totalSegments_ = 0;
  processCalls_ = 0;
  totalWallMs_ = 0.0;
  whisperSampleMs_ = 0.0;
  whisperEncodeMs_ = 0.0;
  whisperDecodeMs_ = 0.0;
  whisperBatchdMs_ = 0.0;
  whisperPromptMs_ = 0.0;
}

namespace {
// Stable numeric mapping from a ggml backend registry name to the
// integer code surfaced on JS as `RuntimeStats.backendId`. Kept in
// lock-step with transcription-whispercpp's `backendIdFromRegName`
// (see qvac/packages/transcription-whispercpp/addon/src/model-interface/
// whisper.cpp/WhisperModel.cpp) and transcription-parakeet's `BackendId`
// enum so the same integer means the same backend family across all
// three speech-stack addons. Match by lowercased substring because
// `ggml_backend_reg_name()` can return indexed strings like "CUDA0" /
// "Vulkan0" / "MTL0" when multiple GPUs of the same family are present.
int64_t backendIdFromRegName(const std::string& nameLower) {
  if (nameLower.find("metal") != std::string::npos ||
      nameLower.find("mtl") != std::string::npos) {
    return 1;
  }
  if (nameLower.find("cuda") != std::string::npos) {
    return 2;
  }
  if (nameLower.find("vulkan") != std::string::npos) {
    return 3;
  }
  if (nameLower.find("opencl") != std::string::npos) {
    return 4;
  }
  return 99;
}

// Read whisper_context_params.use_gpu / .gpu_device out of the
// BCIConfig variant map so captureActiveBackendInfo() can mirror
// whisper.cpp's own backend-pick logic. Defaults match
// `whisper_context_default_params()` (use_gpu=false, gpu_device=0;
// the JS-facing default in this addon is also "CPU unless the
// caller opts in").
bool configUseGpu(const BCIConfig& cfg) {
  const auto it = cfg.whisperContextCfg.find("use_gpu");
  if (it == cfg.whisperContextCfg.end()) {
    return false;
  }
  if (const auto* asBool = std::get_if<bool>(&it->second)) {
    return *asBool;
  }
  return false;
}

int configGpuDeviceIndex(const BCIConfig& cfg) {
  const auto it = cfg.whisperContextCfg.find("gpu_device");
  if (it == cfg.whisperContextCfg.end()) {
    return -1;
  }
  if (const auto* asDouble = std::get_if<double>(&it->second)) {
    return static_cast<int>(*asDouble);
  }
  if (const auto* asInt = std::get_if<int>(&it->second)) {
    return *asInt;
  }
  return -1;
}
} // namespace

void BCIModel::captureActiveBackendInfo() {
  // Reset to CPU defaults so a fresh load() that doesn't end up on a GPU
  // still reports sensible values (parity with WhisperModel /
  // ParakeetModel post-load behaviour).
  backend_device_ = 0;
  backend_id_ = 0;
  backend_name_ = "CPU";
  gpu_mem_total_mb_ = -1;
  gpu_mem_free_mb_ = -1;
  gpu_device_description_.clear();

  const bool useGpu = configUseGpu(cfg_);
  const int gpuDeviceIndex = configGpuDeviceIndex(cfg_);

  if (!useGpu) {
    QLOG(
        qvac_lib_inference_addon_cpp::logger::Priority::INFO,
        "Active backend: CPU (use_gpu=false)");
    return;
  }

  // Mirror whisper.cpp's `whisper_backend_init_gpu()` selection: pick
  // the device at the configured `gpu_device` index when set,
  // otherwise the first `GGML_BACKEND_DEVICE_TYPE_GPU` in ggml's
  // enumeration order. Whisper does NOT consider IGPU / ACCEL, so we
  // mustn't either -- reporting an IGPU here would lie about what
  // whisper actually initialised against and confuse the Device Farm
  // assertions (Mali vs Adreno).
  ggml_backend_dev_t dev = nullptr;
  if (gpuDeviceIndex >= 0) {
    dev = ggml_backend_dev_get(static_cast<size_t>(gpuDeviceIndex));
    if (dev != nullptr &&
        ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
      dev = nullptr;
    }
  } else {
    const size_t devCount = ggml_backend_dev_count();
    for (size_t i = 0; i < devCount; ++i) {
      ggml_backend_dev_t candidate = ggml_backend_dev_get(i);
      if (candidate != nullptr &&
          ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_GPU) {
        dev = candidate;
        break;
      }
    }
  }

  if (dev == nullptr) {
    QLOG(
        qvac_lib_inference_addon_cpp::logger::Priority::WARNING,
        "BCI: use_gpu=true was requested but no GGML GPU device is "
        "registered (silent CPU fallback). Likely causes: the GPU backend "
        "library wasn't loaded (Android: ggml_backend_load_all_from_path "
        "failed for the backendsDir), the device was rejected by the "
        "backend (Adreno-tier policy, missing OpenCL ICD, iOS/Android "
        "simulator without GPU support), or no GPU backend was compiled "
        "into ggml-speech for this triplet.");
    return;
  }

  ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
  const char* regName = (reg != nullptr) ? ggml_backend_reg_name(reg) : "";
  const char* devName = ggml_backend_dev_name(dev);
  const char* devDesc = ggml_backend_dev_description(dev);

  std::string regNameLower = (regName != nullptr) ? regName : "";
  std::transform(
      regNameLower.begin(),
      regNameLower.end(),
      regNameLower.begin(),
      [](unsigned char c) { return std::tolower(c); });

  backend_device_ = 1;
  backend_id_ = backendIdFromRegName(regNameLower);
  backend_name_ = (regName != nullptr) ? regName : "";
  gpu_device_description_ =
      (devDesc != nullptr) ? devDesc : (devName != nullptr ? devName : "");

  size_t freeBytes = 0;
  size_t totalBytes = 0;
  ggml_backend_dev_memory(dev, &freeBytes, &totalBytes);
  constexpr size_t kBytesPerMb = 1024U * 1024U;
  gpu_mem_total_mb_ =
      totalBytes > 0 ? static_cast<int64_t>(totalBytes / kBytesPerMb) : -1;
  gpu_mem_free_mb_ =
      freeBytes > 0 ? static_cast<int64_t>(freeBytes / kBytesPerMb) : -1;

  QLOG(
      qvac_lib_inference_addon_cpp::logger::Priority::INFO,
      std::string("Active backend: id=") + std::to_string(backend_id_) +
          " device=" + std::to_string(backend_device_) + " name='" +
          backend_name_ + "' gpu_device='" + gpu_device_description_ +
          "' mem_total_mb=" + std::to_string(gpu_mem_total_mb_) +
          " mem_free_mb=" + std::to_string(gpu_mem_free_mb_));
}

qvac_lib_inference_addon_cpp::RuntimeStats BCIModel::runtimeStats() const {
  qvac_lib_inference_addon_cpp::RuntimeStats stats;

  const double totalTimeSec = totalWallMs_ / 1000.0;
  const double tps = totalTimeSec > 0.0
                         ? (static_cast<double>(totalTokens_) / totalTimeSec)
                         : 0.0;

  stats.emplace_back("totalTime", totalTimeSec);
  stats.emplace_back("tokensPerSecond", tps);
  stats.emplace_back("totalTokens", totalTokens_);
  stats.emplace_back("totalSegments", totalSegments_);
  stats.emplace_back("processCalls", processCalls_);
  stats.emplace_back("totalWallMs", totalWallMs_);
  stats.emplace_back("whisperSampleMs", whisperSampleMs_);
  stats.emplace_back("whisperEncodeMs", whisperEncodeMs_);
  stats.emplace_back("whisperDecodeMs", whisperDecodeMs_);
  stats.emplace_back("whisperBatchdMs", whisperBatchdMs_);
  stats.emplace_back("whisperPromptMs", whisperPromptMs_);

  // Active backend identity + device memory, captured once at load()
  // by captureActiveBackendInfo(). Field shape mirrors
  // transcription-whispercpp 0.9.0's RuntimeStats:
  //   backendDevice : 0 = CPU, 1 = GPU (post-fallback truth)
  //   backendId     : 0 = CPU, 1 = Metal, 2 = CUDA, 3 = Vulkan,
  //                   4 = OpenCL, 99 = other
  // A use_gpu=true request that fell back to CPU at load() time
  // surfaces as backendDevice=0 / backendId=0 (and load() emits a
  // WARNING explaining why).
  stats.emplace_back("backendDevice", backend_device_);
  stats.emplace_back("backendId", backend_id_);
  stats.emplace_back("gpuMemTotalMb", gpu_mem_total_mb_);
  stats.emplace_back("gpuMemFreeMb", gpu_mem_free_mb_);

  return stats;
}

static void onNewSegment(
    [[maybe_unused]] whisper_context* ctx, whisper_state* state, int nNew,
    void* userData) {
  auto* bci = static_cast<BCIModel*>(userData);
  if (bci == nullptr || state == nullptr) return;

  const int nSegments = whisper_full_n_segments_from_state(state);
  if (nNew <= 0 || nSegments <= 0) return;
  const int startIndex = std::max(0, nSegments - nNew);

  for (int i = startIndex; i < nSegments; i++) {
    Transcript transcript;
    const char* text = whisper_full_get_segment_text_from_state(state, i);
    transcript.text = text != nullptr ? text : "";
    transcript.start =
        static_cast<float>(whisper_full_get_segment_t0_from_state(state, i)) *
        K_SEGMENT_TIMESTAMP_SCALE;
    transcript.end =
        static_cast<float>(whisper_full_get_segment_t1_from_state(state, i)) *
        K_SEGMENT_TIMESTAMP_SCALE;
    transcript.id = i;

    bci->emitSegment(transcript);
    bci->addTranscription(transcript);

    const int nTokens = whisper_full_n_tokens_from_state(state, i);
    bci->recordSegmentStats(nTokens);
  }
}

void BCIModel::warmup() {
  if (!ctx_) return;

  std::vector<float> silentAudio(K_WARMUP_SAMPLE_COUNT, 0.0F);
  whisper_full_params params = toWhisperFullParams(cfg_);
  params.new_segment_callback = nullptr;
  params.new_segment_callback_user_data = nullptr;

  whisper_full(ctx_.get(), params,
               silentAudio.data(),
               static_cast<int>(silentAudio.size()));
}

void BCIModel::process(const Input& rawNeuralData) {
  if (ctx_ == nullptr) {
    throw std::runtime_error("BCI Whisper context is not initialized — call load() first");
  }

  if (cancelRequested_.load(std::memory_order_relaxed)) {
    throw std::runtime_error("Job cancelled");
  }

  QLOG(qvac_lib_inference_addon_cpp::logger::Priority::DEBUG,
       "Processing neural signal (" +
           std::to_string(rawNeuralData.size()) + " bytes)");

  // Default day_idx = 0 matches NeuralProcessor::processToMel and the public
  // JS/TS docs. The reference fixtures in test/fixtures/manifest.json pass
  // day_idx=1 explicitly; callers that omit bciConfig get day 0.
  int dayIdx = 0;
  auto it = cfg_.bciConfig.find("day_idx");
  if (it != cfg_.bciConfig.end()) {
    if (auto* d = std::get_if<double>(&it->second)) {
      dayIdx = static_cast<int>(*d);
    } else if (auto* i = std::get_if<int>(&it->second)) {
      dayIdx = *i;
    }
  }

  if (neuralProcessor_.hasWeights()) {
    const int maxDay =
        static_cast<int>(neuralProcessor_.getNumDays()) - 1;
    if (maxDay >= 0 && (dayIdx < 0 || dayIdx > maxDay)) {
      QLOG(qvac_lib_inference_addon_cpp::logger::Priority::WARNING,
           "day_idx " + std::to_string(dayIdx) +
               " is outside [0, " + std::to_string(maxDay) +
               "]; it will be clamped");
    }
  }

  auto melFeatures = neuralProcessor_.processToMel(rawNeuralData, dayIdx);
  const int melBins = neuralProcessor_.getMelBins();
  const int melFrames = neuralProcessor_.getMelFrames();

  processCalls_ += 1;

  whisper_reset_timings(ctx_.get());

  const auto startTime = std::chrono::steady_clock::now();

  EncoderCallbackData cbData;
  cbData.ctx = ctx_.get();
  cbData.melData = melFeatures.data();
  cbData.melFrames = melFrames;
  cbData.melBins = melBins;

  whisper_full_params params = toWhisperFullParams(cfg_);
  params.new_segment_callback = onNewSegment;
  params.new_segment_callback_user_data = this;
  params.abort_callback = shouldAbortWhisper;
  params.abort_callback_user_data = &cancelRequested_;
  params.encoder_begin_callback = onEncoderBegin;
  params.encoder_begin_callback_user_data = &cbData;

  if (dummyAudioPad_.size() != static_cast<size_t>(K_DUMMY_AUDIO_30S)) {
    dummyAudioPad_.assign(K_DUMMY_AUDIO_30S, 0.0F);
  }

  int result = whisper_full(
      ctx_.get(), params,
      dummyAudioPad_.data(), static_cast<int>(dummyAudioPad_.size()));

  const auto endTime = std::chrono::steady_clock::now();
  totalWallMs_ +=
      std::chrono::duration<double, std::milli>(endTime - startTime).count();

  if (auto* whisperTimings = whisper_get_timings(ctx_.get());
      whisperTimings != nullptr) {
    whisperSampleMs_ += whisperTimings->sample_ms;
    whisperEncodeMs_ += whisperTimings->encode_ms;
    whisperDecodeMs_ += whisperTimings->decode_ms;
    whisperBatchdMs_ += whisperTimings->batchd_ms;
    whisperPromptMs_ += whisperTimings->prompt_ms;
  }

  if (result != 0) {
    if (cancelRequested_.load(std::memory_order_relaxed)) {
      throw std::runtime_error("Job cancelled");
    }
    throw std::runtime_error(
        "Failed to process neural signal (whisper_full returned " +
        std::to_string(result) + ")");
  }
}

std::any BCIModel::process(const std::any& input) {
  AnyInput modelInput;
  if (auto* anyInput = std::any_cast<AnyInput>(
          const_cast<std::any*>(&input))) {
    modelInput = std::move(*anyInput);
  } else if (auto* inputVector = std::any_cast<Input>(
                 const_cast<std::any*>(&input))) {
    modelInput.input = std::move(*inputVector);
  } else {
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        std::string("Invalid input type for BCIModel::process: ") +
            input.type().name());
  }

  const auto previousOutputCallback = on_segment_;
  const bool shouldOverrideCallback =
      static_cast<bool>(modelInput.outputCallback);
  if (shouldOverrideCallback) {
    on_segment_ = modelInput.outputCallback;
  }

  // Clear the cancel flag FIRST so a cancel() call that races with reset()
  // is not silently lost. process(Input&) still checks cancelRequested_ at
  // the top, so a cancel that arrives between these two statements aborts
  // the upcoming whisper_full call via shouldAbortWhisper.
  cancelRequested_.store(false, std::memory_order_relaxed);
  reset();
  try {
    process(modelInput.input);
  } catch (...) {
    if (shouldOverrideCallback) {
      on_segment_ = previousOutputCallback;
    }
    throw;
  }

  if (shouldOverrideCallback) {
    on_segment_ = previousOutputCallback;
  }

  return output_;
}

void BCIModel::saveLoadParams(const BCIConfig& config) {
  setConfig(config);
}

void BCIModel::cancel() const {
  cancelRequested_.store(true, std::memory_order_relaxed);
}

bool BCIModel::configContextIsChanged(
    const BCIConfig& oldCfg, const BCIConfig& newCfg) {
  const std::vector<std::string> contextKeys = {
      "model", "use_gpu", "flash_attn", "gpu_device"};
  return std::ranges::any_of(contextKeys, [&](const std::string& key) {
    const auto oldIt = oldCfg.whisperContextCfg.find(key);
    const auto newIt = newCfg.whisperContextCfg.find(key);
    if (oldIt != oldCfg.whisperContextCfg.end() &&
        newIt != newCfg.whisperContextCfg.end()) {
      return oldIt->second != newIt->second;
    }
    return (oldIt != oldCfg.whisperContextCfg.end()) !=
           (newIt != newCfg.whisperContextCfg.end());
  });
}

void BCIModel::resetContext() { ctx_.reset(); }

void BCIModel::setConfig(const BCIConfig& config) {
  bool contextChanged = configContextIsChanged(cfg_, config);
  cfg_ = config;
  if (contextChanged) reload();
}

} // namespace qvac_lib_inference_addon_bci
