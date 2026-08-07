#pragma once

#include <any>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "inference-addon-cpp/ModelInterfaces.hpp"
#include "inference-addon-cpp/RuntimeStats.hpp"
#include "model-interface/audio8/Audio8Config.hpp"

namespace tts_cpp::audio8 {
class Engine;
}

namespace qvac::ttsggml::audio8 {

inline constexpr int AUDIO8_NATIVE_SAMPLE_RATE = 44100;

class Audio8Model
    : public qvac_lib_inference_addon_cpp::model::IModel,
      public qvac_lib_inference_addon_cpp::model::IModelCancel,
      public qvac_lib_inference_addon_cpp::model::IModelAsyncLoad {
public:
  using Input = std::string;
  using Output = std::vector<int16_t>;

  // Per-call overrides of the constructor's reference recording. Empty means
  // "keep the configured voice"; an audio path with no text is rejected.
  struct VoiceOverride {
    std::string referenceAudio;
    std::string referenceText;
  };

  struct AnyInput {
    std::string text;
    VoiceOverride voice;
  };

  explicit Audio8Model(Audio8Config config);
  ~Audio8Model() noexcept override;

  std::string getName() const override { return "Audio8Model"; }
  std::any process(const std::any& input) override;
  qvac_lib_inference_addon_cpp::RuntimeStats runtimeStats() const override;

  void cancel() const override;

  void load();
  void unload();
  void reload();
  bool isLoaded() const {
    std::lock_guard lk(engineMu_);
    return static_cast<bool>(engine_);
  }

  // IModelAsyncLoad — see the equivalent comment on ChatterboxModel.
  void waitForLoadInitialization() override { load(); }
  void setWeightsForFile(
      const std::string&,
      std::unique_ptr<std::basic_streambuf<char>>&&) override {}

  void setConfig(Audio8Config config);
  const Audio8Config& config() const { return cfg_; }

  int sampleRate() const { return sampleRate_; }

  static void validateConfig(const Audio8Config& cfg);
  // Shared by validateConfig and the per-call path, which can name a
  // different recording. An encoder GGUF and a transcript are both required
  // as soon as either voice field is set.
  static void validateVoice(
      const std::string& audio, const std::string& text,
      const std::string& encoderPath);
  // The voice a call synthesizes with, merging the per-call override over the
  // configured one. Public so the merge rule is directly testable.
  VoiceOverride resolveVoice(const VoiceOverride& perCall) const;

private:
  Output synthesize(const AnyInput& input);

  void loadLocked();
  void unloadLocked();

  Audio8Config cfg_;

  mutable std::mutex engineMu_;
  std::shared_ptr<tts_cpp::audio8::Engine> engine_;

  std::atomic_bool jobInProgress_{false};

  // Mirrors SupertonicModel::cancelRequested_ (see that header).
  mutable std::atomic_bool cancelRequested_{false};

  double totalTime_ = 0.0;
  double audioDurationMs_ = 0.0;
  int64_t totalSamples_ = 0;
  double realTimeFactor_ = 0.0;
  // Frames per second, not characters per second as on the text-paced
  // engines: Audio8 generates on a fixed 46 ms codec frame grid.
  double tokensPerSecond_ = 0.0;
  int generatedFrames_ = 0;
  int sampleRate_ = AUDIO8_NATIVE_SAMPLE_RATE;

  int backendDevice_ = 0;
  int backendId_ = 0;
  std::string backendName_ = "CPU";
  bool gpuUnsupported_ = false;
};

} // namespace qvac::ttsggml::audio8
