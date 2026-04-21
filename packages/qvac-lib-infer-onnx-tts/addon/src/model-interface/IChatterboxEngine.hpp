#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "AudioResult.hpp"

namespace qvac::ttslib::chatterbox {

struct ChatterboxConfig {
  std::string language;
  std::vector<float> referenceAudio;
  std::string tokenizerPath;
  std::string speechEncoderPath;
  std::string embedTokensPath;
  std::string conditionalDecoderPath;
  std::string languageModelPath;
  bool lazySessionLoading = false;
  bool useGPU = false;
  int numThreads = 0;
  // Chain `present.*` outputs back into the matching `past_key_values.*`
  // inputs on the language-model session after every run(), skipping the
  // per-step copy. Default ON; exposed mainly so benchmarks can toggle the
  // optimization for A/B measurement.
  bool kvCacheChaining = true;
};

class IChatterboxEngine {
public:
  IChatterboxEngine() = default;
  virtual ~IChatterboxEngine() = default;
  virtual void load(const ChatterboxConfig &cfg) = 0;
  virtual void unload() = 0;
  virtual bool isLoaded() const = 0;
  virtual AudioResult synthesize(const std::string &text) = 0;
};

} // namespace qvac::ttslib::chatterbox
