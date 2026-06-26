#pragma once

#include <optional>
#include <string>

namespace qvac::ttsggml::supertonic {

struct SupertonicConfig {
  std::string modelGgufPath;
  std::string voice;
  std::string language = "en";
  std::optional<int> steps;
  std::optional<float> speed;
  std::optional<int> seed;
  std::optional<int> threads;
  std::optional<int> nGpuLayers;
  std::optional<int> outputSampleRate;
  /**
   * Tri-state GPU intent (mirrors ChatterboxConfig::useGpu):
   *   - std::nullopt: unspecified, let the engine use its library default.
   *   - true:         if nGpuLayers unset, maps to nGpuLayers=99. Honored on
   *                   GPU-capable hosts (Metal on Apple, Vulkan/CUDA on
   *                   desktop, Vulkan/OpenCL on Android), delegated to
   *                   tts-cpp's per-vendor allowlist (Adreno/Xclipse/Mali);
   *                   it falls back to CPU on GPUs it can't drive.
   *   - false:        if nGpuLayers unset, forces nGpuLayers=0 (CPU).
   *
   * Conflicts with nGpuLayers (true + 0, or false + !=0) are rejected
   * by validateConfig so callers can't silently get the opposite
   * backend they asked for.
   */
  std::optional<bool> useGpu;
  std::string noiseNpyPath;
  std::string backendsDir;
  std::string openclCacheDir;

  // LavaSR neural speech enhancement (QVAC-16579). When `enhancerGgufPath`
  // is set and `enhance` is not explicitly false, the model loads the
  // enhancer GGUF and bandwidth-extends the synthesized PCM to 48 kHz before
  // returning it. Empty path disables enhancement (full backward compat).
  //
  // NOTE: while the enhancer is active the output is ALWAYS 48 kHz; any
  // `outputSampleRate` above is ignored in that case (the JS layer warns).
  // A configurable post-enhancement resample is a planned follow-up.
  std::string enhancerGgufPath;
  std::optional<bool> enhance;
};

}
