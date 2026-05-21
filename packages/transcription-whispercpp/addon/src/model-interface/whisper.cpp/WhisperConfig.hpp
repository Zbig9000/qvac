#ifndef WHISPERCONFIG_H
#define WHISPERCONFIG_H

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>

#include <whisper.h>
namespace qvac_lib_inference_addon_whisper {

class JSAdapter;

using JSValueVariant =
    std::variant<std::monostate, int, double, std::string, bool>;

/*
 Needs to handle both
 - whisper_full_params
 - whisper_context_params
 and probably more later.
*/

template <typename Params>
using HandlerFunction = std::function<void(Params&, const JSValueVariant&)>;

template <typename Params>
using HandlersMap = std::unordered_map<std::string, HandlerFunction<Params>>;

struct WhisperConfig {
  std::map<std::string, JSValueVariant> miscConfig;
  std::map<std::string, JSValueVariant> whisperMainCfg;
  std::map<std::string, JSValueVariant> vadCfg;
  std::map<std::string, JSValueVariant> whisperContextCfg;

  // QVAC-18993: absolute path to the addon's prebuilds folder (set from JS
  // as `configurationParams.backendsDir`, typically `path.join(__dirname,
  // 'prebuilds')`). Combined with the compile-time `BACKENDS_SUBDIR`
  // (`<bare_target>/<module_name>`) at load time to locate the per-arch
  // ggml-backend `.so` modules for `ggml_backend_load_all_from_path()`.
  // Only consulted on Android (`__ANDROID__`) where ggml is built with
  // `GGML_BACKEND_DL=ON` and no backend is statically registered. Empty on
  // every other platform (where ggml's static constructors register the CPU
  // backend before `whisper_init`). Kept as a sibling of the existing
  // handler-driven maps so it doesn't have to be dispatched through
  // `WHISPER_CONTEXT_HANDLERS.at(...)`.
  std::string backendsDir;
};

struct MiscConfig {
  bool captionModeEnabled;
  int seed; // this is an internal c++ calls that is going to be handled
            // seperately.
  // seed is not passed to the main functions but is set ahead of time before
  // the function call. use this struct for future options not passed into
  // whisper.cpp but nevertheless effects model input/output
};

MiscConfig defaultMiscConfig();

// HandlersMap are visible to the js side only.
// uses handlers map form js side.
whisper_full_params toWhisperFullParams(const WhisperConfig& whisperConfig);

// HandlersMap are visible to the js side only, so this function
// uses handlers
whisper_context_params
toWhisperContextParams(const WhisperConfig& whisperConfig);

MiscConfig toMiscConfig(const WhisperConfig& whisperConfig);

std::string convertVariantToString(const JSValueVariant& value);
} // namespace qvac_lib_inference_addon_whisper

#endif // WHISPERCONFIG_H
