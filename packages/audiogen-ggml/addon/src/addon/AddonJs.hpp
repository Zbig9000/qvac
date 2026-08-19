#pragma once

#include <any>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <inference-addon-cpp/JsInterface.hpp>
#include <inference-addon-cpp/JsUtils.hpp>
#include <inference-addon-cpp/ModelInterfaces.hpp>
#include <inference-addon-cpp/addon/AddonJs.hpp>
#include <inference-addon-cpp/handlers/JsOutputHandlerImplementations.hpp>
#include <inference-addon-cpp/handlers/OutputHandler.hpp>
#include <inference-addon-cpp/queue/OutputCallbackJs.hpp>
#include <js.h>

#include "js-interface/JSAdapter.hpp"
#include "model-interface/AudioGenProgress.hpp"
#include "model-interface/acestep/AcestepModel.hpp"
#ifdef AUDIOGEN_HAS_MINIMAX
#include "model-interface/minimax/MinimaxModel.hpp"
#endif

namespace qvac::audiogenggml::addon_js {

namespace js = qvac_lib_inference_addon_cpp::js;

using acestep::AcestepModel;
#ifdef AUDIOGEN_HAS_MINIMAX
using minimax::MinimaxModel;
#endif

inline constexpr double kMaximumSafeInteger = 9007199254740991.0;
inline constexpr int kMaximumMinimaxInferenceSteps = 1000;

inline std::optional<double> readOptionalNumber(
    js::Object object, js_env_t* env, const char* name) {
  js_value_t* raw = object.getProperty(env, name);
  if (js::is<js::Undefined>(env, raw) || js::is<js::Null>(env, raw)) {
    return std::nullopt;
  }
  if (!js::is<js::Number>(env, raw)) {
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        std::string(name) + " must be a number");
  }
  return js::Number::fromValue(raw).as<double>(env);
}

inline int64_t checkedSafeInteger(double value, const char* name) {
  if (!std::isfinite(value) || std::trunc(value) != value ||
      std::fabs(value) > kMaximumSafeInteger) {
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        std::string(name) + " must be a safe integer");
  }
  return static_cast<int64_t>(value);
}

inline int64_t checkedPositiveSafeInteger(double value, const char* name) {
  const int64_t integer = checkedSafeInteger(value, name);
  if (integer <= 0) {
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        std::string(name) + " must be greater than zero");
  }
  return integer;
}

inline int checkedMinimaxInferenceSteps(double value) {
  const int64_t integer = checkedSafeInteger(value, "inferenceSteps");
  if (integer < 0 || integer > kMaximumMinimaxInferenceSteps) {
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        "inferenceSteps must be between 0 and 1000");
  }
  return static_cast<int>(integer);
}

inline float checkedMinimaxCfgScale(double value) {
  const double maximum = std::numeric_limits<float>::max();
  const double minimum = std::numeric_limits<float>::denorm_min();
  if (!std::isfinite(value) || value < 0.0 || value > maximum ||
      (value > 0.0 && value < minimum)) {
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        "cfgScale must be 0 or a positive float32 value");
  }
  return static_cast<float>(value);
}

inline std::vector<int>
copyAudioCodes(js_env_t* env, js::TypedArray<int32_t> array) {
  int32_t* data = nullptr;
  size_t len = 0;
  if (js_get_typedarray_info(
          env,
          array,
          nullptr,
          reinterpret_cast<void**>(&data),
          &len,
          nullptr,
          nullptr) != 0) {
    throw std::runtime_error("audioCodes must be an Int32Array");
  }
  return {data, data + len};
}

inline std::vector<float>
copyFloat32Pcm(js_env_t* env, js::TypedArray<float> array, const char* name) {
  float* data = nullptr;
  size_t len = 0;
  if (js_get_typedarray_info(
          env,
          array,
          nullptr,
          reinterpret_cast<void**>(&data),
          &len,
          nullptr,
          nullptr) != 0) {
    throw std::runtime_error(std::string(name) + " must be a Float32Array");
  }
  return {data, data + len};
}

struct JsAudioOutputHandler
    : qvac_lib_inference_addon_cpp::out_handl::JsBaseOutputHandler<
          std::vector<int16_t>> {
  JsAudioOutputHandler(
      std::function<int()> sampleRate, std::function<int()> channels)
      : qvac_lib_inference_addon_cpp::out_handl::JsBaseOutputHandler<
            std::vector<int16_t>>(
            [this, sampleRate = std::move(sampleRate),
             channels = std::move(channels)](
                const std::vector<int16_t>& data) -> js_value_t* {
              auto result = js::Object::create(this->env_);
              std::span<const int16_t> outputSpan(data.data(), data.size());
              auto typedArray =
                  js::TypedArray<int16_t>::create(this->env_, outputSpan);
              result.setProperty(this->env_, "outputArray", typedArray);
              result.setProperty(
                  this->env_, "sampleRate",
                  js::Number::create(this->env_, sampleRate()));
              result.setProperty(
                  this->env_, "channels",
                  js::Number::create(this->env_, channels()));
              return result;
            }) {}
};

struct JsProgressOutputHandler
    : qvac_lib_inference_addon_cpp::out_handl::JsBaseOutputHandler<
          AudioGenProgress> {
  JsProgressOutputHandler()
      : qvac_lib_inference_addon_cpp::out_handl::JsBaseOutputHandler<
            AudioGenProgress>(
            [this](const AudioGenProgress& p) -> js_value_t* {
              auto result = js::Object::create(this->env_);
              result.setProperty(
                  this->env_, "progressStage",
                  js::String::create(this->env_, p.stage));
              result.setProperty(
                  this->env_, "progressStep",
                  js::Number::create(this->env_, p.step));
              result.setProperty(
                  this->env_, "progressTotal",
                  js::Number::create(this->env_, p.total));
              return result;
            }) {}
};

inline js_value_t* createInstance(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;
  using namespace std;

  JsArgsParser args(env, info);
  auto configurationParams = args.getJsObject(1, "configurationParams");

  JSAdapter adapter;
  const EngineType engineType = adapter.readEngineType(configurationParams, env);
  unique_ptr<model::IModel> model;
  function<int()> sampleRate;
  function<int()> channels;
  function<void(function<void(const AudioGenProgress&)>)> setProgressSink;

  if (engineType == EngineType::Minimax) {
#ifdef AUDIOGEN_HAS_MINIMAX
    auto minimaxModel =
        make_unique<MinimaxModel>(
            adapter.buildMinimaxConfig(configurationParams, env));
    MinimaxModel* modelPtr = minimaxModel.get();
    sampleRate = [modelPtr]() { return modelPtr->sampleRate(); };
    channels = [modelPtr]() { return modelPtr->channels(); };
    setProgressSink = [modelPtr](
                          function<void(const AudioGenProgress&)> sink) {
      modelPtr->setProgressSink(std::move(sink));
    };
    model = std::move(minimaxModel);
#else
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        "MiniMax-Music3 is available on desktop builds only");
#endif
  } else {
    auto acestepModel =
        make_unique<AcestepModel>(
            adapter.buildAcestepConfig(configurationParams, env));
    AcestepModel* modelPtr = acestepModel.get();
    sampleRate = [modelPtr]() { return modelPtr->sampleRate(); };
    channels = [modelPtr]() { return modelPtr->channels(); };
    setProgressSink = [modelPtr](
                          function<void(const AudioGenProgress&)> sink) {
      modelPtr->setProgressSink(std::move(sink));
    };
    model = std::move(acestepModel);
  }

  out_handl::OutputHandlers<out_handl::JsOutputHandlerInterface> outHandlers;
  outHandlers.add(make_shared<JsAudioOutputHandler>(
      std::move(sampleRate), std::move(channels)));
  outHandlers.add(make_shared<JsProgressOutputHandler>());
  unique_ptr<OutputCallBackInterface> callback = make_unique<OutputCallBackJs>(
      env, args.get(0, "jsHandle"), args.getFunction(2, "outputCallback"),
      std::move(outHandlers));

  auto addon = make_unique<AddonJs>(env, std::move(callback), std::move(model));
  auto outputQueue = addon->addonCpp->outputQueue;
  setProgressSink([outputQueue](const AudioGenProgress& p) {
    outputQueue->queueResult(std::any(p));
  });

  return JsInterface::createInstance(env, std::move(addon));
}
JSCATCH

inline js_value_t* runJob(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;

  JsArgsParser args(env, info);
  AddonJs& instance = JsInterface::getInstance(env, args.get(0, "instance"));
  auto [type, jsInput] = JsInterface::getInput(args);

  if (type != "text") {
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        "Unknown input type: " + type);
  }

  auto jobObj = args.getJsObject(1, "inputObj");
  auto optStr = [&](const char* key) -> std::optional<std::string> {
    return jobObj.getOptionalPropertyAs<js::String, std::string>(env, key);
  };
  auto optNum = [&](const char* key) -> std::optional<double> {
    js_value_t* raw = jobObj.getProperty(env, key);
    if (js::is<js::Undefined>(env, raw) || js::is<js::Null>(env, raw)) {
      return std::nullopt;
    }
    if (js::is<js::Number>(env, raw)) {
      return js::Number::fromValue(raw).as<double>(env);
    }
    return std::nullopt;
  };
  auto optBool = [&](const char* key) -> std::optional<bool> {
    js_value_t* raw = jobObj.getProperty(env, key);
    if (js::is<js::Undefined>(env, raw) || js::is<js::Null>(env, raw)) {
      return std::nullopt;
    }
    if (js::is<js::Boolean>(env, raw)) {
      return js::Boolean{env, raw}.as<bool>(env);
    }
    return std::nullopt;
  };

#ifdef AUDIOGEN_HAS_MINIMAX
  if (dynamic_cast<MinimaxModel*>(&instance.addonCpp->model.get())) {
    MinimaxModel::AnyInput modelInput;
    modelInput.caption = js::String(env, jsInput).as<std::string>(env);
    if (auto value = optStr("lyrics")) modelInput.lyrics = *value;
    if (auto value = readOptionalNumber(jobObj, env, "seed")) {
      modelInput.seed = checkedSafeInteger(*value, "seed");
    }
    if (auto value = readOptionalNumber(jobObj, env, "maxFrames")) {
      modelInput.maxFrames = checkedPositiveSafeInteger(*value, "maxFrames");
    }
    if (auto value = readOptionalNumber(jobObj, env, "inferenceSteps")) {
      modelInput.inferenceSteps = checkedMinimaxInferenceSteps(*value);
    }
    if (auto value = readOptionalNumber(jobObj, env, "cfgScale")) {
      modelInput.cfgScale = checkedMinimaxCfgScale(*value);
    }
    return instance.runJob(std::any(std::move(modelInput)));
  }
#endif

  AcestepModel::AnyInput modelInput;
  modelInput.caption = js::String(env, jsInput).as<std::string>(env);
  if (auto v = optStr("lyrics")) modelInput.lyrics = *v;
  if (auto v = optStr("vocalLanguage")) modelInput.vocalLanguage = *v;
  if (auto v = optStr("keyscale")) modelInput.keyscale = *v;
  if (auto v = optStr("timesignature")) modelInput.timesignature = *v;
  if (auto v = optNum("seed")) modelInput.seed = static_cast<long long>(*v);
  if (auto v = optNum("bpm")) modelInput.bpm = static_cast<int>(*v);
  if (auto v = optNum("duration")) modelInput.duration = static_cast<float>(*v);
  if (auto v = optNum("lmTemperature"))
    modelInput.lmTemperature = static_cast<float>(*v);
  if (auto v = optNum("lmTopP"))
    modelInput.lmTopP = static_cast<float>(*v);
  if (auto v = optNum("lmTopK"))
    modelInput.lmTopK = static_cast<int>(*v);
  if (auto v = optNum("lmCfgScale"))
    modelInput.lmCfgScale = static_cast<float>(*v);
  if (auto v = optBool("lmPhase1"))
    modelInput.lmPhase1 = *v;
  if (auto v = optBool("dcwEnabled"))
    modelInput.dcwEnabled = *v;
  if (auto v = optNum("dcwScaler"))
    modelInput.dcwScaler = static_cast<float>(*v);
  if (auto v = optNum("dcwHighScaler"))
    modelInput.dcwHighScaler = static_cast<float>(*v);
  if (auto codes = jobObj.getOptionalProperty<js::TypedArray<int32_t>>(
          env, "audioCodes")) {
    modelInput.audioCodes = copyAudioCodes(env, *codes);
  }
  if (auto ref = jobObj.getOptionalProperty<js::TypedArray<float>>(
          env, "referenceAudio")) {
    modelInput.referenceAudio = copyFloat32Pcm(env, *ref, "referenceAudio");
  }
  if (auto src = jobObj.getOptionalProperty<js::TypedArray<float>>(
          env, "sourceAudio")) {
    modelInput.sourceAudio = copyFloat32Pcm(env, *src, "sourceAudio");
  }
  if (auto v = optStr("taskType"))
    modelInput.taskType = *v;
  if (auto v = optNum("audioCoverStrength"))
    modelInput.audioCoverStrength = static_cast<float>(*v);
  if (auto v = optNum("coverNoiseStrength"))
    modelInput.coverNoiseStrength = static_cast<float>(*v);
  return instance.runJob(std::any(std::move(modelInput)));
}
JSCATCH

inline js_value_t* activate(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;

  JsArgsParser args(env, info);
  AddonJs& instance = JsInterface::getInstance(env, args.get(0, "instance"));

  return js::JsAsyncTask::run(
      env, [addonCpp = instance.addonCpp]() { addonCpp->activate(); });
}
JSCATCH

inline js_value_t* reload(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;
  using namespace std;

  JsArgsParser args(env, info);
  AddonJs& instance = JsInterface::getInstance(env, args.get(0, "instance"));
  auto configurationParams = args.getJsObject(1, "configurationParams");
  JSAdapter adapter;
  const EngineType engineType = adapter.readEngineType(configurationParams, env);

  if (engineType == EngineType::Minimax) {
#ifdef AUDIOGEN_HAS_MINIMAX
    auto newConfig = adapter.buildMinimaxConfig(configurationParams, env);
    return js::JsAsyncTask::run(
        env,
        [addonCpp = instance.addonCpp,
         newConfig = std::move(newConfig)]() mutable {
          auto* model =
              dynamic_cast<MinimaxModel*>(&addonCpp->model.get());
          if (model == nullptr) {
            throw qvac_errors::StatusError(
                qvac_errors::general_error::InvalidArgument,
                "reload cannot change the audiogen engine type");
          }
          model->reload(std::move(newConfig));
        });
#else
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        "MiniMax-Music3 is available on desktop builds only");
#endif
  }

  auto newConfig = adapter.buildAcestepConfig(configurationParams, env);
  return js::JsAsyncTask::run(
      env,
      [addonCpp = instance.addonCpp,
       newConfig = std::move(newConfig)]() mutable {
        auto* model = dynamic_cast<AcestepModel*>(&addonCpp->model.get());
        if (model == nullptr) {
          throw qvac_errors::StatusError(
              qvac_errors::general_error::InvalidArgument,
              "reload cannot change the audiogen engine type");
        }
        model->setConfig(std::move(newConfig));
        model->reload();
      });
}
JSCATCH

}  // namespace qvac::audiogenggml::addon_js
