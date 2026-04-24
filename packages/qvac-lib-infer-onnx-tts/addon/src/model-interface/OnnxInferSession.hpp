#pragma once

#include "IOnnxInferSession.hpp"
#include "OrtTypes.hpp"
#include "onnxruntime_cxx_api.h"

#include <unordered_map>
#include <unordered_set>

namespace qvac::ttslib::chatterbox {

class OnnxInferSession : public IOnnxInferSession {
public:
  explicit OnnxInferSession(const std::string &modelPath, bool useGPU = false,
                            int numThreads = 0);
  ~OnnxInferSession() override = default;

  void run() override;

  const std::vector<std::string> &getInputNames() const override;
  const std::vector<std::string> &getOutputNames() const override;

  OrtTensor getInput(const std::string &inputName) override;
  OrtTensor getOutput(const std::string &outputName) override;

  void initInputTensors(
      const std::vector<std::vector<int64_t>> &inputShapes) override;

  void setOutputToInputChain(
      const std::vector<std::pair<std::string, std::string>> &mapping) override;
  void clearChainedInputs() override;
  bool isInputChained(const std::string &inputName) const override;

private:
  void moveChainedOutputsIntoInputs();

  std::unique_ptr<Ort::Session> session_;

  std::vector<OrtTensor> inputTensors_;
  std::vector<OrtTensor> outputTensors_;

  std::vector<Ort::Value> inputTensorsValues_;
  std::vector<Ort::Value> outputsTensorsValues_;

  std::vector<std::string> inputNames_;
  std::vector<std::string> outputNames_;

  // Cached `const char*` arrays pointing into the stable storage of
  // `inputNames_` / `outputNames_`. Populated once in the constructor and
  // fed directly to `Ort::Session::Run()` on every step, so the hot loop
  // avoids rebuilding the vectors and calling c_str() ~120 times per token.
  std::vector<const char *> inputNamesC_;
  std::vector<const char *> outputNamesC_;

  // ONNX element type per input index, captured once in the constructor.
  // `initInputTensors` used to re-interrogate the session every step via
  // `GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType()` — an
  // FFI round-trip per input on every autoregressive step. The element
  // type is a fixed property of the model graph, so we cache it and hand
  // the cached value to `Ort::Value::CreateTensor` directly.
  std::vector<ONNXTensorElementDataType> inputElementTypes_;

  std::unordered_map<std::string, size_t> inputIndexByName_;
  std::unordered_map<std::string, size_t> outputIndexByName_;

  // (outputName, inputName) pairs describing output->input chaining.
  std::vector<std::pair<std::string, std::string>> outputToInputChain_;
  std::unordered_set<std::string> chainedInputNames_;

  Ort::AllocatorWithDefaultOptions allocator_;
};

} // namespace qvac::ttslib::chatterbox