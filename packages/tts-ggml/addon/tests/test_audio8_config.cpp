// Constructor-validation tests for Audio8Model. Same shape as
// test_parler_config.cpp: validateConfig is driven via the public
// constructor, and the GGUF parse is deferred to load() so stub files are
// enough to exercise every branch that does not touch weights.
//
// Real-GGUF round-trip is gated behind QVAC_TEST_AUDIO8_LM_GGUF.

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "inference-addon-cpp/Errors.hpp"
#include "model-interface/audio8/Audio8Config.hpp"
#include "model-interface/audio8/Audio8Model.hpp"

using qvac::ttsggml::audio8::Audio8Config;
using qvac::ttsggml::audio8::Audio8Model;
using qvac_errors::StatusError;

namespace {

std::filesystem::path tempPath(const std::string& suffix) {
  auto dir =
      std::filesystem::temp_directory_path() / "qvac-tts-ggml-audio8-tests";
  std::filesystem::create_directories(dir);
  return dir / suffix;
}

std::string stubFile(const std::string& name) {
  const auto path = tempPath(name);
  std::ofstream(path, std::ios::binary) << "stub";
  return path.string();
}

std::string envOrEmpty(const char* name) {
  if (const char* v = std::getenv(name))
    return v;
  return "";
}

Audio8Config minimallyValidStubConfig() {
  Audio8Config cfg;
  cfg.lmModelPath = stubFile("audio8-lm-stub.gguf");
  cfg.codecDecoderPath = stubFile("audio8-decoder-stub.gguf");
  return cfg;
}

Audio8Config cloningStubConfig() {
  Audio8Config cfg = minimallyValidStubConfig();
  cfg.codecEncoderPath = stubFile("audio8-encoder-stub.gguf");
  cfg.referenceAudio = stubFile("audio8-reference-stub.wav");
  cfg.referenceText = "What the recording says.";
  return cfg;
}

} // namespace

TEST(Audio8Validate, EmptyLmPathRejected) {
  Audio8Config cfg;
  cfg.codecDecoderPath = stubFile("audio8-decoder-stub.gguf");
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
}

TEST(Audio8Validate, EmptyDecoderPathRejected) {
  Audio8Config cfg;
  cfg.lmModelPath = stubFile("audio8-lm-stub.gguf");
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
}

TEST(Audio8Validate, NonexistentPathsRejected) {
  auto cfg = minimallyValidStubConfig();
  cfg.lmModelPath = "/definitely/does/not/exist/audio8-lm.gguf";
  EXPECT_THROW(Audio8Model{cfg}, StatusError);

  cfg = minimallyValidStubConfig();
  cfg.codecDecoderPath = "/definitely/does/not/exist/audio8-decoder.gguf";
  EXPECT_THROW(Audio8Model{cfg}, StatusError);

  cfg = minimallyValidStubConfig();
  cfg.codecEncoderPath = "/definitely/does/not/exist/audio8-encoder.gguf";
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
}

TEST(Audio8Validate, TextOnlyConfigAccepted) {
  EXPECT_NO_THROW(Audio8Model{minimallyValidStubConfig()});
}

TEST(Audio8Validate, CloningConfigAccepted) {
  EXPECT_NO_THROW(Audio8Model{cloningStubConfig()});
}

TEST(Audio8Validate, ReferenceAudioNeedsEncoder) {
  auto cfg = cloningStubConfig();
  cfg.codecEncoderPath.clear();
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
}

TEST(Audio8Validate, ReferenceAudioNeedsTranscript) {
  // The model conditions on the transcript as the turn the reference answers,
  // so an empty one would degrade the clone silently. Reject it instead.
  auto cfg = cloningStubConfig();
  cfg.referenceText.clear();
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
}

TEST(Audio8Validate, ReferenceTextWithoutAudioRejected) {
  auto cfg = minimallyValidStubConfig();
  cfg.referenceText = "Nothing to attach this to.";
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
}

TEST(Audio8Validate, NonexistentReferenceAudioRejected) {
  auto cfg = cloningStubConfig();
  cfg.referenceAudio = "/definitely/does/not/exist/voice.wav";
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
}

TEST(Audio8Validate, NegativeTemperatureRejected) {
  auto cfg = minimallyValidStubConfig();
  cfg.temperature = -0.1f;
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
}

TEST(Audio8Validate, NegativeTopKRejected) {
  auto cfg = minimallyValidStubConfig();
  cfg.topK = -1;
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
}

TEST(Audio8Validate, TopPOutOfRangeRejected) {
  auto cfg = minimallyValidStubConfig();
  cfg.topP = 0.0f;
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
  cfg.topP = 1.5f;
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
  cfg.topP = 0.9f;
  EXPECT_NO_THROW(Audio8Model{cfg});
}

TEST(Audio8Validate, MaxFramesNonNegative) {
  auto cfg = minimallyValidStubConfig();
  cfg.maxFrames = -1;
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
  cfg.maxFrames = 0; // engine default
  EXPECT_NO_THROW(Audio8Model{cfg});
  cfg.maxFrames = 128;
  EXPECT_NO_THROW(Audio8Model{cfg});
}

TEST(Audio8Validate, OutputSampleRateBand) {
  auto cfg = minimallyValidStubConfig();
  cfg.outputSampleRate = 4000;
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
  cfg.outputSampleRate = 0; // native rate
  EXPECT_NO_THROW(Audio8Model{cfg});
  cfg.outputSampleRate = 16000;
  EXPECT_NO_THROW(Audio8Model{cfg});
}

TEST(Audio8Validate, UseGpuNGpuLayersConflictRejected) {
  auto cfg = minimallyValidStubConfig();
  cfg.useGpu = true;
  cfg.nGpuLayers = 0;
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
  cfg.useGpu = false;
  cfg.nGpuLayers = 99;
  EXPECT_THROW(Audio8Model{cfg}, StatusError);
}

TEST(Audio8Validate, GpuIntentAcceptedAtConstruction) {
  // The engine is CPU-only today and warns rather than failing, so GPU intent
  // must not be rejected here.
  auto cfg = minimallyValidStubConfig();
  cfg.useGpu = true;
  EXPECT_NO_THROW(Audio8Model{cfg});
  cfg.useGpu.reset();
  cfg.nGpuLayers = 99;
  EXPECT_NO_THROW(Audio8Model{cfg});
}

TEST(Audio8Validate, LoadIsDeferredAndStubFailsToParse) {
  auto cfg = minimallyValidStubConfig();
  std::unique_ptr<Audio8Model> m;
  EXPECT_NO_THROW(m = std::make_unique<Audio8Model>(cfg));
  ASSERT_NE(m, nullptr);
  EXPECT_FALSE(m->isLoaded());
  EXPECT_THROW(m->load(), StatusError);
  EXPECT_FALSE(m->isLoaded());
}

TEST(Audio8Validate, ConfigDefaultsAllUnset) {
  Audio8Config cfg;
  EXPECT_FALSE(cfg.greedy.has_value());
  EXPECT_FALSE(cfg.seed.has_value());
  EXPECT_FALSE(cfg.threads.has_value());
  EXPECT_FALSE(cfg.temperature.has_value());
  EXPECT_FALSE(cfg.topK.has_value());
  EXPECT_FALSE(cfg.topP.has_value());
  EXPECT_FALSE(cfg.maxFrames.has_value());
  EXPECT_FALSE(cfg.outputSampleRate.has_value());
  EXPECT_FALSE(cfg.nGpuLayers.has_value());
  EXPECT_FALSE(cfg.useGpu.has_value());
}

TEST(Audio8Voice, PerCallTranscriptOverridesConfigured) {
  Audio8Model model{cloningStubConfig()};
  EXPECT_NO_THROW(
      Audio8Model::validateVoice(
          model.config().referenceAudio,
          "A corrected transcript.",
          model.config().codecEncoderPath));
}

TEST(Audio8Voice, PerCallAudioWithoutTranscriptRejected) {
  const auto cfg = cloningStubConfig();
  EXPECT_THROW(
      Audio8Model::validateVoice(cfg.referenceAudio, "", cfg.codecEncoderPath),
      StatusError);
}

TEST(Audio8Voice, NoPerCallFieldsKeepTheConfiguredVoice) {
  const auto cfg = cloningStubConfig();
  const Audio8Model model{cfg};
  const Audio8Model::VoiceOverride perCall;
  const auto voice = model.resolveVoice(perCall);
  EXPECT_EQ(voice.referenceAudio, cfg.referenceAudio);
  EXPECT_EQ(voice.referenceText, cfg.referenceText);
}

TEST(Audio8Voice, PerCallTranscriptKeepsTheConfiguredRecording) {
  const auto cfg = cloningStubConfig();
  const Audio8Model model{cfg};
  Audio8Model::VoiceOverride perCall;
  perCall.referenceText = "A corrected transcript.";
  const auto voice = model.resolveVoice(perCall);
  EXPECT_EQ(voice.referenceAudio, cfg.referenceAudio);
  EXPECT_EQ(voice.referenceText, perCall.referenceText);
}

TEST(Audio8Voice, PerCallRecordingReplacesBothHalves) {
  const Audio8Model model{cloningStubConfig()};
  Audio8Model::VoiceOverride perCall;
  perCall.referenceAudio = stubFile("audio8-other-reference-stub.wav");
  // The configured transcript describes the configured recording, so a new
  // recording arriving without its own transcript is rejected rather than
  // silently cloned against the wrong text.
  EXPECT_THROW(model.resolveVoice(perCall), StatusError);

  perCall.referenceText = "What the other one says.";
  const auto voice = model.resolveVoice(perCall);
  EXPECT_EQ(voice.referenceAudio, perCall.referenceAudio);
  EXPECT_EQ(voice.referenceText, perCall.referenceText);
}

TEST(Audio8RealGguf, TextOnlySynthesisRoundTrip) {
  const std::string lm = envOrEmpty("QVAC_TEST_AUDIO8_LM_GGUF");
  const std::string decoder = envOrEmpty("QVAC_TEST_AUDIO8_DECODER_GGUF");
  if (lm.empty() || decoder.empty()) {
    GTEST_SKIP() << "set QVAC_TEST_AUDIO8_LM_GGUF and "
                    "QVAC_TEST_AUDIO8_DECODER_GGUF to run this";
  }

  Audio8Config cfg;
  cfg.lmModelPath = lm;
  cfg.codecDecoderPath = decoder;
  cfg.greedy = true;
  cfg.maxFrames = 16;

  Audio8Model model{cfg};
  ASSERT_NO_THROW(model.load());
  ASSERT_TRUE(model.isLoaded());

  Audio8Model::AnyInput input;
  input.text = "Hello from a fully on-device C plus plus pipeline.";
  const auto pcm =
      std::any_cast<Audio8Model::Output>(model.process(std::any(input)));
  EXPECT_FALSE(pcm.empty());
  EXPECT_EQ(
      model.sampleRate(), qvac::ttsggml::audio8::AUDIO8_NATIVE_SAMPLE_RATE);
}
