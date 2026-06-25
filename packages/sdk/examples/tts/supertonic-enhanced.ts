import {
  loadModel,
  textToSpeech,
  unloadModel,
  type ModelProgressUpdate,
  TTS_MULTILINGUAL_SUPERTONIC3_Q8_0,
} from "@qvac/sdk";
import {
  createWav,
  playAudio,
  int16ArrayToBuffer,
  createWavHeader,
} from "./utils";

// Supertonic 3 TTS (GGML) + LavaSR neural enhancement (QVAC-16579).
//
// The `enhancer` block opts into the LavaSR Vocos bandwidth-extension network
// (CPU/GGML), which upsamples the engine's 44.1 kHz output to a 48 kHz signal
// with a synthesised high band. Convert the public LavaSR enhancer to GGUF
// first:
//
//   python qvac-ext-lib-whisper.cpp/tts-cpp/scripts/convert-lavasr-enhancer-to-gguf.py \
//     --backbone enhancer_backbone.onnx --spec-head enhancer_spec_head.onnx \
//     --out lavasr-enhancer.gguf
//
// then point `enhancerSrc` at it (a local path or a registry/s3 source once a
// GGUF registry entry lands). Output is 48 kHz when enhancement is active.
const ENHANCED_SAMPLE_RATE = 48000;
const LAVASR_ENHANCER_GGUF =
  process.env.LAVASR_ENHANCER_GGUF ?? "./models/lavasr/lavasr-enhancer.gguf";

try {
  const modelId = await loadModel({
    modelSrc: TTS_MULTILINGUAL_SUPERTONIC3_Q8_0,
    modelConfig: {
      ttsEngine: "supertonic",
      language: "en",
      voice: "F1",
      ttsSpeed: 1.05,
      ttsNumInferenceSteps: 5,
      enhancer: {
        type: "lavasr",
        enhance: true,
        enhancerSrc: LAVASR_ENHANCER_GGUF,
      },
    },
    onProgress: (p: ModelProgressUpdate) => {
      const mb = (n: number) => (n / 1e6).toFixed(1);
      const line = `▸ Downloading ${p.percentage.toFixed(0)}% (${mb(p.downloaded)}/${mb(p.total)} MB)`;
      process.stderr.write(process.stderr.isTTY ? `\r${line}` : `${line}\n`);
      if (p.percentage >= 100) process.stderr.write("\n");
    },
  });

  console.log(`▸ Model loaded (LavaSR enhancement ON): ${modelId}`);

  const result = textToSpeech({
    modelId,
    text: `QVAC SDK is the canonical entry point to QVAC. With LavaSR enabled, the synthesized speech is neurally upsampled to forty-eight kilohertz.`,
    inputType: "text",
    stream: false,
  });

  const audioBuffer = await result.buffer;
  console.log(`▸ TTS complete. Total samples: ${audioBuffer.length}`);

  createWav(audioBuffer, ENHANCED_SAMPLE_RATE, "supertonic-enhanced-output.wav");
  console.log("▸ Audio saved to supertonic-enhanced-output.wav");

  const audioData = int16ArrayToBuffer(audioBuffer);
  const wavBuffer = Buffer.concat([
    createWavHeader(audioData.length, ENHANCED_SAMPLE_RATE),
    audioData,
  ]);
  playAudio(wavBuffer);

  await unloadModel({ modelId });
  console.log("▸ Model unloaded");
  process.exit(0);
} catch (error) {
  console.error("✖", error);
  process.exit(1);
}
