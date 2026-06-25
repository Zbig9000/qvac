import {
  loadModel,
  textToSpeech,
  unloadModel,
  type ModelProgressUpdate,
  TTS_T3_TURBO_EN_CHATTERBOX_Q8_0,
  TTS_S3GEN_EN_CHATTERBOX,
} from "@qvac/sdk";
import {
  createWav,
  playAudio,
  int16ArrayToBuffer,
  createWavHeader,
} from "./utils";

// Chatterbox TTS (GGML) + LavaSR neural enhancement (QVAC-16579).
//
// Adds the `enhancer` block to the standard Chatterbox config: after
// synthesis the 24 kHz output is neurally bandwidth-extended to 48 kHz
// (CPU/GGML). Convert the public LavaSR enhancer to GGUF first:
//
//   python qvac-ext-lib-whisper.cpp/tts-cpp/scripts/convert-lavasr-enhancer-to-gguf.py \
//     --backbone enhancer_backbone.onnx --spec-head enhancer_spec_head.onnx \
//     --out lavasr-enhancer.gguf --ftype f16
//
// then point `enhancerSrc` at it (local path, or a registry/s3 source once a
// GGUF registry entry lands). Output is 48 kHz when enhancement is active.
// Usage: node chatterbox-enhanced.ts [referenceAudioSrc]
const [referenceAudioSrc] = process.argv.slice(2);

const ENHANCED_SAMPLE_RATE = 48000;
const LAVASR_ENHANCER_GGUF =
  process.env.LAVASR_ENHANCER_GGUF ?? "./models/lavasr/lavasr-enhancer.gguf";

try {
  const modelId = await loadModel({
    modelSrc: TTS_T3_TURBO_EN_CHATTERBOX_Q8_0,
    modelConfig: {
      ttsEngine: "chatterbox",
      language: "en",
      s3genModelSrc: TTS_S3GEN_EN_CHATTERBOX.src,
      cfmSteps: 1,
      threads: 8,
      ...(referenceAudioSrc ? { referenceAudioSrc } : {}),
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

  // Note: enhancement runs on the batch (non-streaming) path, so request a
  // non-streaming synthesis here.
  const result = textToSpeech({
    modelId,
    text: `QVAC SDK is the canonical entry point to QVAC. With LavaSR enabled, Chatterbox output is neurally upsampled to forty-eight kilohertz.`,
    inputType: "text",
    stream: false,
  });

  const audioBuffer = await result.buffer;
  console.log(`▸ TTS complete. Total samples: ${audioBuffer.length}`);

  createWav(audioBuffer, ENHANCED_SAMPLE_RATE, "chatterbox-enhanced-output.wav");
  console.log("▸ Audio saved to chatterbox-enhanced-output.wav");

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
