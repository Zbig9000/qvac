import {
  AudioGen,
  ENGINE_MINIMAX,
  ERR_CODES,
  QvacErrorAudioGen,
  type AudioGenEngine,
  type AudiogenOutputChunk
} from '../../index'

const audioGen = new AudioGen()
const errorCode: number = ERR_CODES.INVALID_INPUT
const errorConstructor: typeof QvacErrorAudioGen = QvacErrorAudioGen
const engine: AudioGenEngine = ENGINE_MINIMAX
const minimax = new AudioGen({
  engine,
  files: {
    lmModel: '/models/mm3-lm.gguf',
    synthModel: '/models/mm3-synth.gguf'
  }
})
const output: AudiogenOutputChunk = {
  outputArray: new Int16Array(0),
  sampleRate: 48000,
  channels: 2
}

void audioGen
void errorCode
void errorConstructor
void minimax
void output
