import asyncio
import os
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
import sounddevice as sd
from collections import deque

from faster_whisper import WhisperModel


# ==========================
# Config
# ==========================
SAMPLE_RATE = int(os.getenv("WHISPER_SAMPLE_RATE", "16000"))
CHANNELS = 1
BLOCK_SIZE = int(os.getenv("WHISPER_BLOCK_SIZE", "512"))  # frames per callback

# VAD parameters (very simple energy-based VAD)
VAD_ENERGY_THRESHOLD = float(os.getenv("WHISPER_VAD_RMS_THRESH", "0.01"))  # ~-40 dBFS
VAD_MIN_SPEECH_FRAMES = int(os.getenv("WHISPER_VAD_MIN_SPEECH_FRAMES", "5"))  # ~5 * block
VAD_MAX_SILENCE_FRAMES = int(os.getenv("WHISPER_VAD_MAX_SILENCE_FRAMES", "20"))  # end utterance
PRESPEECH_BUFFER_SEC = float(os.getenv("WHISPER_PRESPEECH_BUFFER_SEC", "0.3"))  # pad before start

# Interim transcription parameters
INTERIM_TRANSCRIBE_FRAMES = int(os.getenv("WHISPER_INTERIM_FRAMES", "30"))  # frames between interim transcriptions

# Whisper model config
MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny.en")  # e.g., tiny, base, small, medium, large-v3
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # cpu or cuda
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8" if DEVICE == "cpu" else "float16")
LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")


@dataclass
class UtteranceState:
    collecting: bool = False
    speech_frames: int = 0
    silence_frames: int = 0
    frames_since_last_interim: int = 0
    current_chunks: List[np.ndarray] = None
    last_interim_text: str = ""

    def __post_init__(self):
        if self.current_chunks is None:
            self.current_chunks = []

class MicWhisper:
    def __init__(self):
        print(f"Loading Whisper model '{MODEL_NAME}' on {DEVICE} (compute_type={COMPUTE_TYPE})...")
        self.model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
        self.loop = asyncio.get_event_loop()

        self.prespeech_frames = int(PRESPEECH_BUFFER_SEC * SAMPLE_RATE / BLOCK_SIZE)
        self.prespeech_buffer: Deque[np.ndarray] = deque(maxlen=max(self.prespeech_frames, 1))
        self.state = UtteranceState()

    def _rms(self, x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))

    async def _transcribe_async(self, audio: np.ndarray, is_final: bool = False):
        # Run transcription off the audio thread
        segments, _info = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.model.transcribe(
                audio,
                beam_size=1,
                language=LANGUAGE,
                vad_filter=False,  # we already segmented
                without_timestamps=False,
            ),
        )

        text_parts = []
        for seg in segments:
            # seg.text already stripped
            if seg.text:
                text_parts.append(seg.text)
        text = " ".join(text_parts).strip()
        if text:
            if is_final:
                print(f"[final] {text}")
            else:
                # Only print if different from last interim to avoid spam
                if text != self.state.last_interim_text:
                    print(f"{text}")
                    self.state.last_interim_text = text

    def _on_block(self, indata: np.ndarray):
        # indata is float32 [-1,1], shape (frames, channels)
        if indata.ndim == 2 and indata.shape[1] > 1:
            # Mixdown to mono
            block = np.mean(indata, axis=1, dtype=np.float32)
        else:
            block = indata.reshape(-1).astype(np.float32)

        # downsample/ensure 16kHz if needed (sounddevice already configured to 16kHz)

        # Keep a small pre-speech buffer
        self.prespeech_buffer.append(block.copy())

        # VAD
        energy = self._rms(block)
        is_speech = energy >= VAD_ENERGY_THRESHOLD

        if not self.state.collecting:
            if is_speech:
                self.state.speech_frames += 1
                if self.state.speech_frames >= VAD_MIN_SPEECH_FRAMES:
                    # Start collecting: include prespeech
                    self.state.collecting = True
                    self.state.current_chunks.extend(list(self.prespeech_buffer))
                    self.state.current_chunks.append(block)
                    self.state.silence_frames = 0
                    # reset prespeech buffer for next time
                    self.prespeech_buffer.clear()
            else:
                # still idle
                self.state.speech_frames = max(0, self.state.speech_frames - 1)
        else:
            # collecting
            self.state.current_chunks.append(block)
            self.state.frames_since_last_interim += 1
            
            if is_speech:
                self.state.silence_frames = 0
                
                # Check if we should do an interim transcription
                if self.state.frames_since_last_interim >= INTERIM_TRANSCRIBE_FRAMES:
                    audio = np.concatenate(self.state.current_chunks).astype(np.float32)
                    self.state.frames_since_last_interim = 0
                    # schedule interim transcription
                    asyncio.run_coroutine_threadsafe(self._transcribe_async(audio, is_final=False), self.loop)
            else:
                self.state.silence_frames += 1
                if self.state.silence_frames >= VAD_MAX_SILENCE_FRAMES:
                    # End of utterance - this is final
                    audio = np.concatenate(self.state.current_chunks).astype(np.float32)
                    # reset state before async
                    self.state = UtteranceState()
                    # schedule final transcription
                    asyncio.run_coroutine_threadsafe(self._transcribe_async(audio, is_final=True), self.loop)

    async def run(self):
        print("Mic ready. Speak into your microphone. Press Ctrl+C to stop.")

        def callback(indata, frames, time_info, status):
            if status:
                print(f"[audio] {status}")
            self._on_block(indata)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype="float32",
            callback=callback,
        ):
            try:
                # Keep the loop alive
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass
            except KeyboardInterrupt:
                print("Stopping...")


async def main():
    app = MicWhisper()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())