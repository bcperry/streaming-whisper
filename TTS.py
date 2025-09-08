import asyncio
import os
import logging
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
import sounddevice as sd
from collections import deque

from faster_whisper import WhisperModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tts.log')
    ]
)
logger = logging.getLogger(__name__)


# ==========================
# Config
# ==========================
SAMPLE_RATE = int(os.getenv("WHISPER_SAMPLE_RATE", "16000"))
CHANNELS = 1
BLOCK_SIZE = int(os.getenv("WHISPER_BLOCK_SIZE", "512"))  # frames per callback

# VAD parameters (very simple energy-based VAD)
VAD_ENERGY_THRESHOLD = float(os.getenv("WHISPER_VAD_RMS_THRESH", "0.001"))  
VAD_MIN_SPEECH_FRAMES = int(os.getenv("WHISPER_VAD_MIN_SPEECH_FRAMES", "3"))  # ~3 * block (reduced from 5)
VAD_MAX_SILENCE_FRAMES = int(os.getenv("WHISPER_VAD_MAX_SILENCE_FRAMES", "15"))  # end utterance (reduced from 20)
PRESPEECH_BUFFER_SEC = float(os.getenv("WHISPER_PRESPEECH_BUFFER_SEC", "0.3"))  # pad before start

# Interim transcription parameters
INTERIM_TRANSCRIBE_FRAMES = int(os.getenv("WHISPER_INTERIM_FRAMES", "30"))  # frames between interim transcriptions

# Whisper model config
MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny.en")  # e.g., tiny, base, small, medium, large-v3
DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # cpu or cuda
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8, float16, float32
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
        logger.info(f"Loading Whisper model '{MODEL_NAME}' on {DEVICE} (compute_type={COMPUTE_TYPE})...")
        self.model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
        self.loop = asyncio.get_event_loop()

        self.prespeech_frames = int(PRESPEECH_BUFFER_SEC * SAMPLE_RATE / BLOCK_SIZE)
        self.prespeech_buffer: Deque[np.ndarray] = deque(maxlen=max(self.prespeech_frames, 1))
        self.state = UtteranceState()
        self.transcription_callback = None

    def _rms(self, x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))

    def _ends_with_sentence_punctuation(self, text: str) -> bool:
        """Check if text ends with sentence-ending punctuation"""
        if not text:
            return False
        return text.rstrip()[-1:] in '.!?'

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
                logger.info(f"[final] {text}")
                return text, True
            else:
                # Only print interim results if text is different AND ends with sentence punctuation
                if text != self.state.last_interim_text and self._ends_with_sentence_punctuation(text):
                    logger.debug(f"{text}")
                    self.state.last_interim_text = text
                    return text, False
        return None, is_final

    def on_input(self, indata: np.ndarray):
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
                logger.debug(f"Speech detected, frames: {self.state.speech_frames}/{VAD_MIN_SPEECH_FRAMES}")
                if self.state.speech_frames >= VAD_MIN_SPEECH_FRAMES:
                    # Start collecting: include prespeech
                    logger.info(f"Starting new utterance collection (energy: {energy:.4f})")
                    self.state.collecting = True
                    self.state.current_chunks.extend(list(self.prespeech_buffer))
                    self.state.current_chunks.append(block)
                    self.state.silence_frames = 0
                    self.state.frames_since_last_interim = 0
                    # reset prespeech buffer for next time
                    self.prespeech_buffer.clear()
            else:
                # still idle
                if self.state.speech_frames > 0:
                    self.state.speech_frames = max(0, self.state.speech_frames - 1)
                    logger.debug(f"Silence in idle, speech frames: {self.state.speech_frames}")
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
                    asyncio.run_coroutine_threadsafe(
                        self._transcribe_and_callback(audio, is_final=False), 
                        self.loop
                    )
            else:
                self.state.silence_frames += 1
                logger.debug(f"Silence in utterance, frames: {self.state.silence_frames}/{VAD_MAX_SILENCE_FRAMES}")
                if self.state.silence_frames >= VAD_MAX_SILENCE_FRAMES:
                    # End of utterance - this is final
                    logger.info(f"Ending utterance after {self.state.silence_frames} silence frames")
                    audio = np.concatenate(self.state.current_chunks).astype(np.float32)
                    # schedule final transcription
                    asyncio.run_coroutine_threadsafe(
                        self._transcribe_and_callback(audio, is_final=True), 
                        self.loop
                    )
                    # reset state after scheduling transcription
                    self.state = UtteranceState()
                    logger.debug("Reset state after final transcription - ready for new utterance")

    async def _transcribe_and_callback(self, audio: np.ndarray, is_final: bool = False):
        """Combined transcription and callback to avoid Future handling issues"""
        try:
            result = await self._transcribe_async(audio, is_final)
            if result and result[0] and self.transcription_callback:
                await self.transcription_callback(result[0], result[1])
        except Exception as e:
            logger.error(f"Error in transcription callback: {e}")

    async def run(self):
        logger.info("Mic ready. Speak into your microphone. Press Ctrl+C to stop.")

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"[audio] {status}")
            self.on_input(indata)

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
                logger.info("Stopping...")


async def main():
    app = MicWhisper()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())