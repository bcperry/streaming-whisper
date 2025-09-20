import asyncio
import os
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
import sounddevice as sd
from collections import deque

from faster_whisper import WhisperModel
from src.config import audio_settings, vad_settings, whisper_settings, transcription_settings
from src.utils.logging import (
    get_application_logger, 
    configure_application_logging,
    error_handler,
    TranscriptionError,
    log_exception
)

# Configure structured logging
configure_application_logging()
logger = get_application_logger('transcription')


@dataclass
class UtteranceState:
    collecting: bool = False
    speech_frames: int = 0
    silence_frames: int = 0
    frames_since_last_interim: int = 0
    total_frames: int = 0  # total frames collected in current utterance
    interim_count: int = 0  # number of interim transcriptions done
    current_chunks: List[np.ndarray] = None
    last_interim_text: str = ""

    def __post_init__(self):
        if self.current_chunks is None:
            self.current_chunks = []

class MicWhisper:
    def __init__(self):
        logger.info(f"Loading Whisper model '{whisper_settings.model_name}' on {whisper_settings.device} (compute_type={whisper_settings.compute_type})...")
        self.model = WhisperModel(whisper_settings.model_name, device=whisper_settings.device, compute_type=whisper_settings.compute_type)
        self.loop = asyncio.get_event_loop()

        self.prespeech_frames = int(vad_settings.prespeech_buffer_sec * audio_settings.sample_rate / audio_settings.block_size)
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

    def _should_finalize_on_natural_break(self, current_text: str, previous_text: str) -> bool:
        """Check if we should finalize based on natural break detection"""
        if not current_text or not previous_text:
            return False
        
        # If current text ends with sentence punctuation and is significantly different from previous
        if self._ends_with_sentence_punctuation(current_text):
            # Check if this is a natural completion (not just a small addition)
            word_diff = len(current_text.split()) - len(previous_text.split())
            return word_diff >= 2  # At least 2 new words to consider it a natural break
        
        return False

    async def _transcribe_async(self, audio: np.ndarray, is_final: bool = False) -> Optional[tuple[str, bool]]:
        try:
            # Whisper expects (samples,) shape
            if audio.ndim > 1:
                audio = audio.flatten()

            # Transcribe
            segments, _ = self.model.transcribe(
                audio, 
                beam_size=1, 
                language=whisper_settings.language,
                without_timestamps=True,
                vad_filter=False,  # we already did VAD
                condition_on_previous_text=False
            )

            # Extract text
            text = " ".join(segment.text for segment in segments).strip()
            
            if not text:
                return None

            # Determine if this should be considered final
            actual_is_final = is_final
            
            # Check for natural break detection only for interim transcriptions
            if not is_final and self.state.last_interim_text:
                if self._should_finalize_on_natural_break(text, self.state.last_interim_text):
                    logger.info(f"Natural break detected: '{text}' (was: '{self.state.last_interim_text}')")
                    actual_is_final = True
            
            # Update state for interim transcriptions
            if not actual_is_final:
                self.state.last_interim_text = text
                self.state.interim_count += 1
                logger.debug(f"Interim transcription #{self.state.interim_count}: '{text}'")
            else:
                logger.info(f"Final transcription: '{text}'")

            return (text, actual_is_final)

        except Exception as e:
            log_exception(logger, e, component="transcription")
            raise TranscriptionError(f"Transcription failed: {str(e)}") from e

    def on_input(self, block: np.ndarray):
        """Called by the audio callback for each block."""
        # block is shape (512, 1) typically
        if block.ndim > 1:
            block = block[:, 0]  # take first channel

        # downsample/ensure 16kHz if needed (sounddevice already configured to 16kHz)

        # Keep a small pre-speech buffer
        self.prespeech_buffer.append(block.copy())

        # VAD
        energy = self._rms(block)
        is_speech = energy >= vad_settings.energy_threshold

        if not self.state.collecting:
            if is_speech:
                self.state.speech_frames += 1
                logger.debug(f"Speech detected, frames: {self.state.speech_frames}/{vad_settings.min_speech_frames}")
                if self.state.speech_frames >= vad_settings.min_speech_frames:
                    # Start collecting: include prespeech
                    logger.info(f"Starting new utterance collection (energy: {energy:.4f})")
                    self.state.collecting = True
                    self.state.current_chunks.extend(list(self.prespeech_buffer))
                    self.state.current_chunks.append(block)
                    self.state.silence_frames = 0
                    self.state.frames_since_last_interim = 0
                    self.state.total_frames = len(self.state.current_chunks)  # include prespeech in count
                    self.state.interim_count = 0
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
            self.state.total_frames += 1
            
            # Check for length-based finalization
            should_finalize_length = (
                self.state.total_frames >= transcription_settings.max_utterance_frames or 
                self.state.interim_count >= transcription_settings.max_interim_count
            )
            
            if is_speech:
                self.state.silence_frames = 0
                
                # Check if we should do an interim transcription
                if self.state.frames_since_last_interim >= transcription_settings.interim_frames:
                    audio = np.concatenate(self.state.current_chunks).astype(np.float32)
                    self.state.frames_since_last_interim = 0
                    
                    if should_finalize_length:
                        # Force final transcription due to length limits
                        logger.info(f"Forcing final transcription due to length limits (frames: {self.state.total_frames}, interims: {self.state.interim_count})")
                        asyncio.run_coroutine_threadsafe(
                            self._transcribe_and_callback(audio, is_final=True), 
                            self.loop
                        )
                        # reset state after scheduling transcription
                        self.state = UtteranceState()
                        logger.debug("Reset state after length-based final transcription - ready for new utterance")
                    else:
                        # schedule interim transcription
                        asyncio.run_coroutine_threadsafe(
                            self._transcribe_and_callback(audio, is_final=False), 
                            self.loop
                        )
            else:
                self.state.silence_frames += 1
                logger.debug(f"Silence in utterance, frames: {self.state.silence_frames}/{vad_settings.max_silence_frames}")
                
                # Force finalization if we hit length limits, even with some silence
                if should_finalize_length:
                    logger.info(f"Forcing final transcription due to length limits during silence (frames: {self.state.total_frames}, interims: {self.state.interim_count})")
                    audio = np.concatenate(self.state.current_chunks).astype(np.float32)
                    asyncio.run_coroutine_threadsafe(
                        self._transcribe_and_callback(audio, is_final=True), 
                        self.loop
                    )
                    # reset state after scheduling transcription
                    self.state = UtteranceState()
                    logger.debug("Reset state after length-based final transcription during silence - ready for new utterance")
                elif self.state.silence_frames >= vad_settings.max_silence_frames:
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
            if result and result[0]:
                # If natural break detection triggered a final transcription, reset state
                if result[1] and not is_final:  # result[1] is True (final) but is_final was False (natural break)
                    logger.debug("Natural break detected - resetting state for new utterance")
                    self.state = UtteranceState()
                
                if self.transcription_callback:
                    await self.transcription_callback(result[0], result[1])
        except TranscriptionError:
            # Re-raise transcription errors as they're already properly logged
            raise
        except Exception as e:
            log_exception(logger, e, component="transcription_callback")
            raise TranscriptionError(f"Transcription callback failed: {str(e)}") from e

    async def run(self):
        logger.info("Mic ready. Speak into your microphone. Press Ctrl+C to stop.")

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"[audio] {status}")
            self.on_input(indata)

        with sd.InputStream(
            samplerate=audio_settings.sample_rate,
            blocksize=audio_settings.block_size,
            channels=audio_settings.channels,
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