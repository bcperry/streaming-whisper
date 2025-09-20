"""Core transcription functionality for streaming-whisper."""

import asyncio
import os
from dataclasses import dataclass
from typing import Deque, List, Optional, Callable

import numpy as np
import sounddevice as sd
from collections import deque

from faster_whisper import WhisperModel
from src.config import audio_settings, vad_settings, whisper_settings, transcription_settings
from src.utils.logging import (
    get_application_logger, 
    TranscriptionError,
    log_exception
)

logger = get_application_logger('transcription')


@dataclass
class UtteranceState:
    """State tracking for utterance detection and processing."""
    collecting: bool = False
    speech_frames: int = 0
    silence_frames: int = 0
    frames_since_last_interim: int = 0
    interim_count: int = 0


class WhisperTranscriber:
    """Core Whisper transcription engine with voice activity detection."""
    
    def __init__(self):
        self.model = None
        self.state = UtteranceState()
        self.frame_count = 0
        self.last_interim_text = ""
        self.audio_buffer: Deque[np.ndarray] = deque(maxlen=vad_settings.buffer_max_length)
        self.transcription_callback: Optional[Callable[[str, bool], None]] = None
        
        # Load the Whisper model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the Whisper model with configured settings."""
        try:
            logger.info(f"Loading Whisper model '{whisper_settings.model}' on {whisper_settings.device} (compute_type={whisper_settings.compute_type})...")
            self.model = WhisperModel(
                whisper_settings.model, 
                device=whisper_settings.device, 
                compute_type=whisper_settings.compute_type
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            log_exception(logger, e, component="model_loading")
            raise TranscriptionError(f"Failed to load Whisper model: {str(e)}") from e
    
    def _detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """Detect voice activity in audio chunk using energy-based VAD."""
        # Simple energy-based VAD
        energy = np.sum(audio_chunk ** 2) / len(audio_chunk)
        return energy > vad_settings.energy_threshold
    
    def _should_process_utterance(self) -> bool:
        """Determine if current utterance should be processed for transcription."""
        return (self.state.collecting and 
                self.state.speech_frames >= vad_settings.min_speech_frames)
    
    def _should_send_interim(self) -> bool:
        """Determine if an interim transcription should be sent."""
        return (self.state.frames_since_last_interim >= transcription_settings.interim_frame_interval and
                self._should_process_utterance())
    
    def _should_finalize_utterance(self) -> bool:
        """Determine if current utterance should be finalized."""
        return (self.state.collecting and 
                self.state.silence_frames >= vad_settings.silence_frames_threshold)
    
    def _reset_state_for_new_utterance(self) -> None:
        """Reset state when starting a new utterance."""
        self.state = UtteranceState()
        self.last_interim_text = ""
        # Clear audio buffer to start fresh utterance
        self.audio_buffer.clear()
    
    def _get_buffered_audio(self) -> np.ndarray:
        """Get concatenated audio from buffer."""
        if not self.audio_buffer:
            return np.array([])
        
        # Concatenate all audio chunks in buffer
        audio_data = np.concatenate(list(self.audio_buffer))
        return audio_data.flatten()
    
    def transcribe(self, is_final: bool = False) -> Optional[tuple[str, bool]]:
        """
        Transcribe current audio buffer.
        
        Args:
            is_final: Whether this is a final transcription (vs interim)
            
        Returns:
            Tuple of (text, is_final) or None if transcription fails
        """
        try:
            audio_data = self._get_buffered_audio()
            
            if len(audio_data) == 0:
                logger.debug("No audio data to transcribe")
                return None
            
            # Ensure audio is in correct format for Whisper
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Transcribe with Whisper
            segments, info = self.model.transcribe(
                audio_data, 
                beam_size=whisper_settings.beam_size,
                language=whisper_settings.language if whisper_settings.language != "auto" else None
            )
            
            # Extract text from segments
            text = ""
            for segment in segments:
                text += segment.text
            
            text = text.strip()
            
            if not text:
                logger.debug("Empty transcription result")
                return None
            
            # Determine if this is actually final
            actual_is_final = is_final
            
            # Natural break detection for long utterances
            if not is_final and len(text) > transcription_settings.natural_break_threshold:
                if text.endswith(('.', '!', '?', ':')):
                    logger.debug("Natural break detected in transcription")
                    actual_is_final = True
            
            # Track interim transcriptions
            if not actual_is_final:
                self.state.interim_count += 1
                self.last_interim_text = text
                logger.debug(f"Interim transcription #{self.state.interim_count}: '{text}'")
            else:
                logger.info(f"Final transcription: '{text}'")
            
            return (text, actual_is_final)
            
        except Exception as e:
            log_exception(logger, e, component="transcription")
            raise TranscriptionError(f"Transcription failed: {str(e)}") from e
    
    async def _handle_transcription_result(self, result: tuple[str, bool], is_final: bool) -> None:
        """Handle transcription result and call callback if available."""
        try:
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
    
    def on_input(self, block: np.ndarray) -> None:
        """
        Process incoming audio block.
        
        Args:
            block: Audio data block from microphone input
        """
        # Ensure single channel
        if block.ndim > 1:
            block = block[:, 0]  # take first channel
        
        self.frame_count += 1
        
        # Add to buffer
        self.audio_buffer.append(block)
        
        # Voice activity detection
        has_voice = self._detect_voice_activity(block)
        
        if has_voice:
            # Voice detected
            if not self.state.collecting:
                logger.debug("Speech started - beginning utterance collection")
                self.state.collecting = True
                self.state.silence_frames = 0
            
            self.state.speech_frames += 1
            self.state.frames_since_last_interim += 1
            
            # Check for interim transcription
            if self._should_send_interim():
                logger.debug("Sending interim transcription")
                result = self.transcribe(is_final=False)
                if result:
                    asyncio.create_task(self._handle_transcription_result(result, is_final=False))
                self.state.frames_since_last_interim = 0
        
        else:
            # No voice detected
            if self.state.collecting:
                self.state.silence_frames += 1
                self.state.frames_since_last_interim += 1
                
                # Check if we should finalize the utterance
                if self._should_finalize_utterance():
                    logger.debug("Silence threshold reached - finalizing utterance")
                    result = self.transcribe(is_final=True)
                    if result:
                        asyncio.create_task(self._handle_transcription_result(result, is_final=True))
                    self._reset_state_for_new_utterance()


class MicWhisper:
    """
    Microphone-based real-time transcription using Whisper.
    
    This class manages the audio input stream and coordinates with the transcription engine.
    """
    
    def __init__(self):
        self.transcriber = WhisperTranscriber()
        self.transcription_callback: Optional[Callable[[str, bool], None]] = None
    
    @property 
    def transcription_callback(self) -> Optional[Callable[[str, bool], None]]:
        """Get the transcription callback function."""
        return self.transcriber.transcription_callback
    
    @transcription_callback.setter
    def transcription_callback(self, callback: Optional[Callable[[str, bool], None]]) -> None:
        """Set the transcription callback function."""
        self.transcriber.transcription_callback = callback
    
    def on_input(self, block: np.ndarray) -> None:
        """Process incoming audio block."""
        self.transcriber.on_input(block)
    
    async def run(self) -> None:
        """
        Run the microphone transcription loop.
        
        This method starts the audio input stream and processes audio in real-time.
        """
        logger.info("Mic ready. Speak into your microphone. Press Ctrl+C to stop.")
        
        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"[audio] {status}")
            self.on_input(indata)
        
        with sd.InputStream(
            samplerate=audio_settings.sample_rate,
            blocksize=audio_settings.block_size,
            channels=audio_settings.channels,
            dtype=audio_settings.dtype,
            callback=callback
        ):
            try:
                while True:
                    await asyncio.sleep(0.1)  # Keep the event loop alive
            except asyncio.CancelledError:
                logger.info("Transcription cancelled")
                raise
            except KeyboardInterrupt:
                logger.info("Stopping transcription...")
                raise