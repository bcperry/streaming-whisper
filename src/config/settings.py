"""
Centralized configuration management for streaming-whisper application.

This module provides Pydantic Settings classes for managing all configuration
parameters with environment variable support and sensible defaults.
"""

from typing import Literal
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AudioSettings(BaseSettings):
    """Audio processing configuration"""
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels (mono=1)")
    block_size: int = Field(default=512, description="Audio processing block size in frames")
    dtype: str = Field(default="float32", description="Audio data type for sounddevice")
    
    class Config:
        env_prefix = "WHISPER_AUDIO_"


class VADSettings(BaseSettings):
    """Voice Activity Detection configuration"""
    energy_threshold: float = Field(default=0.001, description="RMS energy threshold for speech detection")
    min_speech_frames: int = Field(default=3, description="Minimum consecutive speech frames to start recording")
    max_silence_frames: int = Field(default=15, description="Maximum silence frames before ending utterance")
    silence_frames_threshold: int = Field(default=15, description="Silence frames threshold for utterance end")
    prespeech_buffer_sec: float = Field(default=0.3, description="Pre-speech buffer duration in seconds")
    buffer_max_length: int = Field(default=480, description="Maximum audio buffer length in frames")
    
    class Config:
        env_prefix = "WHISPER_VAD_"


class WhisperSettings(BaseSettings):
    """Whisper model configuration"""
    model: str = Field(default="tiny.en", description="Whisper model name (tiny, base, small, medium, large-v3)")
    device: str = Field(default="auto", description="Processing device (auto, cpu, cuda)")
    compute_type: str = Field(default="int8", description="Compute precision (int8, float16, float32)")
    language: str = Field(default="en", description="Language code for transcription")
    beam_size: int = Field(default=5, description="Beam size for transcription search")
    
    class Config:
        env_prefix = "WHISPER_MODEL_"


class TranscriptionSettings(BaseSettings):
    """Transcription processing configuration"""
    interim_frame_interval: int = Field(default=15, description="Interval between interim transcriptions")
    max_utterance_frames: int = Field(default=480, description="Maximum frames per utterance (~15 seconds)")
    max_interim_count: int = Field(default=5, description="Maximum interim transcriptions before forcing final")
    natural_break_threshold: int = Field(default=100, description="Text length threshold for natural break detection")
    
    class Config:
        env_prefix = "WHISPER_TRANSCRIPTION_"


class StorageSettings(BaseSettings):
    """File storage configuration"""
    transcription_dir: str = Field(default="transcriptions", description="Directory for storing transcription files")
    
    class Config:
        env_prefix = "WHISPER_STORAGE_"


class LoggingSettings(BaseSettings):
    """Logging configuration"""
    level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    transcription_log_file: str = Field(default="transcription.log", description="Transcription module log file")
    api_log_file: str = Field(default="api.log", description="API module log file")
    
    class Config:
        env_prefix = "WHISPER_LOGGING_"


class ServerSettings(BaseSettings):
    """Server configuration"""
    host: str = Field(default="127.0.0.1", description="Server host address")
    port: int = Field(default=8000, description="Server port number")
    
    class Config:
        env_prefix = "WHISPER_SERVER_"


class AppSettings(BaseModel):
    """
    Main application settings combining all configuration sections.
    
    Environment variables can be used with these patterns:
    - WHISPER_AUDIO_SAMPLE_RATE=16000
    - WHISPER_VAD_ENERGY_THRESHOLD=0.001  
    - WHISPER_MODEL_MODEL_NAME=base.en
    - WHISPER_STORAGE_TRANSCRIPTION_DIR=my_transcriptions
    - etc.
    """
    
    # Configuration sections - each loads its own environment variables
    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    whisper: WhisperSettings = Field(default_factory=WhisperSettings)
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)


# Global settings instance
settings = AppSettings()


def get_settings() -> AppSettings:
    """Get the application settings instance"""
    return settings


# Backward compatibility - expose individual settings for easy migration
audio_settings = settings.audio
vad_settings = settings.vad
whisper_settings = settings.whisper
transcription_settings = settings.transcription
storage_settings = settings.storage
logging_settings = settings.logging
server_settings = settings.server
server_settings = settings.server