# Configuration Documentation

## Overview

The streaming-whisper application uses a centralized configuration system based on Pydantic Settings. All configuration options can be set via environment variables or use sensible defaults.

## Environment Variables

All environment variables use the `WHISPER_` prefix followed by the section and parameter name.

### Audio Settings (`WHISPER_AUDIO_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_AUDIO_SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `WHISPER_AUDIO_CHANNELS` | `1` | Number of audio channels (mono=1) |
| `WHISPER_AUDIO_BLOCK_SIZE` | `512` | Audio processing block size in frames |

### Voice Activity Detection (`WHISPER_VAD_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_VAD_ENERGY_THRESHOLD` | `0.001` | RMS energy threshold for speech detection |
| `WHISPER_VAD_MIN_SPEECH_FRAMES` | `3` | Minimum consecutive speech frames to start recording |
| `WHISPER_VAD_MAX_SILENCE_FRAMES` | `15` | Maximum silence frames before ending utterance |
| `WHISPER_VAD_PRESPEECH_BUFFER_SEC` | `0.3` | Pre-speech buffer duration in seconds |

### Whisper Model (`WHISPER_MODEL_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_MODEL_NAME` | `tiny.en` | Whisper model name (tiny, base, small, medium, large-v3) |
| `WHISPER_MODEL_DEVICE` | `auto` | Processing device (auto, cpu, cuda) |
| `WHISPER_MODEL_COMPUTE_TYPE` | `int8` | Compute precision (int8, float16, float32) |
| `WHISPER_MODEL_LANGUAGE` | `en` | Language code for transcription |

### Transcription Processing (`WHISPER_TRANSCRIPTION_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_TRANSCRIPTION_INTERIM_FRAMES` | `30` | Frames between interim transcriptions |
| `WHISPER_TRANSCRIPTION_MAX_UTTERANCE_FRAMES` | `480` | Maximum frames per utterance (~15 seconds) |
| `WHISPER_TRANSCRIPTION_MAX_INTERIM_COUNT` | `5` | Maximum interim transcriptions before forcing final |

### Storage (`WHISPER_STORAGE_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_STORAGE_TRANSCRIPTION_DIR` | `transcriptions` | Directory for storing transcription files |

### Logging (`WHISPER_LOGGING_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_LOGGING_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `WHISPER_LOGGING_FORMAT` | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` | Log message format |
| `WHISPER_LOGGING_TRANSCRIPTION_LOG_FILE` | `transcription.log` | Transcription module log file |
| `WHISPER_LOGGING_API_LOG_FILE` | `api.log` | API module log file |

### Server (`WHISPER_SERVER_*`)

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_SERVER_HOST` | `127.0.0.1` | Server host address |
| `WHISPER_SERVER_PORT` | `8000` | Server port number |

## Usage Examples

### Development with Debug Logging
```bash
export WHISPER_LOGGING_LEVEL=DEBUG
export WHISPER_MODEL_MODEL_NAME=base.en
uv run uvicorn api:app --reload
```

### Production Configuration
```bash
export WHISPER_LOGGING_LEVEL=WARNING
export WHISPER_MODEL_MODEL_NAME=small.en
export WHISPER_MODEL_DEVICE=cuda
export WHISPER_MODEL_COMPUTE_TYPE=float16
export WHISPER_STORAGE_TRANSCRIPTION_DIR=/var/lib/whisper/transcriptions
export WHISPER_SERVER_HOST=0.0.0.0
export WHISPER_SERVER_PORT=80
uv run uvicorn api:app --host 0.0.0.0 --port 80
```

### High-Quality Transcription
```bash
export WHISPER_MODEL_MODEL_NAME=large-v3
export WHISPER_MODEL_COMPUTE_TYPE=float32
export WHISPER_VAD_ENERGY_THRESHOLD=0.0005
export WHISPER_TRANSCRIPTION_INTERIM_FRAMES=20
```

### Low-Latency Configuration
```bash
export WHISPER_MODEL_MODEL_NAME=tiny.en
export WHISPER_AUDIO_BLOCK_SIZE=256
export WHISPER_TRANSCRIPTION_INTERIM_FRAMES=15
export WHISPER_VAD_MIN_SPEECH_FRAMES=2
```

## Programmatic Access

```python
from src.config import settings

# Access individual settings
print(f"Model: {settings.whisper.model_name}")
print(f"Sample rate: {settings.audio.sample_rate}")

# Access all settings in a section
audio_config = settings.audio
vad_config = settings.vad

# Create a custom settings instance with overrides
from src.config import AppSettings
custom_settings = AppSettings()
```

## Migration from Old Configuration

The old hardcoded constants and environment variables have been replaced:

| Old | New |
|-----|-----|
| `SAMPLE_RATE` | `settings.audio.sample_rate` |
| `VAD_ENERGY_THRESHOLD` | `settings.vad.energy_threshold` |
| `MODEL_NAME` | `settings.whisper.model_name` |
| `TRANSCRIPTION_STORAGE_DIR` | `settings.storage.transcription_dir` |

Environment variables have changed from direct names to prefixed structure:
- `WHISPER_SAMPLE_RATE` → `WHISPER_AUDIO_SAMPLE_RATE`
- `WHISPER_MODEL` → `WHISPER_MODEL_MODEL_NAME`
- `WHISPER_VAD_RMS_THRESH` → `WHISPER_VAD_ENERGY_THRESHOLD`