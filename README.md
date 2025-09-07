Local Whisper transcription (microphone)

This app captures mic audio, segments utterances with a simple energy-based VAD, and transcribes locally using faster-whisper. No external websocket or internet required after the model is downloaded the first time.

Quick start

1) Install dependencies (managed by uv):

	- uv sync

2) Run:

	- uv run python main.py

Config (env vars)

- WHISPER_MODEL: Whisper model size (default: tiny). Examples: tiny, tiny.en, base, base.en, small, medium, large-v3.
- WHISPER_DEVICE: cpu or cuda (default: cpu).
- WHISPER_COMPUTE_TYPE: int8 (cpu default) or float16 (typical for cuda).
- WHISPER_SAMPLE_RATE: Input sample rate (default: 16000).
- WHISPER_BLOCK_SIZE: Frames per audio callback (default: 512).
- WHISPER_VAD_RMS_THRESH: Simple VAD RMS threshold (default: 0.01 ~ -40 dBFS).
- WHISPER_VAD_MIN_SPEECH_FRAMES: Min consecutive speech frames to start (default: 5).
- WHISPER_VAD_MAX_SILENCE_FRAMES: Consecutive silence frames to end (default: 20).
- WHISPER_PRESPEECH_BUFFER_SEC: Prepend audio before speech detected (default: 0.3 sec).

Notes

- For NVIDIA GPUs, set WHISPER_DEVICE=cuda and WHISPER_COMPUTE_TYPE=float16 for best performance.
- The first run will download the model weights to a local cache.
- Press Ctrl+C to stop.

