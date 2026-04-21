"""Pluggable ASR backends for the speaker-id pipeline.

Each backend produces a ``contracts.Transcript`` so downstream stages
(diarize / identify / merge / render) don't need to know where the
transcript came from.

Current backends:

- ``faster_whisper``: in-process Python, CPU/MPS, default. Implementation
  stays in ``voicememowhisper.si.transcribe`` for now to avoid churn;
  will migrate here in a later commit.
- ``openai_audio`` (this package): HTTP client for any service that
  speaks the OpenAI Audio API — OpenAI Whisper, FunASR slim, groq,
  deepinfra, or any compatible proxy. The protocol name is ``openai-audio``.

Adding a new protocol (deepgram / azure-stt / aliyun-nls / ...) means
adding a sibling module here with its own request/response adapter.
The public surface is just ``transcribe(audio_path, config) -> Transcript``.
"""

# Submodules are imported on demand. Avoid top-level re-exports here so
# ``python -m voicememowhisper.si.asr_backends.openai_audio`` doesn't
# trigger the "found in sys.modules after import of package" runtime
# warning. Import explicitly:
#   from voicememowhisper.si.asr_backends import openai_audio
