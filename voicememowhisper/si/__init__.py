"""Speaker-ID pipeline as an in-package subpackage.

Stages (in order):
    1. transcribe → Whisper turns audio into a JSON transcript with timestamps.
    2. diarize    → pyannote slices audio into per-speaker segments + embeddings.
    3. identify   → match each anonymous speaker against speaker-library centroids.
    4. merge      → align transcript segments with speaker labels and names.
    5. render     → emit transcript.md (with speaker turns + ⚠️) and transcript.txt.

Each stage has its own module. `pipeline.run()` orchestrates all five with
stage caching and an optional `from_step` / `to_step` window. The CLI is in
`voicememowhisper.si.cli` and is mounted under the main `voicememo-whisper si`
command group.

ML dependencies (faster-whisper, pyannote.audio, torch, soundfile, numpy)
ship as the `[speaker-id]` optional extra. Without them, the main
voicememo-whisper transcription path falls back to WhisperKit.
"""

# Avoid eagerly importing pipeline — it pulls numpy/torch/pyannote which
# are only available with the [speaker-id] extra. Callers that need the
# pipeline import it explicitly: `from voicememowhisper.si.pipeline import run`.
