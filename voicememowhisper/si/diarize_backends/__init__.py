"""Pluggable diarization backends for the speaker-id pipeline.

Each backend produces a ``contracts.Diarization`` plus an embeddings .npz
file, matching the interface of ``voicememowhisper.si.diarize.run_diarization``
so the pipeline can switch backends without downstream changes.

Current backends:

- ``local_pyannote``: in-process pyannote.audio on CPU/MPS, the existing
  default. Lives in ``voicememowhisper.si.diarize`` for backwards
  compatibility.
- ``http`` (this package): HTTP client for the self-hosted pyannote
  service (FastAPI over Traefik on the GPU host). Matches what
  ``diarize-service/server.py`` in the ops tree exposes.
"""
