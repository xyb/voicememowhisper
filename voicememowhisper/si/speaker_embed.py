"""Speaker embedding helper: shared across 04_library.py and 05_identify.py.

Loads pyannote speaker-diarization-community-1 pipeline and reuses its
internal `_embedding` component (PyannoteAudioPretrainedSpeakerEmbedding)
to compute fixed-size speaker embeddings for arbitrary audio clips.

This keeps 04/05 identification bit-compatible with the centroids that
02_diarize_pyannote already dumped.

Usage
-----
    from speaker_embed import CommunityOneEmbedder
    emb = CommunityOneEmbedder()
    vec = emb.embed_audio(Path("clip.wav"))        # whole file
    vec = emb.embed_audio(audio_path, start=3.0, end=8.0)  # slice
    print(vec.shape)  # (dim,)  — 1D numpy array

Cost: loading community-1 takes ~30s and a few GB RAM. Call once per
process; instances are not designed to be recreated.
"""

from __future__ import annotations

import subprocess
import tempfile
import wave
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="torchcodec is not installed")


def _ffmpeg_decode_to_pcm(
    audio_path: Path,
    target_sr: int,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    """Decode any audio → mono float32 numpy array at target_sr, [-1, 1].

    Uses ffmpeg subprocess so we avoid torchaudio/torchcodec entirely.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
        ]
        if start is not None:
            cmd += ["-ss", f"{start:.3f}"]
        if end is not None and start is not None:
            cmd += ["-t", f"{max(0.0, end - start):.3f}"]
        elif end is not None:
            cmd += ["-to", f"{end:.3f}"]
        cmd += [
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-f",
            "wav",
            str(tmp_path),
        ]
        subprocess.run(cmd, check=True)

        with wave.open(str(tmp_path), "rb") as w:
            nframes = w.getnframes()
            sampwidth = w.getsampwidth()
            raw = w.readframes(nframes)
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sampwidth]
        arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        arr /= float(1 << (8 * sampwidth - 1))
        return arr
    finally:
        tmp_path.unlink(missing_ok=True)


class CommunityOneEmbedder:
    """Speaker embedder backed by pyannote community-1's internal model.

    Loading is lazy: the pipeline is constructed on first use. Subsequent
    calls reuse the same in-memory embedder.
    """

    def __init__(self, hf_token: str | None = None) -> None:
        self._hf_token = hf_token
        self._loaded = False
        self._embedder = None  # PyannoteAudioPretrainedSpeakerEmbedding
        self._sample_rate: int | None = None
        self._dimension: int | None = None
        self._min_num_samples: int | None = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from .._lock import acquire_compute_lock

        acquire_compute_lock(what="the pyannote speaker embedder")
        # Lazy import so module import is cheap.
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=self._hf_token,
        )
        self._embedder = pipeline._embedding
        self._sample_rate = int(self._embedder.sample_rate)
        self._dimension = int(self._embedder.dimension)
        self._min_num_samples = int(self._embedder.min_num_samples)
        self._loaded = True

    @property
    def sample_rate(self) -> int:
        self._ensure_loaded()
        return self._sample_rate  # type: ignore[return-value]

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._dimension  # type: ignore[return-value]

    @property
    def min_duration_sec(self) -> float:
        self._ensure_loaded()
        return self._min_num_samples / self._sample_rate  # type: ignore[operator]

    def embed_audio(
        self,
        audio_path: Path,
        start: float | None = None,
        end: float | None = None,
    ) -> np.ndarray:
        """Embed a single audio clip (file or slice) to a 1-D vector."""
        import torch

        self._ensure_loaded()
        arr = _ffmpeg_decode_to_pcm(
            audio_path, target_sr=self._sample_rate, start=start, end=end  # type: ignore[arg-type]
        )
        if arr.size < self._min_num_samples:  # type: ignore[operator]
            raise ValueError(
                f"clip too short: got {arr.size} samples "
                f"({arr.size / self._sample_rate:.2f}s), "  # type: ignore[operator]
                f"need at least {self._min_num_samples} "
                f"({self.min_duration_sec:.2f}s)"
            )
        waveform = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        with torch.inference_mode():
            embeddings = self._embedder(waveform)  # type: ignore[misc]
        # embeddings shape: (1, dim) numpy
        vec = np.asarray(embeddings)[0]
        return vec

    def embed_batch(
        self,
        audio_path: Path,
        segments: list[tuple[float, float]],
    ) -> np.ndarray:
        """Embed multiple segments from one audio file.

        Returns shape (N, dim). Segments shorter than min duration are
        padded with zeros and a warning is printed; caller should filter
        those out if needed.
        """
        import torch

        self._ensure_loaded()
        min_samples = self._min_num_samples  # type: ignore[assignment]
        sr = self._sample_rate  # type: ignore[assignment]

        clips: list[np.ndarray] = []
        for start, end in segments:
            arr = _ffmpeg_decode_to_pcm(audio_path, target_sr=sr, start=start, end=end)
            clips.append(arr)

        max_len = max(max(len(c) for c in clips), min_samples)
        batch = np.zeros((len(clips), 1, max_len), dtype=np.float32)
        for i, c in enumerate(clips):
            batch[i, 0, : len(c)] = c

        waveform = torch.from_numpy(batch)
        with torch.inference_mode():
            embeddings = self._embedder(waveform)  # type: ignore[misc]
        return np.asarray(embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D embeddings, in [-1, 1]."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
