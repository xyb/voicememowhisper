"""Shared dataclass contracts for speaker-id PoC scripts.

All PoC scripts communicate via these types only. When stage 3 merges
this work into the main project, this file moves there unchanged.

Design rules:
- Every dataclass serializes to JSON via `to_json(path)` and loads via
  `from_json(path)`. indent=2, ensure_ascii=False.
- No ML dependencies. stdlib only. numpy optional (for embeddings).
- Each artifact carries enough provenance (backend, model, config_hash)
  that two runs from different backends can be compared later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


# ---------- generic helpers ----------


def _dump_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _load_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------- audio input ----------


@dataclass
class AudioInput:
    """A single audio file to process.

    `recording_id` is the stable key used across runs/ and outputs/.
    Usually it's the sample filename stem, e.g. `sample-recording`.
    """

    path: str
    recording_id: str
    duration_sec: float | None = None
    sample_rate: int | None = None
    language_hint: str | None = None

    def to_json(self, path: Path) -> None:
        _dump_json(asdict(self), path)

    @classmethod
    def from_json(cls, path: Path) -> "AudioInput":
        return cls(**_load_json(path))


# ---------- transcript ----------


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


@dataclass
class Transcript:
    """Output of a `transcribe` stage.

    Carries enough provenance to compare two backends later.
    """

    recording_id: str
    backend: str  # e.g. "faster_whisper", "whisperkit", "vibevoice_asr"
    model: str  # e.g. "large-v3"
    language: str
    duration_sec: float
    segments: list[Segment] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        _dump_json(_to_dict(self), path)

    @classmethod
    def from_json(cls, path: Path) -> "Transcript":
        data = _load_json(path)
        segs = [_segment_from_dict(s) for s in data.get("segments", [])]
        return cls(
            recording_id=data["recording_id"],
            backend=data["backend"],
            model=data["model"],
            language=data["language"],
            duration_sec=data["duration_sec"],
            segments=segs,
        )


def _segment_from_dict(d: dict) -> Segment:
    words = None
    if d.get("words"):
        words = [Word(**w) for w in d["words"]]
    return Segment(
        start=d["start"],
        end=d["end"],
        text=d["text"],
        words=words,
        avg_logprob=d.get("avg_logprob"),
        no_speech_prob=d.get("no_speech_prob"),
    )


# ---------- diarization ----------


@dataclass
class SpeakerSegment:
    start: float
    end: float
    label: str  # anonymous, e.g. "SPEAKER_00"
    confidence: float | None = None


@dataclass
class Diarization:
    recording_id: str
    backend: str  # e.g. "pyannote", "sherpa_onnx"
    model: str
    num_speakers: int
    segments: list[SpeakerSegment] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        _dump_json(_to_dict(self), path)

    @classmethod
    def from_json(cls, path: Path) -> "Diarization":
        data = _load_json(path)
        segs = [SpeakerSegment(**s) for s in data.get("segments", [])]
        return cls(
            recording_id=data["recording_id"],
            backend=data["backend"],
            model=data["model"],
            num_speakers=data["num_speakers"],
            segments=segs,
        )


# ---------- canonical transcript (cross-check output) ----------


@dataclass
class Divergence:
    """A place where two transcripts disagreed on text within the same time span."""

    segment_idx: int
    char_span: tuple[int, int]  # within the chosen text
    variants: dict[str, str]  # backend name → that backend's text
    chosen: str  # which backend's version was kept
    reason: str  # "majority", "highest_confidence", "main_only", ...


@dataclass
class CanonicalSegment:
    start: float
    end: float
    text: str
    sources: list[str]  # list of backend names that contributed
    words: list[Word] | None = None


@dataclass
class CanonicalTranscript:
    """Either a passthrough of a single transcript, or the result of cross-checking."""

    recording_id: str
    source_backends: list[str]
    duration_sec: float
    language: str
    segments: list[CanonicalSegment] = field(default_factory=list)
    divergences: list[Divergence] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        _dump_json(_to_dict(self), path)

    @classmethod
    def from_json(cls, path: Path) -> "CanonicalTranscript":
        data = _load_json(path)
        segs = [_canonical_segment_from_dict(s) for s in data.get("segments", [])]
        divs = [
            Divergence(
                segment_idx=d["segment_idx"],
                char_span=tuple(d["char_span"]),
                variants=d["variants"],
                chosen=d["chosen"],
                reason=d["reason"],
            )
            for d in data.get("divergences", [])
        ]
        return cls(
            recording_id=data["recording_id"],
            source_backends=data["source_backends"],
            duration_sec=data["duration_sec"],
            language=data["language"],
            segments=segs,
            divergences=divs,
        )

    @classmethod
    def from_transcript(cls, tr: Transcript) -> "CanonicalTranscript":
        """Passthrough: use a single transcript as canonical."""
        segs = [
            CanonicalSegment(
                start=s.start,
                end=s.end,
                text=s.text,
                sources=[tr.backend],
                words=s.words,
            )
            for s in tr.segments
        ]
        return cls(
            recording_id=tr.recording_id,
            source_backends=[tr.backend],
            duration_sec=tr.duration_sec,
            language=tr.language,
            segments=segs,
            divergences=[],
        )


def _canonical_segment_from_dict(d: dict) -> CanonicalSegment:
    words = None
    if d.get("words"):
        words = [Word(**w) for w in d["words"]]
    return CanonicalSegment(
        start=d["start"],
        end=d["end"],
        text=d["text"],
        sources=d["sources"],
        words=words,
    )


# ---------- identification ----------


IdentificationSource = Literal[
    "library_match",  # matched against speaker-library centroid
    "manual",  # hand-labeled
    "unknown",  # below threshold, no name assigned
]


@dataclass
class Assignment:
    label: str  # anonymous diarization label, e.g. "SPEAKER_00"
    source: IdentificationSource
    name: str | None = None  # resolved display name, or None for unknown
    confidence: float | None = None


@dataclass
class Identification:
    recording_id: str
    backend: str  # e.g. "pyannote_embedding"
    model: str
    threshold: float
    assignments: list[Assignment] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        _dump_json(_to_dict(self), path)

    @classmethod
    def from_json(cls, path: Path) -> "Identification":
        data = _load_json(path)
        asg = [Assignment(**a) for a in data.get("assignments", [])]
        return cls(
            recording_id=data["recording_id"],
            backend=data["backend"],
            model=data["model"],
            threshold=data["threshold"],
            assignments=asg,
        )

    def name_for(self, label: str) -> tuple[str | None, float | None]:
        for a in self.assignments:
            if a.label == label:
                return a.name, a.confidence
        return None, None


# ---------- merged transcript (final structured output) ----------


@dataclass
class MergedSegment:
    start: float
    end: float
    text: str
    speaker_label: str  # anonymous label, always set
    speaker_name: str | None = None  # resolved name if any
    confidence: float | None = None
    needs_review: bool = False


@dataclass
class MergedTranscript:
    recording_id: str
    duration_sec: float
    language: str
    pipeline: str  # human-readable pipeline description
    segments: list[MergedSegment] = field(default_factory=list)
    unresolved_labels: list[str] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        _dump_json(_to_dict(self), path)

    @classmethod
    def from_json(cls, path: Path) -> "MergedTranscript":
        data = _load_json(path)
        segs = [MergedSegment(**s) for s in data.get("segments", [])]
        return cls(
            recording_id=data["recording_id"],
            duration_sec=data["duration_sec"],
            language=data["language"],
            pipeline=data["pipeline"],
            segments=segs,
            unresolved_labels=data.get("unresolved_labels", []),
        )


# ---------- stage metadata ----------


@dataclass
class StageMeta:
    """Per-run resource and config record. Written alongside every stage output."""

    stage_id: str  # e.g. "01_transcribe_faster_whisper"
    backend: str
    model: str
    started_at: str  # ISO 8601
    elapsed_sec: float
    peak_rss_bytes: int | None = None
    config: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_json(self, path: Path) -> None:
        _dump_json(asdict(self), path)

    @classmethod
    def from_json(cls, path: Path) -> "StageMeta":
        return cls(**_load_json(path))


# ---------- dict conversion (handles nested dataclasses) ----------


def _to_dict(obj: Any) -> Any:
    """Recursive dataclass → dict, dropping None values to keep JSON clean."""
    return _strip_none(asdict(obj))


def _strip_none(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _strip_none(x) for k, x in v.items() if x is not None}
    if isinstance(v, list):
        return [_strip_none(x) for x in v]
    return v
