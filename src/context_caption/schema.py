"""Serializable project state; original machine output is retained alongside edits."""
from dataclasses import asdict, dataclass, field
import math

SCHEMA_VERSION = 2
PALETTE = ["#0F766E", "#7C3AED", "#B45309", "#BE185D", "#1D4ED8", "#475569"]


def validate_interval(start, end):
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError("Timestamps must be finite with 0 <= start < end.")


@dataclass
class Word:
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str = "UNKNOWN"
    score: float | None = None

    def __post_init__(self):
        if (self.start is None) != (self.end is None):
            raise ValueError("Word start/end must both be provided or both be absent.")
        if self.start is not None:
            validate_interval(self.start, self.end)


@dataclass
class Caption:
    id: int
    start: float
    end: float
    speaker: str
    text: str
    original_text: str
    words: list[Word] = field(default_factory=list)
    alignment_valid: bool = True
    emotion: str | None = None
    emotion_score: float | None = None

    def __post_init__(self):
        validate_interval(self.start, self.end)
        if not self.text.strip():
            raise ValueError("Caption text cannot be empty.")


@dataclass
class Sound:
    """A non-speech sound cue. Text is stored bare; formatting adds the SDH brackets."""
    id: int
    start: float
    end: float
    text: str
    original_text: str
    score: float | None = None

    def __post_init__(self):
        validate_interval(self.start, self.end)
        if not self.text.strip(" []()"):
            raise ValueError("Sound description cannot be empty.")


@dataclass
class Project:
    captions: list[Caption]
    language: str = "en"
    speakers: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    sounds: list[Sound] = field(default_factory=list)

    def __post_init__(self):
        # Version 1 projects predate sound cues; they load unchanged and save as the current version.
        if self.schema_version not in (1, SCHEMA_VERSION):
            raise ValueError("Unsupported project schema version.")
        self.schema_version = SCHEMA_VERSION
        if len({c.id for c in self.captions}) != len(self.captions):
            raise ValueError("Caption IDs must be unique.")
        if len({s.id for s in self.sounds}) != len(self.sounds):
            raise ValueError("Sound IDs must be unique.")
        for caption in self.captions:
            self.speakers.setdefault(caption.speaker, caption.speaker)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        payload = dict(data)
        payload["captions"] = [
            Caption(**{**c, "words": [Word(**w) for w in c.get("words", [])]})
            for c in payload["captions"]
        ]
        payload["sounds"] = [Sound(**s) for s in payload.get("sounds", [])]
        return cls(**payload)

    def colors(self):
        return {key: PALETTE[i % len(PALETTE)] for i, key in enumerate(self.speakers)}
