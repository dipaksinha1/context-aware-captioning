"""Real WhisperX/pyannote adapter. ML dependencies load only for real processing."""
from dataclasses import dataclass, field
import gc
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from .captions import make_captions
from .media import normalize_audio
from .schema import Project


@dataclass
class Settings:
    model: str = "base"
    device: str = "cpu"
    batch_size: int = 4
    language: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    emotions: bool = False
    token: str = field(default="", repr=False)

    def validate(self):
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Choose cpu or cuda.")
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive.")
        for value in (self.min_speakers, self.max_speakers):
            if value is not None and value < 1:
                raise ValueError("Speaker counts must be positive.")
        if self.min_speakers and self.max_speakers and self.min_speakers > self.max_speakers:
            raise ValueError("Minimum speakers cannot exceed maximum speakers.")
        if not (self.token or os.getenv("HF_TOKEN")):
            raise ValueError("A Hugging Face read token and diarization model access are required.")


def release_memory(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def speaker_turns(captions):
    """Group consecutive cues by the same speaker so emotion sees a whole speaker segment."""
    turns = []
    for caption in sorted(captions, key=lambda c: (c.start, c.id)):
        if turns and turns[-1][0].speaker == caption.speaker:
            turns[-1].append(caption)
        else:
            turns.append([caption])
    return turns


def annotate_emotions(project):
    if project.language != "en":
        project.metadata["emotion_status"] = "Skipped: classifier supports English only."
        return
    from transformers import pipeline
    classifier = pipeline(
        "text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=-1
    )
    turns = speaker_turns(project.captions)
    texts = [" ".join(c.text for c in turn) for turn in turns]
    predictions = classifier(texts, truncation=True, batch_size=8)
    for turn, prediction in zip(turns, predictions):
        # Low confidence is unknown, rather than incorrectly relabeled as neutral.
        if prediction["score"] >= 0.6:
            for c in turn:
                c.emotion = prediction["label"]
                c.emotion_score = float(prediction["score"])
    project.metadata["emotion_status"] = (
        "Text inference per speaker turn; threshold 0.60. Not vocal emotion."
    )


def process(source: Path, settings: Settings, progress=lambda value, label: None):
    settings.validate()
    try:
        import torch
        import whisperx
        from whisperx.diarize import DiarizationPipeline
    except ImportError as exc:
        raise RuntimeError('Install the ML extra first: pip install -e ".[ml]"') from exc
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select CPU.")
    with TemporaryDirectory(prefix="context-caption-") as directory:
        progress(0.05, "Preparing audio")
        wav = Path(directory) / "audio.wav"
        duration = normalize_audio(source, wav)
        audio = whisperx.load_audio(str(wav))
        progress(0.15, "Transcribing speech")
        model = whisperx.load_model(
            settings.model, settings.device,
            compute_type="int8" if settings.device == "cpu" else "float16",
            language=settings.language,
            vad_method="silero",
        )
        result = model.transcribe(audio, batch_size=settings.batch_size)
        language = result.get("language") or settings.language or "en"
        del model
        release_memory(torch)
        if not result.get("segments"):
            return Project([], language, metadata={"duration": duration, "mode": "real"})
        progress(0.4, "Aligning words")
        aligner, metadata = whisperx.load_align_model(language_code=language, device=settings.device)
        aligned = whisperx.align(
            result["segments"], aligner, metadata, audio, settings.device,
            return_char_alignments=False,
        )
        del aligner
        release_memory(torch)
        progress(0.65, "Identifying speaker turns")
        diarizer = DiarizationPipeline(
            model_name="pyannote/speaker-diarization-community-1",
            token=settings.token or os.getenv("HF_TOKEN"), device=settings.device,
        )
        turns = diarizer(
            audio, min_speakers=settings.min_speakers, max_speakers=settings.max_speakers
        )
        aligned = whisperx.assign_word_speakers(turns, aligned, fill_nearest=False)
        del diarizer
        release_memory(torch)
        project = Project(make_captions(aligned["segments"]), language, metadata={
            "duration": duration, "mode": "real", "asr_model": settings.model,
            "diarization_model": "pyannote/speaker-diarization-community-1",
            "word_timings": "Forced alignment; missing word times remain null.",
        })
        if settings.emotions and project.captions:
            progress(0.9, "Estimating text emotion")
            try:
                annotate_emotions(project)
            except Exception:
                # Emotion is optional; a failure must not discard a usable transcript.
                project.metadata["emotion_status"] = "Unavailable. Captions retained without cues."
        progress(1.0, "Ready to review")
        return project
