"""Real WhisperX/pyannote/AST adapter. ML dependencies load only for real processing."""
from dataclasses import dataclass, field
import gc
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from .captions import make_captions
from .media import normalize_audio
from .schema import Project
from .sounds import MODEL as SOUND_MODEL, THRESHOLD, detect_sounds


@dataclass
class Settings:
    model: str = "base"
    device: str = "cpu"
    batch_size: int = 4
    language: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    emotions: bool = False
    sounds: bool = False
    sound_threshold: float = THRESHOLD
    token: str = field(default="", repr=False)

    def validate(self):
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Choose cpu or cuda.")
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive.")
        for value in (self.min_speakers, self.max_speakers):
            if value is not None and value < 1:
                raise ValueError("Speaker counts must be positive.")
        if not 0 < self.sound_threshold < 1:
            raise ValueError("Sound threshold must be between 0 and 1.")
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
        metadata = {"duration": duration, "mode": "real", "asr_model": settings.model}
        captions = []
        if result.get("segments"):
            progress(0.4, "Aligning words")
            aligner, align_metadata = whisperx.load_align_model(
                language_code=language, device=settings.device
            )
            aligned = whisperx.align(
                result["segments"], aligner, align_metadata, audio, settings.device,
                return_char_alignments=False,
            )
            del aligner
            release_memory(torch)
            progress(0.6, "Identifying speaker turns")
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
            captions = make_captions(aligned["segments"])
            metadata.update({
                "diarization_model": "pyannote/speaker-diarization-community-1",
                "word_timings": "Forced alignment; missing word times remain null.",
            })
        project = Project(captions, language, metadata=metadata)
        if settings.sounds:
            # Runs even without speech: a recording can be all rain and barking.
            progress(0.8, "Detecting background sounds")
            try:
                project.sounds = detect_sounds(audio, settings.device, settings.sound_threshold)
                project.metadata["sound_model"] = SOUND_MODEL
                project.metadata["sound_status"] = (
                    f"Audio tagging on 2 s windows; threshold {settings.sound_threshold:.2f}. "
                    "Cue times are approximate; review before export."
                )
            except Exception:
                # Sound cues are optional; a failure must not discard a usable transcript.
                project.metadata["sound_status"] = "Unavailable. Captions retained without cues."
            release_memory(torch)
        if settings.emotions and project.captions:
            progress(0.9, "Estimating text emotion")
            try:
                annotate_emotions(project)
            except Exception:
                # Emotion is optional; a failure must not discard a usable transcript.
                project.metadata["emotion_status"] = "Unavailable. Captions retained without cues."
        progress(1.0, "Ready to review")
        return project
