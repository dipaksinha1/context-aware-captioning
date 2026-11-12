"""Adapter contract tests with stubbed ML modules, not model quality tests."""
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import wave

from context_caption.pipeline import Settings, annotate_emotions, process, speaker_turns
from context_caption.schema import Caption, Project, Sound
from context_caption.sounds import RATE, VOCABULARY, detect_sounds


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, *, empty=False, emotion=False, emotion_failure=False, sounds=None):
        torch = ModuleType("torch")
        torch.cuda = SimpleNamespace(is_available=lambda: False)
        whisperx = ModuleType("whisperx")
        whisperx.load_audio = MagicMock(return_value="waveform")
        result = {"language": "en", "segments": [] if empty else [
            {"start": 0, "end": 1, "text": "Hello there"}]}
        asr = MagicMock()
        asr.transcribe.return_value = result
        whisperx.load_model = MagicMock(return_value=asr)
        whisperx.load_align_model = MagicMock(return_value=("aligner", {}))
        aligned = {"segments": [{"start": 0, "end": 1, "text": "Hello there", "words": [
            {"word": "Hello", "start": 0, "end": .4},
            {"word": "there", "start": .5, "end": 1}]}]}
        whisperx.align = MagicMock(return_value=aligned)

        def assign(turns, transcript, fill_nearest):
            self.assertFalse(fill_nearest)
            for word in transcript["segments"][0]["words"]:
                word["speaker"] = "SPEAKER_00"
            return transcript

        whisperx.assign_word_speakers = MagicMock(side_effect=assign)
        diarize = ModuleType("whisperx.diarize")
        diarize.DiarizationPipeline = MagicMock(return_value=MagicMock(return_value="turns"))
        transformers = ModuleType("transformers")
        if emotion_failure:
            transformers.pipeline = MagicMock(side_effect=RuntimeError("offline"))
        else:
            transformers.pipeline = MagicMock(return_value=MagicMock(return_value=[
                {"label": "joy", "score": .85}]))
        modules = {"torch": torch, "whisperx": whisperx, "whisperx.diarize": diarize,
                   "transformers": transformers}
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", modules):
            source = Path(directory) / "input.wav"
            with wave.open(str(source), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\0" * 32000)
            progress = MagicMock()
            settings = Settings(token="secret-test-token", emotions=emotion,
                                sounds=sounds is not None)
            with patch("context_caption.pipeline.detect_sounds", side_effect=sounds) as detect:
                project = process(source, settings, progress)
            self.assertEqual(detect.call_count, int(sounds is not None))
        self.assertNotIn("secret-test-token", str(project.to_dict()))
        return project, whisperx, diarize

    def test_real_adapter_contract(self):
        project, whisperx, diarize = self.run_pipeline()
        self.assertEqual(project.captions[0].text, "Hello there")
        self.assertEqual(project.captions[0].speaker, "SPEAKER_00")
        self.assertEqual(whisperx.load_model.call_args.kwargs["compute_type"], "int8")
        self.assertEqual(diarize.DiarizationPipeline.call_args.kwargs["model_name"],
                         "pyannote/speaker-diarization-community-1")

    def test_no_speech_skips_alignment_and_diarization(self):
        project, whisperx, diarize = self.run_pipeline(empty=True)
        self.assertEqual(project.captions, [])
        whisperx.align.assert_not_called()
        diarize.DiarizationPipeline.assert_not_called()

    def test_optional_emotion_and_failure(self):
        project, _, _ = self.run_pipeline(emotion=True)
        self.assertEqual(project.captions[0].emotion, "joy")
        project, _, _ = self.run_pipeline(emotion=True, emotion_failure=True)
        self.assertEqual(project.captions[0].text, "Hello there")
        self.assertIsNone(project.captions[0].emotion)
        self.assertIn("Unavailable", project.metadata["emotion_status"])

    def test_optional_sounds_and_failure(self):
        bark = [Sound(1, 0.5, 2, "dog barking", "dog barking", .9)]
        project, _, _ = self.run_pipeline(sounds=lambda audio, device, threshold: bark)
        self.assertEqual(project.sounds, bark)
        self.assertIn("ast-finetuned-audioset", project.metadata["sound_model"])
        project, _, _ = self.run_pipeline(sounds=RuntimeError("offline"))
        self.assertEqual((project.captions[0].text, project.sounds), ("Hello there", []))
        self.assertIn("Unavailable", project.metadata["sound_status"])
        self.assertEqual(self.run_pipeline()[0].sounds, [])

    def test_sounds_detected_without_speech(self):
        bark = [Sound(1, 0.5, 2, "dog barking", "dog barking", .9)]
        project, whisperx, _ = self.run_pipeline(empty=True, sounds=lambda *args: bark)
        self.assertEqual((project.captions, project.sounds), ([], bark))
        whisperx.align.assert_not_called()

    def test_sound_detector_windows_and_scores(self):
        """Stubbed tagger: checks windowing, silence skipping, and sigmoid class grouping."""
        import numpy as np
        names = ["Speech", "Bark", "Slam"] + sorted(
            {n for group in VOCABULARY.values() for n in group} - {"Bark", "Slam"})
        seen = []

        class Extractor:
            def __call__(self, chunks, sampling_rate, return_tensors):
                seen.extend(len(chunk) for chunk in chunks)
                return SimpleNamespace(input_values=FakeTensor(
                    [[float(np.abs(chunk).max())] for chunk in chunks]))

        class FakeTensor:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=float)

            def to(self, device):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        def model(inputs):
            # Loud windows "bark"; speech always scores highest and must never become a cue.
            logits = np.full((len(inputs.values), len(names)), -9.0)
            logits[:, 0] = 9.0
            logits[inputs.values[:, 0] > .5, 1] = 2.0
            return SimpleNamespace(logits=FakeTensor(logits))

        model.config = SimpleNamespace(id2label=dict(enumerate(names)))
        model.to = lambda device: model
        model.eval = lambda: model
        torch = ModuleType("torch")
        torch.inference_mode = MagicMock()
        torch.sigmoid = lambda tensor: FakeTensor(1 / (1 + np.exp(-tensor.values)))
        transformers = ModuleType("transformers")
        transformers.AutoFeatureExtractor = SimpleNamespace(from_pretrained=lambda name: Extractor())
        transformers.AutoModelForAudioClassification = SimpleNamespace(
            from_pretrained=lambda name: model)
        audio = np.zeros(10 * RATE, dtype=np.float32)
        audio[:6 * RATE] = .1   # quiet speech-like level for 0-6 s
        audio[2 * RATE:4 * RATE] = .9  # loud event at 2-4 s; 6-10 s is digital silence
        with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
            sounds = detect_sounds(audio)
        self.assertEqual(seen, [2 * RATE] * 6)  # windows at 0-5 s; silent ones never classified
        self.assertEqual([(s.text, s.start, s.end) for s in sounds], [("dog barking", 2.0, 4.0)])
        self.assertAlmostEqual(sounds[0].score, .881, places=3)

    def test_emotion_classifies_whole_speaker_turns(self):
        captions = [Caption(1, 0, 1, "A", "I passed", "I passed"),
                    Caption(2, 1, 2, "A", "the exam!", "the exam!"),
                    Caption(3, 2, 3, "B", "Oh no.", "Oh no."),
                    Caption(4, 3, 4, "A", "Really.", "Really.")]
        self.assertEqual([[c.id for c in t] for t in speaker_turns(captions)], [[1, 2], [3], [4]])
        project = Project(captions, "en")
        classifier = MagicMock(return_value=[{"label": "joy", "score": .9},
                                             {"label": "sadness", "score": .7},
                                             {"label": "neutral", "score": .4}])
        transformers = ModuleType("transformers")
        transformers.pipeline = MagicMock(return_value=classifier)
        with patch.dict("sys.modules", {"transformers": transformers}):
            annotate_emotions(project)
        self.assertEqual(classifier.call_args.args[0], ["I passed the exam!", "Oh no.", "Really."])
        self.assertEqual([c.emotion for c in project.captions], ["joy", "joy", "sadness", None])
        self.assertEqual([c.text for c in project.captions],
                         ["I passed", "the exam!", "Oh no.", "Really."])
