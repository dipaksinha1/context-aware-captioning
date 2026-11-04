"""Adapter contract tests with stubbed ML modules, not model quality tests."""
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import wave

import numpy as np

from context_caption.pipeline import Settings, annotate_emotions, process, speaker_turns
from context_caption.schema import Caption, Project


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, *, empty=False, emotion=False, emotion_failure=False):
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
                # Silent first half ("Hello"), then a 1 kHz square wave at 0.25 full scale ("there").
                tone = np.tile(np.array([8192] * 8 + [-8192] * 8, np.int16), 500)
                stream.writeframes(np.zeros(8000, np.int16).tobytes() + tone.tobytes())
            progress = MagicMock()
            project = process(source, Settings(token="secret-test-token", emotions=emotion), progress)
        self.assertNotIn("secret-test-token", str(project.to_dict()))
        return project, whisperx, diarize

    def test_real_adapter_contract(self):
        project, whisperx, diarize = self.run_pipeline()
        self.assertEqual(project.captions[0].text, "Hello there")
        self.assertEqual(project.captions[0].speaker, "SPEAKER_00")
        self.assertEqual(whisperx.load_model.call_args.kwargs["compute_type"], "int8")
        self.assertEqual(diarize.DiarizationPipeline.call_args.kwargs["model_name"],
                         "pyannote/speaker-diarization-community-1")
        hello, there = project.captions[0].words
        self.assertEqual(hello.level_db, -90.0)
        self.assertAlmostEqual(there.level_db, -12.0, delta=0.1)

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
