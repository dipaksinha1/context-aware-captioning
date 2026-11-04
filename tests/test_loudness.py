import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from context_caption.loudness import (
    add_word_levels, has_levels, level_db, read_wav, reference_level, size_scale,
)
from context_caption.player import player_html
from context_caption.schema import Caption, Project, Word


def project():
    return Project([Caption(1, 0.0, 3.0, "A", "quiet LOUD 42", "quiet LOUD 42", [
        Word("quiet", 0.0, 1.0, "A"), Word("LOUD", 1.0, 2.0, "A"), Word("42", None, None, "A")])])


def write_wav(path, samples, rate=16000):
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes((np.asarray(samples) * 32767).astype(np.int16).tobytes())


class LoudnessTests(unittest.TestCase):
    def test_levels_from_wav(self):
        t = np.arange(16000) / 16000
        sine = np.sin(2 * np.pi * 440 * t)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.wav"
            write_wav(path, np.concatenate([0.05 * sine, 0.5 * sine, np.zeros(16000)]))
            samples, rate = read_wav(path)
        p = add_word_levels(project(), samples, rate)
        quiet, loud, untimed = p.captions[0].words
        # A sine's RMS is amplitude / sqrt(2): 0.5 -> -9.0 dBFS, 0.05 -> -29.0 dBFS.
        self.assertAlmostEqual(loud.level_db, -9.0, delta=0.1)
        self.assertAlmostEqual(quiet.level_db, -29.0, delta=0.1)
        self.assertIsNone(untimed.level_db)
        self.assertTrue(has_levels(p))
        self.assertAlmostEqual(reference_level(p), -19.0, delta=0.1)

    def test_silence_and_empty(self):
        self.assertEqual(level_db(np.zeros(100)), -90.0)
        self.assertIsNone(level_db(np.zeros(0)))
        self.assertFalse(has_levels(project()))
        self.assertIsNone(reference_level(project()))

    def test_size_scale(self):
        self.assertEqual(size_scale(-20, -20), 1.0)
        self.assertEqual(size_scale(-10, -20), 1.35)
        self.assertEqual(size_scale(10, -20), 1.45)    # clamped large
        self.assertEqual(size_scale(-60, -20), 0.75)   # clamped small
        self.assertEqual(size_scale(None, -20), 1.0)
        self.assertEqual(size_scale(-10, None), 1.0)

    def test_player_scales_and_toggle(self):
        p = project()
        p.captions[0].words[0].level_db, p.captions[0].words[1].level_db = -29.0, -9.0
        payload = player_html(b"x", "audio/wav", p)
        self.assertIn('"scale": 0.75', payload)   # -10 dB below the median, clamped
        self.assertIn('"scale": 1.35', payload)   # +10 dB above the median
        self.assertIn("dBFS", payload)
        flat = player_html(b"x", "audio/wav", p, loudness=False)
        self.assertNotIn('"scale": 0.75', flat)
        self.assertIn('"scale": 1.0', flat)

    def test_round_trip_and_old_projects(self):
        p = project()
        p.captions[0].words[0].level_db = -21.5
        again = Project.from_dict(json.loads(json.dumps(p.to_dict())))
        self.assertEqual(again.captions[0].words[0].level_db, -21.5)
        old = p.to_dict()
        for w in old["captions"][0]["words"]:
            del w["level_db"]
        self.assertIsNone(Project.from_dict(old).captions[0].words[0].level_db)


if __name__ == "__main__":
    unittest.main()
