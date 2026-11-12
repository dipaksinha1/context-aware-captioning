import json
from pathlib import Path
import tempfile
import unittest

from context_caption.captions import apply_edits, make_captions
from context_caption.cli import main
from context_caption.exporters import timestamp, to_json, to_srt, to_vtt
from context_caption.pipeline import Settings
from context_caption.player import player_html
from context_caption.schema import Caption, Project, Sound, Word
from context_caption.sounds import apply_sound_edits, build_sounds, format_sound


def fixture():
    return Project([
        Caption(1, 0.1, 1.2, "A", "Hello world.", "Hello world.",
                [Word("Hello", .1, .5, "A"), Word("world.", .6, 1.2, "A")], emotion="joy", emotion_score=.9),
        Caption(2, 1.5, 2.4, "B", "Welcome back.", "Welcome back.")
    ], speakers={"A": "Alex", "B": "Sam"})


def sound_fixture():
    project = fixture()
    project.sounds = [Sound(1, 1.0, 2.5, "dog barking", "dog barking", .8),
                      Sound(2, 3.0, 4.5, "door slams", "door slams", .6)]
    return project


def frames(*scores, hop=1.0):
    """Half-overlapping two-second windows, one score dict per window."""
    return [(i * hop, i * hop + 2 * hop, score) for i, score in enumerate(scores)]


class CaptionTests(unittest.TestCase):
    def test_speaker_change_within_segment(self):
        captions = make_captions([{"start": 0, "end": 3, "text": "Hello Sam thanks",
            "words": [{"word": "Hello", "start": 0, "end": .5, "speaker": "A"},
                      {"word": "Sam", "start": .6, "end": 1, "speaker": "A"},
                      {"word": "thanks", "start": 1.5, "end": 2, "speaker": "B"}]}])
        self.assertEqual([c.speaker for c in captions], ["A", "B"])
        self.assertEqual([c.text for c in captions], ["Hello Sam", "thanks"])

    def test_untimed_tokens_preserved(self):
        captions = make_captions([{"start": 0, "end": 2, "words": [
            {"word": "$13", "speaker": "A"},
            {"word": "today", "start": 1, "end": 2, "speaker": "A"}]}])
        self.assertEqual(captions[0].text, "$13 today")
        self.assertIsNone(captions[0].words[0].start)

    def test_no_words_fallback(self):
        captions = make_captions([{"start": 0, "end": 2, "text": "Hello"}])
        self.assertEqual(captions[0].speaker, "UNKNOWN")
        self.assertEqual(captions[0].text, "Hello")

    def test_unknown_word_does_not_inherit_dominant_speaker(self):
        captions = make_captions([{"start": 0, "end": 2, "speaker": "A", "words": [
            {"word": "Uncertain", "start": 0, "end": 1}]}])
        self.assertEqual(captions[0].speaker, "UNKNOWN")

    def test_limits_split_and_retain_text(self):
        words = [{"word": str(i), "start": i, "end": i + .8, "speaker": "A"}
                 for i in range(15)]
        captions = make_captions([{"start": 0, "end": 15, "words": words}], max_duration=3)
        self.assertTrue(all(c.end - c.start <= 3 for c in captions))
        self.assertEqual(" ".join(c.text for c in captions), " ".join(str(i) for i in range(15)))

    def test_atomic_edit(self):
        project = fixture()
        rows = [{"id": c.id, "start": c.start, "end": c.end,
                 "speaker": c.speaker, "text": c.text} for c in project.captions]
        rows[0]["text"] = "Corrected words"
        edited = apply_edits(project, rows)
        self.assertEqual(project.captions[0].text, "Hello world.")
        self.assertEqual(edited.captions[0].original_text, "Hello world.")
        self.assertIsNone(edited.captions[0].emotion)
        rows[0]["end"] = -1
        with self.assertRaises(ValueError):
            apply_edits(project, rows)
        self.assertEqual(project.captions[0].end, 1.2)

    def test_invalid_timestamps(self):
        for start, end in [(0, 0), (-1, 1), (float("nan"), 3), (1, float("inf"))]:
            with self.assertRaises(ValueError):
                Caption(1, start, end, "A", "Hello", "Hello")

    def test_json_round_trip(self):
        project = sound_fixture()
        restored = Project.from_dict(json.loads(to_json(project)))
        self.assertEqual(restored.to_dict(), project.to_dict())
        self.assertEqual(restored.colors(), project.colors())

    def test_millisecond_carry(self):
        self.assertEqual(timestamp(59.9999), "00:01:00.000")
        self.assertEqual(timestamp(3600.1, ","), "01:00:00,100")

    def test_exports(self):
        project = fixture()
        self.assertIn("Alex: Hello world.", to_srt(project))
        self.assertNotIn("joy", to_srt(project))
        self.assertIn("Alex [joy]", to_srt(project, True))
        self.assertIn("00:00:00,100 --> 00:00:01,200", to_srt(project))
        self.assertTrue(to_vtt(project).startswith("WEBVTT\n\nSTYLE"))
        project.captions[0].text = "<script>& danger"
        self.assertIn("&lt;script&gt;&amp;", to_vtt(project))

    def test_player_injection_and_edited_text(self):
        project = fixture()
        project.captions[0].text = "</script><script>alert(1)</script>"
        markup = player_html(b"abc", "audio/wav", project)
        self.assertNotIn("</script><script>alert", markup)
        self.assertIn('"words": []', markup)
        self.assertIn("\\u003c/script", markup)

    def test_player_sound_cues(self):
        project = sound_fixture()
        project.sounds[0].text = "</script>"
        markup = player_html(b"abc", "audio/wav", project)
        self.assertIn('"text": "[door slams]"', markup)
        self.assertIn('"[\\u003c/script\\u003e]"', markup)
        self.assertIn('"sounds": []', player_html(b"abc", "audio/wav", project, sounds=False))

    def test_sound_cues_are_bracketed_and_separate_from_speech(self):
        project = sound_fixture()
        srt = to_srt(project)
        self.assertIn("2\n00:00:01,000 --> 00:00:02,500\n[dog barking]\n", srt)
        self.assertNotIn(": [dog barking]", srt)
        self.assertLess(srt.index("Alex: Hello"), srt.index("[dog barking]"))
        self.assertLess(srt.index("[dog barking]"), srt.index("Sam: Welcome"))
        vtt = to_vtt(project)
        self.assertIn("sound-2\n00:00:03.000 --> 00:00:04.500\n[door slams]\n", vtt)
        self.assertNotIn("[dog barking]", to_srt(project, sounds=False))
        self.assertNotIn("[dog barking]", to_vtt(project, sounds=False))
        project.sounds[0].text = "[<b>Rex</b> barking]"
        self.assertEqual(format_sound(project.sounds[0]), "[<b>Rex</b> barking]")
        self.assertIn("[&lt;b&gt;Rex&lt;/b&gt; barking]", to_vtt(project))

    def test_sound_windows_become_cues(self):
        quiet, bark, slam = {}, {"dog barking": .9}, {"door slams": .5, "thud": .4, "crash": .35}
        sounds = build_sounds(frames(quiet, bark, bark, bark, quiet, quiet, slam, slam, quiet))
        self.assertEqual([(s.text, s.start, s.end) for s in sounds], [
            # Three overlapping windows (1-3, 2-4, 3-5) share 2-4; two (6-8, 7-9) share 7-8,
            # which is then held for the 1.5 s minimum reading time.
            ("dog barking", 2.0, 4.0), ("door slams", 7.0, 8.5), ("thud", 7.0, 8.5)])
        self.assertEqual(sounds[0].score, .9)
        self.assertEqual([s.id for s in sounds], [1, 2, 3])
        self.assertEqual(build_sounds(frames({"dog barking": .24})), [])
        self.assertEqual(build_sounds([]), [])

    def test_sound_cue_marks_onset_and_merges_brief_pauses(self):
        rain, quiet = {"rain falling": .7}, {}
        long = build_sounds(frames(*[rain] * 30))
        self.assertEqual([(s.start, s.end) for s in long], [(1.0, 7.0)])
        paused = build_sounds(frames(rain, rain, quiet, quiet, rain, rain, *[quiet] * 8, rain))
        self.assertEqual([(s.start, s.end) for s in paused], [(1.0, 6.0), (14.0, 16.0)])
        clipped = build_sounds(frames(rain, hop=.5))
        self.assertEqual([(s.start, s.end) for s in clipped], [(0.0, 1.0)])

    def test_sound_edits(self):
        project = sound_fixture()
        rows = [{"id": 1, "start": 1.0, "end": 2.5, "text": " [Dog  whining] "},
                {"id": None, "start": 5, "end": 6, "text": "rain falling"},
                {"id": float("nan"), "start": None, "end": float("nan"), "text": ""}]
        edited = apply_sound_edits(project, rows)
        self.assertEqual([(s.id, s.text, s.original_text, s.score) for s in edited.sounds], [
            (1, "Dog whining", "dog barking", .8), (3, "rain falling", "rain falling", None)])
        self.assertEqual(len(project.sounds), 2)
        self.assertEqual(edited.captions, project.captions)
        for bad in ({"id": 1, "start": 2, "end": 1, "text": "thud"},
                    {"id": 1, "start": 1, "end": 2, "text": "[]"},
                    {"id": 1, "start": 1, "end": None, "text": "thud"}):
            with self.assertRaises(ValueError):
                apply_sound_edits(project, [bad])
        self.assertEqual(apply_sound_edits(project, []).sounds, [])

    def test_version_1_project_loads_without_sounds(self):
        legacy = fixture().to_dict()
        del legacy["sounds"]
        legacy["schema_version"] = 1
        restored = Project.from_dict(legacy)
        self.assertEqual((restored.sounds, restored.schema_version), ([], 2))
        with self.assertRaises(ValueError):
            Project.from_dict({**legacy, "schema_version": 3})
        with self.assertRaises(ValueError):
            Project([], sounds=[Sound(1, 0, 1, "thud", "thud"), Sound(1, 2, 3, "thud", "thud")])

    def test_settings_reject_bad_counts(self):
        with self.assertRaises(ValueError):
            Settings(min_speakers=3, max_speakers=2, token="test").validate()
        with self.assertRaises(ValueError):
            Settings(sound_threshold=1.5, token="test").validate()
        self.assertNotIn("secret", repr(Settings(token="secret")))

    def test_cli_export(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "project.json"
            source.write_text(to_json(sound_fixture()), encoding="utf-8")
            dest = Path(directory) / "output"
            main(["--project", str(source), "--output", str(dest)])
            self.assertEqual(len(list(dest.iterdir())), 3)
            self.assertIn("Alex", (dest / "captions.vtt").read_text())
            self.assertIn("[dog barking]", (dest / "captions.srt").read_text())


if __name__ == "__main__":
    unittest.main()
