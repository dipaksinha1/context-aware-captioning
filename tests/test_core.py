import json
from pathlib import Path
import tempfile
import unittest

from context_caption.captions import apply_edits, make_captions
from context_caption.cli import main
from context_caption.exporters import timestamp, to_json, to_srt, to_vtt
from context_caption.pipeline import Settings
from context_caption.player import player_html
from context_caption.schema import Caption, Project, Word, emotion_label


def fixture():
    return Project([
        Caption(1, 0.1, 1.2, "A", "Hello world.", "Hello world.",
                [Word("Hello", .1, .5, "A"), Word("world.", .6, 1.2, "A")], emotion="joy", emotion_score=.9),
        Caption(2, 1.5, 2.4, "B", "Welcome back.", "Welcome back.")
    ], speakers={"A": "Alex", "B": "Sam"})


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

    def test_untimed_word_without_speaker_stays_in_cue(self):
        # WhisperX leaves untimed words (e.g. numbers) without a speaker. They must not become
        # a separate UNKNOWN cue spanning the whole segment.
        captions = make_captions([{"start": 0, "end": 4, "words": [
            {"word": "In", "start": 0, "end": .3, "speaker": "A"},
            {"word": "2026", "score": None},
            {"word": "we", "start": .8, "end": 1, "speaker": "A"},
            {"word": "Yes", "start": 2, "end": 2.5, "speaker": "B"},
            {"word": "42"}]}])
        self.assertEqual([(c.speaker, c.text) for c in captions], [("A", "In 2026 we"), ("B", "Yes 42")])
        self.assertEqual((captions[1].start, captions[1].end), (2, 2.5))
        self.assertEqual(captions[0].words[1].speaker, "UNKNOWN")
        self.assertIsNone(captions[0].words[1].start)

    def test_leading_untimed_word_joins_next_speaker(self):
        captions = make_captions([{"start": 0, "end": 2, "words": [
            {"word": "$13"}, {"word": "today", "start": 1, "end": 2, "speaker": "B"}]}])
        self.assertEqual([(c.speaker, c.text) for c in captions], [("B", "$13 today")])

    def test_only_untimed_words_stay_unknown(self):
        captions = make_captions([{"start": 0, "end": 2, "words": [{"word": "42"}]}])
        self.assertEqual((captions[0].speaker, captions[0].start, captions[0].end), ("UNKNOWN", 0, 2))

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
        project = fixture()
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
        self.assertIn("Alex [\U0001F604 joy]", to_srt(project, True))
        self.assertIn("Alex [\U0001F604 joy]", to_vtt(project, True))
        self.assertEqual(emotion_label("sadness"), "\U0001F622 sadness")
        self.assertEqual(emotion_label("curious"), "curious")
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

    def test_settings_reject_bad_counts(self):
        with self.assertRaises(ValueError):
            Settings(min_speakers=3, max_speakers=2, token="test").validate()
        self.assertNotIn("secret", repr(Settings(token="secret")))

    def test_cli_export(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "project.json"
            source.write_text(to_json(fixture()), encoding="utf-8")
            dest = Path(directory) / "output"
            main(["--project", str(source), "--output", str(dest)])
            self.assertEqual(len(list(dest.iterdir())), 3)
            self.assertIn("Alex", (dest / "captions.vtt").read_text())


if __name__ == "__main__":
    unittest.main()
