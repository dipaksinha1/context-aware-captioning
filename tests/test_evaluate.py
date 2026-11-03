import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from context_caption.evaluate import align, diarization, evaluate, main, normalize
from context_caption.schema import Caption, Project, Word


def project(*cues):
    return Project([
        Caption(i, start, end, spk, text, text,
                [Word(t, *(times[k] or (None, None)), spk)
                 for k, t in enumerate(text.split())] if times is not None else [])
        for i, (start, end, spk, text, times) in enumerate(cues, 1)
    ])


class EvaluateTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("Hello, World! It’s $13."), ["hello", "world", "it's", "13"])

    def test_align_counts(self):
        counts, matches = align("a b c d".split(), "a x c d e".split())
        self.assertEqual(counts, {"substitutions": 1, "deletions": 0, "insertions": 1})
        self.assertEqual(matches, [(0, 0), (2, 2), (3, 3)])

    def test_perfect_match_with_swapped_labels(self):
        ref = project((0, 2, "Alex", "Hello there.", [(0, 1), (1, 2)]),
                      (3, 5, "Sam", "Hi Alex.", [(3, 4), (4, 5)]))
        hyp = project((0, 2, "SPEAKER_01", "hello there", [(0, 1), (1, 2)]),
                      (3, 5, "SPEAKER_00", "hi alex", [(3.1, 4), (4, 5.2)]))
        result = evaluate(ref, hyp, collar=0)
        self.assertEqual(result["words"]["wer"], 0)
        self.assertEqual(result["speakers"]["der"], 0)
        self.assertEqual(result["speakers"]["mapping"], {"SPEAKER_01": "Alex", "SPEAKER_00": "Sam"})
        self.assertEqual(result["speakers"]["word_speaker_accuracy"], 1)
        self.assertEqual(result["timestamps"]["compared_words"], 4)
        self.assertAlmostEqual(result["timestamps"]["start_error"]["p95"], 0.1)
        self.assertAlmostEqual(result["timestamps"]["end_error"]["p95"], 0.2)

    def test_der_components(self):
        ref = project((0, 4, "A", "one", None), (4, 8, "B", "two", None))
        hyp = project((0, 6, "X", "one", None), (6, 10, "Y", "two", None))
        s = diarization(ref, hyp, collar=0)
        # 2s of B labelled X (confusion), 2s hypothesis speech after reference ends (false alarm).
        self.assertAlmostEqual(s["speaker_confusion"], 2)
        self.assertAlmostEqual(s["false_alarm_speech"], 2)
        self.assertAlmostEqual(s["missed_speech"], 0)
        self.assertAlmostEqual(s["der"], 0.5)

    def test_collar_excludes_boundaries(self):
        ref = project((0, 4, "A", "one", None), (4, 8, "B", "two", None))
        hyp = project((0, 4.2, "X", "one", None), (4.2, 8, "Y", "two", None))
        self.assertGreater(diarization(ref, hyp, collar=0)["der"], 0)
        self.assertEqual(diarization(ref, hyp, collar=0.25)["der"], 0)

    def test_edited_reference_and_untimed_words(self):
        ref = project((0, 2, "A", "In 2026 we", None))
        hyp = project((0, 2, "S0", "In 2025 we", [(0, .5), None, (1, 2)]))
        result = evaluate(ref, hyp)
        self.assertAlmostEqual(result["words"]["wer"], 1 / 3)
        self.assertEqual(result["timestamps"]["compared_words"], 0)
        self.assertIsNone(result["timestamps"]["start_error"])
        self.assertAlmostEqual(result["timestamps"]["hypothesis_untimed_rate"], 1 / 3)

    def test_invalid_alignment_words_ignored(self):
        ref = project((0, 2, "A", "hello", [(0, 1)]))
        ref.captions[0].alignment_valid = False
        hyp = project((0, 2, "B", "hello", [(0, 1)]))
        self.assertEqual(evaluate(ref, hyp)["timestamps"]["compared_words"], 0)

    def test_empty_hypothesis(self):
        ref = project((0, 2, "A", "hello world", None))
        result = evaluate(ref, Project([]), collar=0)
        self.assertEqual(result["words"]["deletions"], 2)
        self.assertEqual(result["words"]["wer"], 1)
        self.assertAlmostEqual(result["speakers"]["der"], 1)
        self.assertIsNone(result["speakers"]["word_speaker_accuracy"])

    def test_cli(self):
        ref = project((0, 2, "A", "hello world", [(0, 1), (1, 2)]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            path.write_text(json.dumps(ref.to_dict()), encoding="utf-8")
            out = io.StringIO()
            with redirect_stdout(out):
                main(["--reference", str(path), "--hypothesis", str(path)])
            self.assertIn("WER: 0.0%", out.getvalue())
            out = io.StringIO()
            with redirect_stdout(out):
                main(["--reference", str(path), "--hypothesis", str(path), "--json"])
            self.assertEqual(json.loads(out.getvalue())["speakers"]["der"], 0)
            with self.assertRaises(SystemExit):
                main(["--reference", str(path), "--hypothesis", str(Path(tmp) / "missing.json")])


if __name__ == "__main__":
    unittest.main()
