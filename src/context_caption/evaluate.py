"""Score a machine-output project against a manually checked reference project.

Both inputs use the project JSON schema. Build the reference by correcting a copy of the output in the
app (text, speakers, timing) and saving it. Metrics follow docs/EVALUATION.md:

- WER over normalized tokens (lowercase, NFKC, punctuation removed; numbers are not expanded).
- DER over caption-level speaker turns, with optimal one-to-one label mapping and a scoring collar
  around reference boundaries. Overlapping reference turns are scored, not excluded.
- Word speaker accuracy on matched words, under the same label mapping.
- Word timestamp error on matched words timed in both projects; untimed-word rate reported separately.
"""
import argparse
from itertools import permutations
import json
from pathlib import Path
import statistics
import sys
import unicodedata

from .schema import Project

FRAME = 0.01


def normalize(text):
    tokens = []
    for raw in unicodedata.normalize("NFKC", text).lower().replace("’", "'").split():
        token = "".join(ch for ch in raw if ch.isalnum() or ch == "'").strip("'")
        if token:
            tokens.append(token)
    return tokens


def _tokens(project):
    """(token, speaker) from caption text, which is where reference corrections are made."""
    return [(t, c.speaker) for c in project.captions for t in normalize(c.text)]


def _timed_words(project):
    """(token, start, end, speaker) from word records still valid for their caption text."""
    out = []
    for c in project.captions:
        if not c.alignment_valid:
            continue
        for w in c.words:
            for t in normalize(w.text):
                out.append((t, w.start, w.end, c.speaker))
    return out


def align(ref, hyp):
    """Levenshtein alignment. Returns counts and (ref_index, hyp_index) pairs of exact matches."""
    n, m = len(ref), len(hyp)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dist[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
            dist[i][j] = min(sub, dist[i - 1][j] + 1, dist[i][j - 1] + 1)
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0}
    matches = []
    i, j = n, m
    while i or j:
        if i and j and dist[i][j] == dist[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] == hyp[j - 1]:
                matches.append((i - 1, j - 1))
            else:
                counts["substitutions"] += 1
            i, j = i - 1, j - 1
        elif i and dist[i][j] == dist[i - 1][j] + 1:
            counts["deletions"] += 1
            i -= 1
        else:
            counts["insertions"] += 1
            j -= 1
    matches.reverse()
    return counts, matches


def _frames(project, total):
    frames = [set() for _ in range(total)]
    for c in project.captions:
        for k in range(round(c.start / FRAME), min(round(c.end / FRAME), total)):
            frames[k].add(c.speaker)
    return frames


def _best_mapping(overlap, ref_labels, hyp_labels):
    """Hypothesis->reference mapping maximizing overlap; exhaustive when small, greedy otherwise."""
    small, large = sorted([ref_labels, hyp_labels], key=len)
    if len(large) <= 8:
        best, best_score = {}, -1
        for perm in permutations(large, len(small)):
            pairs = zip(small, perm) if small is ref_labels else zip(perm, small)
            pairs = list(pairs)
            score = sum(overlap.get(p, 0) for p in pairs)
            if score > best_score:
                best, best_score = {h: r for r, h in pairs}, score
        return best
    mapping, used = {}, set()
    for (r, h), _ in sorted(overlap.items(), key=lambda item: -item[1]):
        if h not in mapping and r not in used:
            mapping[h] = r
            used.add(r)
    return mapping


def diarization(reference, hypothesis, collar=0.25):
    end = max([c.end for c in reference.captions + hypothesis.captions], default=0)
    total = round(end / FRAME) + 1
    ref_frames, hyp_frames = _frames(reference, total), _frames(hypothesis, total)
    scored = [True] * total
    half = round(collar / FRAME)
    for c in reference.captions:
        for edge in (round(c.start / FRAME), round(c.end / FRAME)):
            for k in range(max(0, edge - half), min(total, edge + half)):
                scored[k] = False
    overlap = {}
    for k in range(total):
        if scored[k]:
            for r in ref_frames[k]:
                for h in hyp_frames[k]:
                    overlap[(r, h)] = overlap.get((r, h), 0) + 1
    mapping = _best_mapping(overlap, sorted({c.speaker for c in reference.captions}),
                            sorted({c.speaker for c in hypothesis.captions}))
    missed = false_alarm = confusion = speech = 0
    for k in range(total):
        if not scored[k]:
            continue
        r, h = ref_frames[k], hyp_frames[k]
        speech += len(r)
        missed += max(0, len(r) - len(h))
        false_alarm += max(0, len(h) - len(r))
        confusion += min(len(r), len(h)) - len(r & {mapping.get(x) for x in h})
    seconds = {"reference_speech": speech * FRAME, "missed_speech": missed * FRAME,
               "false_alarm_speech": false_alarm * FRAME, "speaker_confusion": confusion * FRAME}
    seconds = {key: round(value, 3) for key, value in seconds.items()}
    return {**seconds, "der": (missed + false_alarm + confusion) / speech if speech else None,
            "collar": collar, "mapping": mapping}


def _percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def evaluate(reference, hypothesis, collar=0.25):
    ref, hyp = _tokens(reference), _tokens(hypothesis)
    counts, matches = align([t for t, _ in ref], [t for t, _ in hyp])
    errors = sum(counts.values())
    speakers = diarization(reference, hypothesis, collar)
    mapping = speakers["mapping"]
    agree = sum(mapping.get(hyp[j][1]) == ref[i][1] for i, j in matches)

    ref_words, hyp_words = _timed_words(reference), _timed_words(hypothesis)
    _, word_matches = align([w[0] for w in ref_words], [w[0] for w in hyp_words])
    start_err, end_err = [], []
    for i, j in word_matches:
        r, h = ref_words[i], hyp_words[j]
        if r[1] is not None and h[1] is not None:
            start_err.append(abs(r[1] - h[1]))
            end_err.append(abs(r[2] - h[2]))
    hyp_all = [w for c in hypothesis.captions for w in c.words]

    def summary(values):
        if not values:
            return None
        return {"median": round(statistics.median(values), 3),
                "p95": round(_percentile(values, 0.95), 3)}

    return {
        "words": {"reference_words": len(ref), "hypothesis_words": len(hyp), **counts,
                  "wer": errors / len(ref) if ref else None},
        "speakers": {**speakers, "matched_words": len(matches),
                     "word_speaker_accuracy": agree / len(matches) if matches else None},
        "timestamps": {"compared_words": len(start_err), "start_error": summary(start_err),
                       "end_error": summary(end_err),
                       "hypothesis_untimed_rate": (sum(w.start is None for w in hyp_all) / len(hyp_all)
                                                   if hyp_all else None)},
    }


def _fmt(value, percent=False):
    if value is None:
        return "n/a"
    return f"{value:.1%}" if percent else f"{value}"


def report(result):
    w, s, t = result["words"], result["speakers"], result["timestamps"]
    lines = [
        f"WER: {_fmt(w['wer'], True)} ({w['substitutions']} sub, {w['deletions']} del, "
        f"{w['insertions']} ins / {w['reference_words']} reference words)",
        f"DER: {_fmt(s['der'], True)} (missed {s['missed_speech']}s, false alarm "
        f"{s['false_alarm_speech']}s, confusion {s['speaker_confusion']}s / "
        f"{s['reference_speech']}s reference speech, collar {s['collar']}s)",
        "Speaker mapping (hypothesis -> reference): "
        + (", ".join(f"{h} -> {r}" for h, r in sorted(s["mapping"].items())) or "none"),
        f"Word speaker accuracy: {_fmt(s['word_speaker_accuracy'], True)} "
        f"over {s['matched_words']} matched words",
    ]
    if t["compared_words"]:
        lines.append(f"Word start error: median {t['start_error']['median']}s, "
                     f"p95 {t['start_error']['p95']}s; end error: median "
                     f"{t['end_error']['median']}s, p95 {t['end_error']['p95']}s "
                     f"over {t['compared_words']} words")
    else:
        lines.append("Word timestamps: n/a (no matched words timed in both projects)")
    lines.append(f"Hypothesis untimed-word rate: {_fmt(t['hypothesis_untimed_rate'], True)}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score captions against a reference project")
    parser.add_argument("--reference", type=Path, required=True, help="Manually checked project JSON")
    parser.add_argument("--hypothesis", type=Path, required=True, help="Unedited model project JSON")
    parser.add_argument("--collar", type=float, default=0.25,
                        help="Seconds excluded on each side of reference turn boundaries")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    args = parser.parse_args(argv)
    if args.collar < 0:
        parser.error("--collar must be non-negative")
    try:
        load = lambda p: Project.from_dict(json.loads(p.read_text(encoding="utf-8")))  # noqa: E731
        result = evaluate(load(args.reference), load(args.hypothesis), args.collar)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, indent=2) if args.json else report(result))


if __name__ == "__main__":
    sys.exit(main())
