"""Build short cues, splitting on word-level speaker turns without dropping tokens."""
from .schema import Caption, Word


def _floating(word):
    # Untimed words (often numbers/symbols) get no speaker from diarization because they have
    # no time. They carry no evidence of a speaker change, so they stay inside the surrounding cue.
    return word.start is None and word.speaker == "UNKNOWN"


def _cue_speaker(words):
    return next((w.speaker for w in words if not _floating(w)), "UNKNOWN")


def make_captions(segments, max_chars=84, max_duration=6.0):
    captions = []
    for segment in segments:
        raw_words = segment.get("words", [])
        if not raw_words:
            if segment.get("text", "").strip():
                captions.append(Caption(
                    len(captions) + 1, float(segment["start"]), float(segment["end"]),
                    segment.get("speaker", "UNKNOWN"), segment["text"].strip(),
                    segment["text"].strip(),
                ))
            continue
        pending = []

        def flush():
            if not pending:
                return
            text = " ".join(w.text for w in pending).strip()
            timed = [w for w in pending if w.start is not None]
            start = min(w.start for w in timed) if timed else float(segment["start"])
            end = max(w.end for w in timed) if timed else float(segment["end"])
            if end <= start:
                end = start + 0.001
            captions.append(Caption(
                len(captions) + 1, start, end, _cue_speaker(pending), text, text, list(pending)
            ))
            pending.clear()

        for raw in raw_words:
            text = raw.get("word", "").strip()
            if not text:
                continue
            start, end = raw.get("start"), raw.get("end")
            # Alignment sometimes cannot time numbers/symbols. Keep them, explicitly untimed.
            if start is None or end is None or end <= start:
                start = end = None
            word = Word(text, start, end, raw.get("speaker") or "UNKNOWN", raw.get("score"))
            timed = [w for w in pending if w.start is not None]
            too_long = timed and end is not None and end - timed[0].start > max_duration
            too_wide = sum(len(w.text) + 1 for w in pending) + len(text) > max_chars
            speaker_change = (
                not _floating(word) and any(not _floating(w) for w in pending)
                and _cue_speaker(pending) != word.speaker
            )
            if pending and (speaker_change or too_long or too_wide):
                flush()
            pending.append(word)
        flush()
    return captions


def apply_edits(project, rows):
    """Atomic validation. Word timings refer to original text; never fabricate new alignment."""
    from copy import deepcopy
    edited = deepcopy(project)
    by_id = {c.id: c for c in edited.captions}
    if len(rows) != len(by_id) or {int(r["id"]) for r in rows} != set(by_id):
        raise ValueError("Caption IDs must match the current project.")
    for row in rows:
        c = by_id[int(row["id"])]
        c.start, c.end = float(row["start"]), float(row["end"])
        c.text, c.speaker = str(row["text"]).strip(), str(row["speaker"])
        if c.speaker not in edited.speakers:
            raise ValueError("Choose an existing speaker.")
        old = next(old for old in project.captions if old.id == c.id)
        if c.text != old.text:
            c.emotion = c.emotion_score = None
        if (c.text, c.start, c.end) != (old.text, old.start, old.end):
            c.alignment_valid = False
        c.__post_init__()
    edited.captions.sort(key=lambda c: (c.start, c.id))
    return edited
