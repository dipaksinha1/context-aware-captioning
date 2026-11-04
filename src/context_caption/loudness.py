"""Per-word loudness for loudness-scaled captions.

Levels are RMS over each aligned word's time span in dBFS (decibels relative to digital full scale)
of the normalized 16 kHz mono waveform. They describe the recording, not sound pressure at the
speaker: microphone distance and gain shift every level, so display sizes are relative to the
recording's median word level rather than to an absolute threshold.
"""
import math
import wave

import numpy as np

FLOOR_DB = -90.0


def read_wav(path):
    """16-bit PCM WAV (as written by media.normalize_audio) -> (float samples in [-1, 1], rate)."""
    with wave.open(str(path)) as stream:
        if stream.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM audio.")
        rate, channels = stream.getframerate(), stream.getnchannels()
        samples = np.frombuffer(stream.readframes(stream.getnframes()), np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples.astype(np.float32) / 32768.0, rate


def level_db(samples):
    if not len(samples):
        return None
    rms = math.sqrt(float(np.mean(np.square(samples, dtype=np.float64))))
    return round(max(FLOOR_DB, 20 * math.log10(rms)) if rms > 0 else FLOOR_DB, 1)


def add_word_levels(project, samples, rate):
    """Set Word.level_db for every timed word; untimed words keep None."""
    for caption in project.captions:
        for word in caption.words:
            if word.start is None:
                word.level_db = None
                continue
            start = int(word.start * rate)
            end = max(int(word.end * rate), start + 1)
            word.level_db = level_db(samples[start:end])
    return project


def has_levels(project):
    return any(w.level_db is not None for c in project.captions for w in c.words)


def reference_level(project):
    """Median word level of the recording, the 1.0x size point."""
    levels = [w.level_db for c in project.captions for w in c.words
              if w.level_db is not None and w.level_db > FLOOR_DB]
    return float(np.median(levels)) if levels else None


def size_scale(level, reference, per_db=0.035, low=0.75, high=1.45):
    """Font-size multiplier: +10 dB above the median is 1.35x; clamped to keep captions readable."""
    if level is None or reference is None:
        return 1.0
    return round(min(high, max(low, 1 + (level - reference) * per_db)), 3)
