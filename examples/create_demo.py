"""Rebuild the synthetic two-voice demo using FFmpeg with the flite filter.

Two synthesized sound effects (a doorbell and a beep) show how sound cues look beside speech.
"""
from pathlib import Path
import json
import math
import random
import struct
import subprocess
from tempfile import TemporaryDirectory
import wave

from context_caption.schema import Caption, Project, Sound, Word


def chime(rate, pitch, length, decay):
    """A struck-bell tone: a decaying fundamental with two quieter harmonics."""
    return [math.exp(-decay * n / rate) * sum(
        level * math.sin(2 * math.pi * pitch * harmonic * n / rate)
        for harmonic, level in ((1, 1.0), (2, 0.5), (3, 0.25))) for n in range(int(length * rate))]


def doorbell(rate):
    """Ding-dong: a high chime, then a lower one."""
    return chime(rate, 659, 0.7, 4) + chime(rate, 523, 1.1, 3)


def beeps(rate, count=3, length=0.2):
    """1 kHz tone bursts separated by equal silences."""
    burst = [math.sin(2 * math.pi * 1000 * n / rate) for n in range(int(length * rate))]
    return (burst + [0.0] * len(burst)) * count


def mix(frames, at, samples, gain):
    """Add float samples into 16-bit mono PCM, starting at sample index `at`."""
    for n, value in enumerate(samples):
        position = 2 * (at + n)
        mixed = struct.unpack_from("<h", frames, position)[0] + int(gain * 32767 * value)
        struct.pack_into("<h", frames, position, max(-32768, min(32767, mixed)))


def create():
    lines = [
        ("SPEAKER_00", "slt", "Welcome to our captioning project."),
        ("SPEAKER_01", "rms", "Every speaker gets a consistent label."),
        ("SPEAKER_00", "slt", "You can edit captions and rename the voices."),
        ("SPEAKER_01", "rms", "Then export subtitles for your own media player."),
    ]
    rate = 16000
    frames = bytearray(b"\0" * rate)  # half-second lead-in, 16-bit mono
    captions, sounds = [], []
    with TemporaryDirectory() as directory:
        for i, (speaker, voice, text) in enumerate(lines, 1):
            part = Path(directory) / f"part{i}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                            f"flite=text='{text}':voice={voice}", "-ar", str(rate),
                            "-ac", "1", "-c:a", "pcm_s16le", str(part)], check=True)
            with wave.open(str(part)) as stream:
                audio = stream.readframes(stream.getnframes())
            start = len(frames) / (2 * rate)
            frames.extend(audio)
            end = len(frames) / (2 * rate)
            tokens = text.split()
            step = (end - start) / len(tokens)
            words = [Word(t, start + n * step, start + (n + 1) * step, speaker)
                     for n, t in enumerate(tokens)]
            captions.append(Caption(i, start, end, speaker, text, text, words))
            if i == 1:  # a doorbell in a longer pause, before the second voice answers
                at = len(frames) // 2 + rate // 4
                frames.extend(b"\0" * (5 * rate))
                mix(frames, at, doorbell(rate), 0.25)
                sounds.append(Sound(1, at / rate, at / rate + 1.8, "doorbell rings",
                                    "doorbell rings"))
            else:
                frames.extend(b"\0" * rate)
            if i == 3:  # beeping in the background, underneath speech
                at = int((start + 0.5) * rate)
                mix(frames, at, beeps(rate), 0.12)
                sounds.append(Sound(2, at / rate, at / rate + 1.2, "beeping", "beeping"))
    # Faint room tone. Real recordings are never digitally silent, and the sound tagger
    # misses the doorbell when it is surrounded by exact zeros.
    hiss = random.Random(7)
    mix(frames, 0, (hiss.uniform(-1, 1) for _ in range(len(frames) // 2)), 0.002)
    project = Project(captions, speakers={"SPEAKER_00": "Alex", "SPEAKER_01": "Sam"}, metadata={
        "mode": "demo", "duration": len(frames) / (2 * rate),
        "note": "Flite synthetic speech, predefined captions, and illustrative evenly spaced word times. "
                "The doorbell and beep are synthesized effects with predefined sound cues. "
                "No ASR, forced alignment, speaker inference, or sound detection was performed.",
    }, sounds=sounds)
    folder = Path(__file__).parent
    (folder / "demo_project.json").write_text(json.dumps(project.to_dict(), indent=2), encoding="utf-8")
    with wave.open(str(folder / "demo.wav"), "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(frames)


if __name__ == "__main__":
    create()
