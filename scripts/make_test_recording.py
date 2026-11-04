"""Generate a synthetic multi-speaker test video plus its reference project for evaluation.

Uses Piper neural text-to-speech voices (downloaded from Hugging Face on first use) and FFmpeg.
The reference holds the scripted text, speaker, and utterance timing for every turn; it has no word
timestamps, so word-timing error is not measurable from it. Synthetic voices are cleaner than real
conversation, so scores from this file are a pipeline check, not an accuracy claim.

    pip install piper-tts
    python scripts/make_test_recording.py --output outputs/test_media [--noise 0.02]
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from context_caption.schema import Caption, Project  # noqa: E402

RATE = 22050
VOICES = {
    "Maya": "en_US-lessac-medium",
    "Raj": "en_US-ryan-medium",
    "Alba": "en_GB-alba-medium",
}
# Alternating and returning speakers, a short reply, rapid turns, numbers, and proper nouns.
SCRIPT = [
    ("Maya", "Good morning everyone, thanks for joining the accessibility review.", 0.7),
    ("Raj", "Happy to be here. I tested the caption editor on Monday.", 0.6),
    ("Maya", "Great. What did you find?", 0.5),
    ("Raj", "Speaker labels stayed consistent, but long sentences were split too early.", 0.8),
    ("Alba", "I noticed that as well, especially in the lecture recording from Toronto.", 0.6),
    ("Maya", "Was the timing accurate?", 0.3),
    ("Alba", "Mostly. Each caption appeared within a fraction of a second.", 0.7),
    ("Raj", "Yes.", 0.3),
    ("Maya", "Good. The budget for the next study is about 300 dollars.", 0.6),
    ("Alba", "That should cover twelve participants if sessions stay under one hour.", 0.7),
    ("Raj", "I can schedule them for the second week of March.", 0.5),
    ("Maya", "Perfect. Let's meet again on Friday and compare notes.", 1.0),
]


def synthesize(voice, text, path):
    with wave.open(str(path), "wb") as stream:
        voice.synthesize_wav(text, stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("outputs/test_media"))
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Gaussian background noise amplitude, 0-1 of full scale")
    parser.add_argument("--voices", type=Path, default=Path("outputs/piper_voices"))
    args = parser.parse_args(argv)
    from piper import PiperVoice
    from piper.download_voices import download_voice

    args.output.mkdir(parents=True, exist_ok=True)
    args.voices.mkdir(parents=True, exist_ok=True)
    voices = {}
    for name, voice_id in VOICES.items():
        model = args.voices / f"{voice_id}.onnx"
        if not model.exists():
            download_voice(voice_id, args.voices)
        voices[name] = PiperVoice.load(str(model))

    samples = bytearray(b"\0\0" * int(RATE * 0.5))
    captions = []
    for i, (speaker, text, pause) in enumerate(SCRIPT, 1):
        part = args.output / f"_part{i}.wav"
        synthesize(voices[speaker], text, part)
        with wave.open(str(part)) as stream:
            if stream.getframerate() != RATE or stream.getnchannels() != 1:
                raise RuntimeError(f"Unexpected format from {VOICES[speaker]}")
            audio = stream.readframes(stream.getnframes())
        part.unlink()
        # Trim TTS leading/trailing silence so reference turn times follow audible speech.
        values = memoryview(audio).cast("h")
        loud = [n for n in range(len(values)) if abs(values[n]) > 300]
        if loud:
            audio = audio[2 * max(0, loud[0] - 220): 2 * min(len(values), loud[-1] + 220)]
        start = len(samples) / (2 * RATE)
        samples.extend(audio)
        end = len(samples) / (2 * RATE)
        captions.append(Caption(i, round(start, 3), round(end, 3), speaker, text, text))
        samples.extend(b"\0\0" * int(RATE * pause))

    if args.noise:
        rng = random.Random(0)
        values = memoryview(samples).cast("h")
        for n in range(len(values)):
            noisy = values[n] + int(rng.gauss(0, args.noise) * 32767)
            values[n] = max(-32768, min(32767, noisy))

    wav = args.output / "test_conversation.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(bytes(samples))
    video = args.output / "test_conversation.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(wav), "-filter_complex",
        "[0:a]showwaves=s=640x360:mode=cline:colors=0x0F766E,format=yuv420p[v]",
        "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "128k", "-shortest", str(video),
    ], check=True)
    duration = len(samples) / (2 * RATE)
    reference = Project(captions, speakers={k: k for k in VOICES}, metadata={
        "mode": "reference", "duration": duration, "noise": args.noise,
        "note": "Scripted Piper TTS conversation; utterance-level reference without word times.",
        "voices": VOICES,
    })
    (args.output / "reference.json").write_text(json.dumps(reference.to_dict(), indent=2),
                                                encoding="utf-8")
    print(f"Wrote {video} ({duration:.1f}s, {len(captions)} turns, 3 speakers) and reference.json")


if __name__ == "__main__":
    main()
