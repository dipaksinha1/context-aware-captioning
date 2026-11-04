"""Generate a synthetic multi-speaker test video plus its reference project for evaluation.

Uses Piper neural text-to-speech voices (downloaded from Hugging Face on first use) and FFmpeg.
The reference holds the scripted text, speaker, and utterance timing for every turn; it has no word
timestamps, so word-timing error is not measurable from it. Synthetic voices are cleaner than real
conversation, so scores from these files are a pipeline check, not an accuracy claim.

    pip install piper-tts
    python scripts/make_test_recording.py [--script conversation|podcast] [--output outputs/test_media]

Scripts:
  conversation  41 s, three voices: quick replies, a returning speaker, numbers, proper nouns.
  podcast       about 100 s, four voices debating a fictional city's congestion-charge policy, with
                two short moments of crosstalk (overlapping speech).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from context_caption.schema import Caption, Project  # noqa: E402

RATE = 22050

# Each line: (speaker, spoken text, gap after this line in seconds, reference text or None).
# A negative gap starts the next line that many seconds before this one ends (crosstalk).
# Reference text differs from spoken text only where speech is written differently ("$40 million").
SCRIPTS = {
    "conversation": {
        "file": "test_conversation",
        "voices": {"Maya": "en_US-lessac-medium", "Raj": "en_US-ryan-medium",
                   "Alba": "en_GB-alba-medium"},
        "lines": [
            ("Maya", "Good morning everyone, thanks for joining the accessibility review.", 0.7, None),
            ("Raj", "Happy to be here. I tested the caption editor on Monday.", 0.6, None),
            ("Maya", "Great. What did you find?", 0.5, None),
            ("Raj", "Speaker labels stayed consistent, but long sentences were split too early.", 0.8, None),
            ("Alba", "I noticed that as well, especially in the lecture recording from Toronto.", 0.6, None),
            ("Maya", "Was the timing accurate?", 0.3, None),
            ("Alba", "Mostly. Each caption appeared within a fraction of a second.", 0.7, None),
            ("Raj", "Yes.", 0.3, None),
            ("Maya", "Good. The budget for the next study is about 300 dollars.", 0.6, None),
            ("Alba", "That should cover twelve participants if sessions stay under one hour.", 0.7, None),
            ("Raj", "I can schedule them for the second week of March.", 0.5, None),
            ("Maya", "Perfect. Let's meet again on Friday and compare notes.", 1.0, None),
        ],
    },
    # A fictional city and policy; viewpoints are deliberately balanced.
    "podcast": {
        "file": "podcast_policy",
        "title": "POLICY ROUNDTABLE",
        "subtitle": "Synthetic podcast · AI voices · illustrated presenters · fictional city",
        # Illustrated presenters for the animated studio video (scripts/podcast_scene.py).
        "cast": {
            "Dana": {"role": "Host", "skin": "#f1c27d", "shade": "#c8955a", "hair": "#3b2416",
                     "style": "bob", "shirt": "#0f766e", "bg": "#d7eeeb", "accent": "#14b8a6"},
            "Marcus": {"role": "Former city budget officer", "skin": "#8d5524", "shade": "#6b3f1a",
                       "hair": "#111111", "style": "short", "shirt": "#1e3a8a", "bg": "#dfe5f8",
                       "accent": "#3b82f6", "glasses": True},
            "Elena": {"role": "Bakery owner, Main Street", "skin": "#ffdbac", "shade": "#e0b48a",
                      "hair": "#9a3412", "style": "bun", "shirt": "#be185d", "bg": "#f8e1ec",
                      "accent": "#f472b6"},
            "Tom": {"role": "Transport researcher", "skin": "#e0ac69", "shade": "#b9844a",
                    "hair": "#7c5a2e", "style": "wavy", "shirt": "#4d7c0f", "bg": "#e6f2d3",
                    "accent": "#84cc16", "beard": "#7c5a2e"},
        },
        "voices": {"Dana": "en_US-amy-medium", "Marcus": "en_US-ryan-medium",
                   "Elena": "en_GB-alba-medium", "Tom": "en_GB-alan-medium"},
        "lines": [
            ("Dana", "Welcome back to Policy Roundtable. I'm Dana, and today we're debating "
                     "Riverton's plan to charge drivers entering downtown and use the money for free buses.",
             0.6, None),
            ("Marcus", "Thanks, Dana. I'm Marcus, and I spent a decade in the city budget office.", 0.5, None),
            ("Elena", "Hi everyone, I'm Elena. I run a bakery on Main Street, so this affects me directly.",
             0.5, None),
            ("Tom", "And I'm Tom. I study public transport at the university.", 0.7, None),
            ("Dana", "Marcus, let's start with the numbers. What would the charge raise?", 0.5, None),
            ("Marcus", "The council estimates forty million dollars a year, at eight dollars per car per day.",
             0.4, "The council estimates $40 million a year, at $8 per car per day."),
            ("Tom", "That's roughly enough to make every bus route free and add night service.", 0.6, None),
            ("Elena", "Honestly, I'm worried. If customers stop driving in, small shops like mine "
                      "could close within a year.", -0.45, None),
            ("Dana", "Right.", 0.5, None),
            ("Dana", "That's a real fear. Tom, what do the studies say?", 0.5, None),
            ("Tom", "The studies the council cites show traffic falling by about a fifth, "
                    "but the effect on shops is mixed.", 0.4, None),
            ("Elena", "Mixed isn't good enough when it's your livelihood on the line.", 0.5, None),
            ("Marcus", "I hear you. That's why the plan includes a discount for delivery vans "
                       "in the first year.", 0.4, None),
            ("Elena", "Really? I didn't know that. That actually helps.", 0.7, None),
            ("Dana", "Let's talk about fairness. Who pays the most?", 0.4, None),
            ("Tom", "Commuters from the outer suburbs with no train line. For them, it's a genuine burden.",
             -0.4, None),
            ("Marcus", "Exactly.", 0.4, None),
            ("Marcus", "So the council added exemptions for low income households and people with "
                       "disabilities.", 0.5, None),
            ("Elena", "It's sad that we only think about fairness after the backlash.", 0.7, None),
            ("Dana", "Quick round to finish. Would you vote yes? Marcus?", 0.4, None),
            ("Marcus", "Yes, with an independent review after a couple of years.", 0.4, None),
            ("Elena", "Maybe. Show me the delivery rules in writing first.", 0.4, None),
            ("Tom", "Yes. Free buses would change this city for the better.", 0.6, None),
            ("Dana", "Thank you all. That's Policy Roundtable. Stay curious, and we'll see you next week.",
             1.0, None),
        ],
    },
}


def synthesize(voice, text):
    """Mono int16 samples with leading/trailing TTS silence trimmed to about 10 ms."""
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as stream:
            voice.synthesize_wav(text, stream)
        with wave.open(tmp.name) as stream:
            if stream.getframerate() != RATE or stream.getnchannels() != 1:
                raise RuntimeError("Unexpected Piper output format")
            audio = np.frombuffer(stream.readframes(stream.getnframes()), np.int16)
    loud = np.flatnonzero(np.abs(audio.astype(np.int32)) > 300)
    if len(loud):
        audio = audio[max(0, loud[0] - 220): loud[-1] + 220]
    return audio.astype(np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--script", choices=sorted(SCRIPTS), default="conversation")
    parser.add_argument("--output", type=Path, default=Path("outputs/test_media"))
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Gaussian background noise amplitude, 0-1 of full scale")
    parser.add_argument("--voices", type=Path, default=Path("outputs/piper_voices"))
    args = parser.parse_args(argv)
    from piper import PiperVoice
    from piper.download_voices import download_voice

    spec = SCRIPTS[args.script]
    args.output.mkdir(parents=True, exist_ok=True)
    args.voices.mkdir(parents=True, exist_ok=True)
    voices = {}
    for name, voice_id in spec["voices"].items():
        model = args.voices / f"{voice_id}.onnx"
        if not model.exists():
            download_voice(voice_id, args.voices)
        voices[name] = PiperVoice.load(str(model))

    placed, captions, cursor = [], [], 0.5
    for i, (speaker, text, gap, reference) in enumerate(spec["lines"], 1):
        audio = synthesize(voices[speaker], text)
        start, end = cursor, cursor + len(audio) / RATE
        placed.append((speaker, int(round(start * RATE)), audio))
        captions.append(Caption(i, round(start, 3), round(end, 3), speaker, reference or text,
                                reference or text))
        cursor = end + gap
    total = max(offset + len(audio) for _, offset, audio in placed) + RATE
    mix = np.zeros(total, np.float32)
    tracks = {name: np.zeros(total, np.float32) for name in spec["voices"]}
    for speaker, offset, audio in placed:
        mix[offset:offset + len(audio)] += audio  # overlapping lines are summed
        tracks[speaker][offset:offset + len(audio)] += audio
    if args.noise:
        mix += np.random.default_rng(0).normal(0, args.noise * 32767, total).astype(np.float32)
    samples = np.clip(mix, -32768, 32767).astype(np.int16)

    base = args.output / spec["file"]
    wav = base.with_suffix(".wav")
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(samples.tobytes())
    duration = total / RATE
    if spec.get("cast"):
        from podcast_scene import render
        render(wav, base.with_suffix(".mp4"), spec["cast"],
               [(c.speaker, c.start, c.end) for c in captions], tracks, RATE, duration,
               spec["title"], spec["subtitle"])
    else:
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", str(wav), "-filter_complex",
            "[0:a]showwaves=s=640x360:mode=cline:rate=15:colors=0x0F766E,format=yuv420p[v]",
            "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
            "-c:a", "aac", "-b:a", "64k", "-shortest", str(base.with_suffix(".mp4")),
        ], check=True)
    overlaps = sum(1 for _, _, gap, _ in spec["lines"] if gap < 0)
    reference = Project(captions, speakers={k: k for k in spec["voices"]}, metadata={
        "mode": "reference", "duration": duration, "noise": args.noise, "voices": spec["voices"],
        "note": "Scripted Piper TTS recording; utterance-level reference without word times.",
        "overlapping_lines": overlaps,
    })
    ref_path = args.output / f"{spec['file']}_reference.json"
    ref_path.write_text(json.dumps(reference.to_dict(), indent=2), encoding="utf-8")
    print(f"Wrote {base.with_suffix('.mp4')} ({duration:.1f}s, {len(captions)} lines, "
          f"{len(spec['voices'])} speakers, {overlaps} overlaps) and {ref_path.name}")


if __name__ == "__main__":
    main()
