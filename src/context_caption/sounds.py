"""Non-speech sound cues written the SDH way: bracketed, lowercase, present tense — [dog barking].

An AudioSet tagger scores half-overlapping windows. Only the curated classes below become cues;
speech classes are deliberately absent because speech is already captioned.
"""
from copy import deepcopy
import math

from .schema import Sound

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
RATE = 16000
WINDOW = 2.0  # seconds; the hop is half a window so neighbours overlap by half
THRESHOLD = 0.3
SILENCE_RMS = 0.001  # about -60 dBFS; quieter windows are skipped, not classified

# Caption wording -> AudioSet classes that trigger it. Sustained sounds take the present
# participle ("dog barking"), abrupt ones the third-person verb ("door slams").
VOCABULARY = {
    # People, non-verbal
    "laughter": ["Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh",
                 "Chuckle, chortle"],
    "crying": ["Crying, sobbing", "Whimper", "Wail, moan"],
    "baby crying": ["Baby cry, infant cry"],
    "screaming": ["Screaming"],
    "coughing": ["Cough"],
    "sneezes": ["Sneeze"],
    "snoring": ["Snoring"],
    "whistling": ["Whistling"],
    "singing": ["Singing", "Choir", "Male singing", "Female singing", "Child singing"],
    "applause": ["Applause", "Clapping"],
    "crowd cheering": ["Cheering"],
    "footsteps": ["Walk, footsteps"],
    # Animals
    "dog barking": ["Dog", "Bark", "Bow-wow", "Yip"],
    "dog howling": ["Howl"],
    "dog growling": ["Growling"],
    "cat meowing": ["Cat", "Meow", "Caterwaul"],
    "cat purring": ["Purr"],
    "birds chirping": ["Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet"],
    "crow cawing": ["Crow", "Caw"],
    "owl hooting": ["Owl", "Hoot"],
    "rooster crowing": ["Crowing, cock-a-doodle-doo"],
    "duck quacking": ["Duck", "Quack"],
    "horse neighing": ["Neigh, whinny"],
    "hooves clopping": ["Clip-clop"],
    "cow mooing": ["Cattle, bovinae", "Moo"],
    "sheep bleating": ["Sheep", "Bleat"],
    "pig oinking": ["Pig", "Oink"],
    "crickets chirping": ["Cricket"],
    "insects buzzing": ["Mosquito", "Fly, housefly", "Bee, wasp, etc."],
    "frogs croaking": ["Frog", "Croak"],
    # Music
    "music playing": ["Music"],
    # Weather and nature
    "rain falling": ["Rain", "Raindrop", "Rain on surface"],
    "thunder rumbling": ["Thunder", "Thunderstorm"],
    "wind blowing": ["Wind"],
    "waves crashing": ["Ocean", "Waves, surf"],
    "water running": ["Stream", "Waterfall", "Water tap, faucet", "Sink (filling or washing)",
                      "Bathtub (filling or washing)"],
    "water splashing": ["Splash, splatter"],
    "water dripping": ["Drip"],
    "fire crackling": ["Fire", "Crackle"],
    # Home
    "door slams": ["Slam"],
    "knocking": ["Knock"],
    "doorbell rings": ["Doorbell", "Ding-dong"],
    "door creaking": ["Creak"],
    "glass shatters": ["Shatter"],
    "glasses clinking": ["Chink, clink"],
    "dishes clattering": ["Dishes, pots, and pans", "Cutlery, silverware"],
    "food sizzling": ["Frying (food)", "Sizzle"],
    "phone ringing": ["Telephone bell ringing", "Ringtone"],
    "alarm blaring": ["Alarm clock", "Smoke detector, smoke alarm", "Fire alarm", "Car alarm"],
    "buzzer sounds": ["Buzzer"],
    "beeping": ["Beep, bleep", "Reversing beeps"],
    "bell ringing": ["Bell", "Church bell", "Jingle bell", "Bicycle bell"],
    "clock ticking": ["Tick", "Tick-tock"],
    "keyboard typing": ["Typing", "Computer keyboard", "Typewriter"],
    "keys jangling": ["Keys jangling"],
    "camera shutter clicks": ["Camera", "Single-lens reflex camera"],
    "vacuum cleaner running": ["Vacuum cleaner"],
    "toilet flushes": ["Toilet flush"],
    # Vehicles
    "car horn honking": ["Vehicle horn, car horn, honking", "Toot", "Air horn, truck horn"],
    "siren wailing": ["Siren", "Police car (siren)", "Ambulance (siren)",
                      "Fire engine, fire truck (siren)", "Civil defense siren"],
    "engine running": ["Engine", "Idling", "Light engine (high frequency)",
                       "Medium engine (mid frequency)", "Heavy engine (low frequency)"],
    "engine revving": ["Accelerating, revving, vroom"],
    "engine starts": ["Engine starting"],
    "tires squealing": ["Tire squeal", "Skidding"],
    "traffic passing": ["Traffic noise, roadway noise", "Car passing by"],
    "motorcycle revving": ["Motorcycle"],
    "train passing": ["Train", "Railroad car, train wagon"],
    "train horn blowing": ["Train horn", "Train whistle"],
    "aircraft overhead": ["Aircraft", "Fixed-wing aircraft, airplane", "Jet engine"],
    "helicopter whirring": ["Helicopter"],
    # Impacts and tools
    "gunfire": ["Gunshot, gunfire", "Machine gun", "Fusillade"],
    "explosion": ["Explosion"],
    "fireworks popping": ["Fireworks", "Firecracker"],
    "crash": ["Smash, crash"],
    "thud": ["Thump, thud"],
    "hammering": ["Hammer"],
    "drilling": ["Drill", "Jackhammer"],
    "sawing": ["Sawing"],
    "chainsaw buzzing": ["Chainsaw"],
    "lawn mower running": ["Lawn mower"],
}


def clean_text(text):
    """Bare description: brackets are presentation, added once by format_sound."""
    return " ".join(str(text).split()).strip("[]() ")


def format_sound(sound):
    return f"[{clean_text(sound.text)}]"


def build_sounds(frames, threshold=THRESHOLD, top=2, merge_gap=3.0,
                 min_duration=1.5, max_duration=6.0):
    """Turn window scores into cues.

    frames: time-ordered (start, end, {text: score}) windows, each overlapping the next by half.
    A cue marks where a sound begins, so long sounds are shown for max_duration, not throughout.
    """
    if not frames:
        return []
    duration = max(end for _, end, _ in frames)
    runs = {}  # text -> runs of consecutive detecting windows
    for start, end, scores in frames:
        hits = sorted(((s, t) for t, s in scores.items() if s >= threshold), reverse=True)[:top]
        for score, text in hits:
            history = runs.setdefault(text, [])
            if history and start < history[-1][-1][1]:
                history[-1].append((start, end, score))
            else:
                history.append([(start, end, score)])
    found = []
    for text, history in runs.items():
        spans = []
        for run in history:
            # A sound heard by overlapping windows lies in their shared part, not their union.
            start, end = (run[1][0], run[-2][1]) if len(run) > 1 else run[0][:2]
            score = max(s for _, _, s in run)
            if spans and start - spans[-1][1] <= merge_gap:
                spans[-1] = (spans[-1][0], max(end, spans[-1][1]), max(score, spans[-1][2]))
            else:
                spans.append((start, end, score))
        for start, end, score in spans:
            end = min(max(end, start + min_duration), start + max_duration, duration)
            if end > start:
                found.append((start, end, text, score))
    return [Sound(i, start, end, text, text, round(score, 3))
            for i, (start, end, text, score) in enumerate(sorted(found), 1)]


def score_windows(audio, device="cpu", batch_size=8):
    """Score each window against the vocabulary: (start, end, {text: score}) per window."""
    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    extractor = AutoFeatureExtractor.from_pretrained(MODEL)
    model = AutoModelForAudioClassification.from_pretrained(MODEL).to(device).eval()
    index = {name: int(i) for i, name in model.config.id2label.items()}
    groups = {text: [index[name] for name in names] for text, names in VOCABULARY.items()}
    size = int(WINDOW * RATE)
    hop = size // 2
    windows = []
    for offset in range(0, max(len(audio) - hop, 1), hop):
        chunk = np.asarray(audio[offset:offset + size], dtype=np.float32)
        if len(chunk) and float(np.sqrt(np.mean(chunk ** 2))) >= SILENCE_RMS:
            windows.append((offset / RATE, (offset + len(chunk)) / RATE, chunk))
    frames = []
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        inputs = extractor([chunk for _, _, chunk in batch], sampling_rate=RATE,
                           return_tensors="pt")
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device)).logits
        # AudioSet is multi-label: independent sigmoids, not a softmax over classes.
        scores = torch.sigmoid(logits).cpu().numpy()
        for (start, end, _), row in zip(batch, scores):
            frames.append((start, end, {text: float(row[ids].max())
                                        for text, ids in groups.items()}))
    return frames


def detect_sounds(audio, device="cpu", threshold=THRESHOLD):
    """audio: mono float waveform at 16 kHz, the same array the transcriber uses."""
    return build_sounds(score_windows(audio, device), threshold)


def _blank(value):
    return (value is None or (isinstance(value, float) and math.isnan(value))
            or (isinstance(value, str) and not value.strip()))


def apply_sound_edits(project, rows):
    """Atomically replace the sound cues. Rows without a known id are new manual cues."""
    known = {s.id: s for s in project.sounds}
    next_id = max(known, default=0) + 1
    sounds = []
    for row in rows:
        fields = [row.get(key) for key in ("start", "end", "text")]
        if all(_blank(value) for value in fields):
            continue
        if any(_blank(value) for value in fields):
            raise ValueError("Each sound cue needs a start, an end, and a description.")
        old = None if _blank(row.get("id")) else known.pop(int(row["id"]), None)
        if old is None:
            identifier, next_id = next_id, next_id + 1
        else:
            identifier = old.id
        text = clean_text(row["text"])
        sounds.append(Sound(identifier, float(row["start"]), float(row["end"]), text,
                            old.original_text if old else text, old.score if old else None))
    edited = deepcopy(project)
    edited.sounds = sorted(sounds, key=lambda s: (s.start, s.id))
    return edited
