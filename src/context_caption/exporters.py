"""Portable subtitle exports use text speaker names; VTT also includes color classes.

Sound cues are separate cues with no speaker prefix, in SDH brackets: [dog barking].
"""
import html
import json
import textwrap

from .sounds import format_sound


def timestamp(seconds, separator="."):
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def subtitle_text(caption, project, emotions=False):
    name = " ".join(project.speakers[caption.speaker].split())
    emotion = f" [{caption.emotion}]" if emotions and caption.emotion else ""
    text = " ".join(caption.text.split())
    return f"{name}{emotion}: {text}"


def timeline(project, sounds=True):
    """Speech and sound cues in one playback order; a sound sorts before speech it starts with."""
    cues = [(c.start, 1, c.id, c) for c in project.captions]
    if sounds:
        cues += [(s.start, 0, s.id, s) for s in project.sounds]
    return [(kind == 0, cue) for _, kind, _, cue in sorted(cues, key=lambda cue: cue[:3])]


def to_srt(project, emotions=False, sounds=True):
    blocks = []
    for i, (is_sound, c) in enumerate(timeline(project, sounds), 1):
        line = format_sound(c) if is_sound else subtitle_text(c, project, emotions)
        text = "\n".join(textwrap.wrap(line, 60))
        blocks.append(f"{i}\n{timestamp(c.start, ',')} --> {timestamp(c.end, ',')}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(project, emotions=False, sounds=True):
    classes = {key: f"speaker{i}" for i, key in enumerate(project.speakers)}
    colors = project.colors()
    styles = "\n".join(
        f"::cue(.{classes[key]}) {{ color: {color}; }}" for key, color in colors.items()
    )
    blocks = ["WEBVTT", "STYLE\n" + styles]
    for is_sound, c in timeline(project, sounds):
        times = f"{timestamp(c.start)} --> {timestamp(c.end)}"
        if is_sound:
            blocks.append(f"sound-{c.id}\n{times}\n{html.escape(format_sound(c))}")
            continue
        text = html.escape(subtitle_text(c, project, emotions))
        blocks.append(f"{c.id}\n{times}\n<c.{classes[c.speaker]}>{text}</c>")
    return "\n\n".join(blocks) + "\n"


def to_json(project):
    return json.dumps(project.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
