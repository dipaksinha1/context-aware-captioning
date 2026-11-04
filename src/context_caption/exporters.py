"""Portable subtitle exports use text speaker names; VTT also includes color classes."""
import html
import json
import textwrap

from .schema import emotion_label


def timestamp(seconds, separator="."):
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{separator}{millis:03}"


def subtitle_text(caption, project, emotions=False):
    name = " ".join(project.speakers[caption.speaker].split())
    emotion = f" [{emotion_label(caption.emotion)}]" if emotions and caption.emotion else ""
    text = " ".join(caption.text.split())
    return f"{name}{emotion}: {text}"


def to_srt(project, emotions=False):
    blocks = []
    for i, c in enumerate(sorted(project.captions, key=lambda c: (c.start, c.id)), 1):
        text = "\n".join(textwrap.wrap(subtitle_text(c, project, emotions), 60))
        blocks.append(f"{i}\n{timestamp(c.start, ',')} --> {timestamp(c.end, ',')}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(project, emotions=False):
    classes = {key: f"speaker{i}" for i, key in enumerate(project.speakers)}
    colors = project.colors()
    styles = "\n".join(
        f"::cue(.{classes[key]}) {{ color: {color}; }}" for key, color in colors.items()
    )
    blocks = ["WEBVTT", "STYLE\n" + styles]
    for c in sorted(project.captions, key=lambda c: (c.start, c.id)):
        text = html.escape(subtitle_text(c, project, emotions))
        blocks.append(
            f"{c.id}\n{timestamp(c.start)} --> {timestamp(c.end)}\n"
            f"<c.{classes[c.speaker]}>{text}</c>"
        )
    return "\n\n".join(blocks) + "\n"


def to_json(project):
    return json.dumps(project.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
