"""Render an animated four-person podcast studio for a synthetic test recording.

Illustrated presenters (clearly not real people) sit in a 2x2 grid with headphones and microphones.
Each mouth follows that speaker's own audio track, eyes blink, listeners look toward whoever is
talking, and the active speaker's tile is highlighted. Frames are drawn with Pillow at 2x and
downsampled for smooth edges, then piped to FFmpeg with the mixed audio.
"""
import math
from multiprocessing import get_context
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SS = 1280, 720, 25, 2   # output size, frame rate, supersampling factor
TITLE_H, MARGIN, GAP = 56, 16, 12
TILE_W = (W - 2 * MARGIN - GAP) // 2
TILE_H = (H - TITLE_H - 2 * MARGIN - GAP) // 2
FONT_DIR = "/usr/share/fonts/opentype/inter/"


def font(weight, size):
    try:
        return ImageFont.truetype(f"{FONT_DIR}Inter-{weight}.otf", size)
    except OSError:
        return ImageFont.load_default(size)


def tile_origin(index):
    col, row = index % 2, index // 2
    return MARGIN + col * (TILE_W + GAP), TITLE_H + MARGIN + row * (TILE_H + GAP)


def tile_center(index):
    x, y = tile_origin(index)
    return x + TILE_W / 2, y + TILE_H / 2


class Tile:
    """Static layers of one presenter's tile, drawn at SS scale."""

    def __init__(self, name, look):
        self.name, self.look = name, look
        w, h = TILE_W * SS, TILE_H * SS
        self.cx, self.cy = w * 0.42, h * 0.40          # head centre
        self.base = Image.new("RGBA", (w, h), look["bg"])
        d = ImageDraw.Draw(self.base)
        s, cx, cy = SS, self.cx, self.cy
        # Shoulders, neck, collar, then the desk in front.
        d.ellipse((cx - 118 * s, cy + 78 * s, cx + 118 * s, cy + 280 * s), fill=look["shirt"])
        d.rectangle((cx - 19 * s, cy + 50 * s, cx + 19 * s, cy + 92 * s), fill=look["skin"])
        d.polygon([(cx - 26 * s, cy + 86 * s), (cx, cy + 112 * s), (cx + 26 * s, cy + 86 * s)],
                  fill=look["skin"])
        d.rectangle((0, h - 58 * s, w, h), fill="#7c5a3a")
        d.rectangle((0, h - 58 * s, w, h - 52 * s), fill="#9a7450")
        # Overlay: microphone on a boom, name plate.
        self.overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        o = ImageDraw.Draw(self.overlay)
        mx, my = cx + 66 * s, cy + 86 * s
        o.line((cx + 150 * s, h - 58 * s, cx + 150 * s, my + 30 * s, mx + 10 * s, my + 30 * s),
               fill="#334155", width=6 * s, joint="curve")
        o.rounded_rectangle((mx - 14 * s, my - 26 * s, mx + 14 * s, my + 30 * s), 13 * s, fill="#1f2937")
        for k in range(5):
            yy = my - 18 * s + k * 9 * s
            o.line((mx - 9 * s, yy, mx + 9 * s, yy), fill="#4b5563", width=2 * s)
        title, role = font("Bold", 21 * s), font("Medium", 15 * s)
        px, py = 18 * s, h - 60 * s
        tw = max(o.textlength(name, font=title), o.textlength(look["role"], font=role)) + 28 * s
        o.rounded_rectangle((px, py - 4 * s, px + tw, py + 44 * s), 10 * s, fill=(15, 23, 42, 225))
        o.text((px + 14 * s, py), name, font=title, fill="#ffffff")
        o.text((px + 14 * s, py + 24 * s), look["role"], font=role, fill="#cbd5e1")

    def head(self, d, openness, blink, gaze, bob):
        s, look = SS, self.look
        cx, cy = self.cx, self.cy + bob * s
        rx, ry = 60 * s, 70 * s
        hair = look["hair"]
        style = look["style"]
        if style == "bob":      # hair behind the head
            d.rounded_rectangle((cx - rx - 12 * s, cy - ry - 10 * s, cx + rx + 12 * s, cy + 52 * s),
                                40 * s, fill=hair)
        if style == "bun":
            d.ellipse((cx - 4 * s, cy - ry - 44 * s, cx + 46 * s, cy - ry + 4 * s), fill=hair)
        d.ellipse((cx - rx - 9 * s, cy - 6 * s, cx - rx + 9 * s, cy + 22 * s), fill=look["skin"])
        d.ellipse((cx + rx - 9 * s, cy - 6 * s, cx + rx + 9 * s, cy + 22 * s), fill=look["skin"])
        d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=look["skin"])
        # Front hair.
        if style == "bob":
            d.chord((cx - rx - 6 * s, cy - ry - 14 * s, cx + rx + 6 * s, cy + 20 * s), 180, 360, fill=hair)
            d.polygon([(cx - 8 * s, cy - ry - 6 * s), (cx + rx + 4 * s, cy - 14 * s),
                       (cx + rx + 6 * s, cy - ry + 10 * s)], fill=hair)
        elif style == "short":
            d.chord((cx - rx - 3 * s, cy - ry - 8 * s, cx + rx + 3 * s, cy + 6 * s), 190, 350, fill=hair)
        elif style == "bun":
            d.chord((cx - rx - 4 * s, cy - ry - 10 * s, cx + rx + 4 * s, cy + 14 * s), 185, 355, fill=hair)
        elif style == "wavy":
            for k in range(9):
                a = math.radians(200 + k * 17.5)
                hx, hy = cx + (rx + 2 * s) * math.cos(a), cy - 8 * s + (ry + 4 * s) * math.sin(a)
                d.ellipse((hx - 19 * s, hy - 19 * s, hx + 19 * s, hy + 19 * s), fill=hair)
            d.chord((cx - rx, cy - ry - 4 * s, cx + rx, cy + 4 * s), 185, 355, fill=hair)
        if look.get("beard"):
            d.chord((cx - rx + 4 * s, cy - 6 * s, cx + rx - 4 * s, cy + ry + 6 * s), 10, 170,
                    fill=look["beard"])
            d.ellipse((cx - 30 * s, cy + 14 * s, cx + 30 * s, cy + 50 * s), fill=look["skin"])
        # Eyes, pupils following gaze, blinking.
        gx, gy = gaze
        brow = 3 * s * openness
        for side in (-1, 1):
            ex, ey = cx + side * 23 * s, cy - 6 * s
            if blink:
                d.line((ex - 10 * s, ey, ex + 10 * s, ey), fill="#1f2937", width=3 * s)
            else:
                d.ellipse((ex - 11 * s, ey - 8 * s, ex + 11 * s, ey + 8 * s), fill="#ffffff")
                px, py = ex + gx * 5 * s, ey + gy * 3 * s
                d.ellipse((px - 5 * s, py - 5 * s, px + 5 * s, py + 5 * s), fill="#1f2937")
            d.line((ex - 12 * s, ey - 17 * s - brow, ex + 11 * s, ey - 19 * s - brow),
                   fill=look.get("brow", hair), width=4 * s)
        if look.get("glasses"):
            for side in (-1, 1):
                ex = cx + side * 23 * s
                d.rounded_rectangle((ex - 17 * s, cy - 18 * s, ex + 17 * s, cy + 8 * s), 7 * s,
                                    outline="#111827", width=3 * s)
            d.line((cx - 6 * s, cy - 8 * s, cx + 6 * s, cy - 8 * s), fill="#111827", width=3 * s)
        d.line((cx, cy + 2 * s, cx - 4 * s, cy + 16 * s, cx + 3 * s, cy + 17 * s),
               fill=look["shade"], width=3 * s)
        # Mouth opens with the speaker's own audio level.
        my = cy + 34 * s
        if openness < 0.08:
            d.arc((cx - 16 * s, my - 12 * s, cx + 16 * s, my + 6 * s), 25, 155, fill="#7f1d1d", width=3 * s)
        else:
            mh, mw = (4 + 20 * openness) * s, (24 - 5 * openness) * s
            d.ellipse((cx - mw / 2, my - mh / 2, cx + mw / 2, my + mh / 2), fill="#4c0519")
            if openness > 0.3:
                d.rectangle((cx - mw / 2 + 4 * s, my - mh / 2 + 1 * s, cx + mw / 2 - 4 * s, my - mh / 2 + 4 * s),
                            fill="#f8fafc")
                d.ellipse((cx - 7 * s, my + mh / 2 - 7 * s, cx + 7 * s, my + mh / 2), fill="#f472b6")
        # Headphones over the hair.
        band = (cx - rx - 12 * s, cy - ry - 18 * s, cx + rx + 12 * s, cy + ry - 10 * s)
        d.arc(band, 196, 344, fill="#111827", width=9 * s)
        for side in (-1, 1):
            hx = cx + side * (rx + 4 * s)
            d.rounded_rectangle((hx - 13 * s, cy - 16 * s, hx + 13 * s, cy + 30 * s), 9 * s, fill="#111827")
            d.rounded_rectangle((hx - 7 * s, cy - 10 * s, hx + 7 * s, cy + 24 * s), 6 * s, fill=look["accent"])

    def render(self, openness, blink, gaze, bob, speaking, level):
        img = self.base.copy()
        d = ImageDraw.Draw(img)
        self.head(d, openness, blink, gaze, bob)
        img.alpha_composite(self.overlay)
        if speaking:
            s, w = SS, img.width
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((3 * s, 3 * s, w - 3 * s, img.height - 3 * s), 16 * s,
                                outline="#16a34a", width=6 * s)
            for k in range(4):      # small level meter
                bh = (6 + 22 * level * (0.6 + 0.4 * math.sin(k * 1.7 + level * 9))) * s
                bx = w - 70 * s + k * 12 * s
                d.rounded_rectangle((bx, 34 * s - bh / 2, bx + 7 * s, 34 * s + bh / 2), 3 * s, fill="#16a34a")
        return img.resize((TILE_W, TILE_H), Image.LANCZOS)


_STATE = {}


def _frame(i):
    st = _STATE
    frame = st["background"].copy()
    names = st["names"]
    active = [k for k, n in enumerate(names) if st["speaking"][n][i]]
    t = i / FPS
    for k, name in enumerate(names):
        level = float(st["open"][name][i])
        speaking = bool(st["speaking"][name][i])
        others = [a for a in active if a != k]
        if others:   # look at whoever else is talking
            tx, ty = tile_center(others[0])
            ox, oy = tile_center(k)
            dist = math.hypot(tx - ox, ty - oy) or 1
            gaze = ((tx - ox) / dist, (ty - oy) / dist)
        else:
            gaze = (0.35 * math.sin(t * 0.7 + k), 0.2 * math.sin(t * 0.5 + 2 * k))
        bob = -3 * level + 1.5 * level * math.sin(t * 9 + k) if speaking else 1.2 * math.sin(t * 0.8 + k * 1.3)
        tile = st["tiles"][name].render(level if speaking else 0.0, bool(st["blink"][name][i]),
                                         gaze, bob, speaking, level)
        frame.paste(tile.convert("RGB"), tile_origin(k))
    return frame.tobytes()


def _background(title, subtitle):
    img = Image.new("RGB", (W, H), "#e2e8f0")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, TITLE_H), fill="#0f172a")
    d.ellipse((MARGIN, 18, MARGIN + 20, 38), fill="#ef4444")
    d.text((MARGIN + 30, 14), title, font=font("Bold", 22), fill="#ffffff")
    d.text((W - MARGIN - d.textlength(subtitle, font=font("Medium", 15)), 20), subtitle,
           font=font("Medium", 15), fill="#94a3b8")
    return img


def envelopes(tracks, rate, frames):
    """Per-speaker mouth openness per frame from each speaker's own audio track."""
    hop = rate // FPS
    out = {}
    for name, track in tracks.items():
        rms = np.array([np.sqrt(np.mean(track[f * hop:(f + 1) * hop] ** 2)) if f * hop < len(track) else 0.0
                        for f in range(frames)])
        loud = rms[rms > 200]
        ref = np.percentile(loud, 90) if len(loud) else 1.0
        level = np.clip(rms / ref, 0, 1) ** 0.8
        smooth = np.zeros(frames)
        for f in range(frames):
            smooth[f] = max(level[f], smooth[f - 1] * 0.55 if f else 0.0)
        out[name] = smooth
    return out


def render(wav_path, out_path, cast, lines, tracks, rate, duration, title, subtitle):
    """cast: {name: look}; lines: [(name, start_s, end_s)]; tracks: {name: float32 samples}."""
    frames = int(math.ceil(duration * FPS))
    names = list(cast)
    speaking = {n: np.zeros(frames, bool) for n in names}
    for name, start, end in lines:
        speaking[name][int(start * FPS):int(math.ceil(end * FPS))] = True
    rng = np.random.default_rng(7)
    blink = {}
    for n in names:
        b = np.zeros(frames, bool)
        t = rng.uniform(0.5, 3)
        while t < duration:
            b[int(t * FPS):int(t * FPS) + 3] = True
            t += rng.uniform(2.5, 5.5)
        blink[n] = b
    _STATE.update(names=names, speaking=speaking, blink=blink, open=envelopes(tracks, rate, frames),
                  tiles={n: Tile(n, cast[n]) for n in names}, background=_background(title, subtitle))
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(FPS), "-i", "-", "-i", str(wav_path), "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "medium", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", str(out_path),
    ], stdin=subprocess.PIPE)
    with get_context("fork").Pool() as pool:
        for data in pool.imap(_frame, range(frames), chunksize=16):
            encoder.stdin.write(data)
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("FFmpeg failed to encode the podcast video")
