"""Assemble the captured clips into the two showcase videos (title cards, speed-ups, web-sized H.264)."""

import glob
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

D = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(D, "clips")
SEG = os.path.join(D, "segments")
OUT = os.path.join(D, "final")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FPS = 25
BG = (16, 22, 28)
ACCENT = (118, 185, 0)  # Isaac green-ish
SUB = (170, 182, 190)


def sh(cmd):
    subprocess.run(cmd, check=True)


def card(name, title, lines=(), kicker="", seconds=3.0):
    """Render a title card as an mp4 segment."""
    img = Image.new("RGB", (1280, 720), BG)
    d = ImageDraw.Draw(img)
    y = 250 - 22 * len(lines)
    if kicker:
        f = ImageFont.truetype(FONT_BOLD, 22)
        d.text((110, y - 46), kicker.upper(), font=f, fill=ACCENT)
    d.rectangle([110, y - 10, 116, y + 64], fill=ACCENT)
    d.text((136, y - 6), title, font=ImageFont.truetype(FONT_BOLD, 52), fill=(240, 244, 246))
    f = ImageFont.truetype(FONT, 28)
    for i, line in enumerate(lines):
        d.text((136, y + 96 + 46 * i), line, font=f, fill=SUB)
    png = os.path.join(SEG, f"{name}.png")
    img.save(png)
    out = os.path.join(SEG, f"{name}.mp4")
    sh(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", png, "-t", str(seconds), "-r", str(FPS),
        "-vf", "fade=in:st=0:d=0.4,fade=out:st=%.2f:d=0.4" % (seconds - 0.4),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out])
    return out


def demo_counter_filters(count):
    """Filters that redraw the bottom-right "recorded demos" HUD line of an expert clip with ``count``."""
    text = f"recorded demos: {count}"
    tw = int(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(text, font=ImageFont.truetype(FONT, 22)))
    widest = int(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength("recorded demos: 00", font=ImageFont.truetype(FONT, 22)))
    return [f"drawbox=x={1280 - 34 - widest}:y=660:w={widest + 22}:h=40:color=0x161b20@1:t=fill",
            f"drawtext=fontfile={FONT}:text='{text.replace(':', chr(92) + ':')}':x={1280 - 22 - tw}:y=666:fontsize=22:fontcolor=white"]


def segment(name, src, start=None, end=None, speed=1.0, fade=True, demos=None):
    """Cut [start, end) frames of ``src``, play it ``speed`` times faster, add short fades.

    ``demos`` redraws the clip's "recorded demos" HUD counter with that number."""
    out = os.path.join(SEG, f"{name}.mp4")
    filters = demo_counter_filters(demos) if demos is not None else []
    if start is not None or end is not None:
        trim = "trim="
        parts = []
        if start is not None:
            parts.append(f"start_frame={start}")
        if end is not None:
            parts.append(f"end_frame={end}")
        filters.append(trim + ":".join(parts))
    filters.append(f"setpts=(PTS-STARTPTS)/{speed}")
    filters.append(f"fps={FPS}")
    if fade:
        dur = probe_frames(src, start, end) / FPS / speed
        filters.append("fade=in:st=0:d=0.3")
        filters.append("fade=out:st=%.2f:d=0.3" % max(dur - 0.3, 0))
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", ",".join(filters), "-an",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS), out])
    return out


def badge_hold(name, src, frame_index, success, demo_index=None, seconds=1.6):
    """Freeze the last action frame of an episode and stamp the outcome on it."""
    png = os.path.join(SEG, f"{name}.png")
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", f"select=eq(n\\,{frame_index})", "-frames:v", "1", png])
    img = Image.open(png).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    text = (f"SUCCESS  ->  saved demo_{demo_index} to HDF5" if demo_index is not None else "SUCCESS  ->  saved to HDF5") \
        if success else "FAILED  ->  attempt discarded"
    color = (20, 140, 60, 225) if success else (170, 40, 40, 225)
    f = ImageFont.truetype(FONT_BOLD, 34)
    W, H = img.size
    tw = d.textlength(text, font=f)
    d.rounded_rectangle([W / 2 - tw / 2 - 22, H / 2 - 36, W / 2 + tw / 2 + 22, H / 2 + 30], 12, fill=color)
    d.text((W / 2 - tw / 2, H / 2 - 28), text, font=f, fill=(255, 255, 255))
    img.save(png)
    out = os.path.join(SEG, f"{name}.mp4")
    sh(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", png, "-t", str(seconds), "-r", str(FPS),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out])
    return out


def orbit(name, prefix, fade=1.0):
    """Crossfade the orbit slices (``<prefix>NN_*.mp4``) into one continuous camera move through every kitchen style."""
    srcs = sorted(glob.glob(f"{CLIPS}/{prefix}[0-9][0-9]_*.mp4"))
    out = os.path.join(SEG, f"{name}.mp4")
    inputs, chain, prev, t = [], [], "0:v", 0.0
    for i, src in enumerate(srcs):
        inputs += ["-i", src]
        if i == 0:
            t = seg_seconds(src)
            continue
        t -= fade
        label = f"x{i}"
        chain.append(f"[{prev}][{i}:v]xfade=transition=fade:duration={fade}:offset={t:.3f}[{label}]")
        prev = label
        t += seg_seconds(src)
    # Style label, switched halfway through each crossfade (baked-in labels would blend into each other).
    labels = []
    for i, src in enumerate(srcs):
        style = os.path.basename(src)[4:-4]
        start = 0.0 if i == 0 else i * (seg_seconds(srcs[0]) - fade) + fade / 2
        end = t if i == len(srcs) - 1 else (i + 1) * (seg_seconds(srcs[0]) - fade) + fade / 2
        labels.append(f"drawtext=fontfile={FONT_BOLD}:text='Kitchen one_wall / {style}   ({i + 1}/{len(srcs)})'"
                      f":x=22:y=22:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8"
                      f":enable='between(t,{start:.3f},{end:.3f})'")
    chain.append(f"[{prev}]{','.join(labels)},fade=in:st=0:d=0.4,fade=out:st={t - 0.4:.3f}:d=0.4,fps={FPS}[out]")
    sh(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(chain), "-map", "[out]", "-an",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS), out])
    return out


def probe_frames(src, start, end):
    meta = json.load(open(src.replace(".mp4", ".json")))
    total = meta.get("frames", 0)
    return (end if end is not None else total) - (start or 0)


def seg_seconds(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout
    return float(out)


def concat(parts, out, target_mb=13.5):
    chapters, t = [], 0.0
    paths = []
    for item in parts:
        path, label = item if isinstance(item, tuple) else (item, None)
        if label:
            entry = dict(label) if isinstance(label, dict) else {"label": label}
            entry["t"] = round(t, 2)
            chapters.append(entry)
        t += seg_seconds(path)
        paths.append(path)
    with open(out.replace(".mp4", "_chapters.json"), "w") as f:
        json.dump(chapters, f, indent=1)
    parts = paths
    lst = os.path.join(SEG, os.path.basename(out) + ".txt")
    with open(lst, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    joined = out.replace(".mp4", "_joined.mp4")
    sh(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", joined])
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", joined],
                               capture_output=True, text=True).stdout)
    kbps = int(target_mb * 8 * 1024 / dur * 0.95)
    for pas in (1, 2):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", joined, "-c:v", "libx264", "-preset", "slow",
               "-b:v", f"{kbps}k", "-pass", str(pas), "-passlogfile", out + ".log", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-an"]
        sh(cmd + (["-f", "mp4", "/dev/null"] if pas == 1 else [out]))
    # poster frame
    sh(["ffmpeg", "-y", "-loglevel", "error", "-ss", "4", "-i", out, "-frames:v", "1", "-q:v", "3",
        out.replace(".mp4", "_poster.jpg")])
    print(f"{out}: {dur:.1f}s, {os.path.getsize(out) / 1e6:.1f} MB @ {kbps} kbps")


def env_intro(robot=True):
    parts = [(card("c00", "G1 Loco-Manipulation Data Factory",
                  ["Unitree G1 on Isaac Lab-Arena: WBC legs + PINK IK arms + Dex3 hands",
                   "New scenes, keyboard teleop and automatic demo generation"], kicker="Isaac Lab-Arena", seconds=3.5), "Title")]
    parts.append((card("c01", "Before: one G1 scene", ["Arena's official G1 task: carry a box from a shelf to a bin",
                                                      "Demos could only be collected with an XR headset"],
                      kicker="Baseline", seconds=3.0), "Baseline: Galileo warehouse"))
    parts.append(segment("s_t00", f"{CLIPS}/t00_galileo.mp4", start=10))
    parts.append((card("c02", "New: G1 kitchen environment", ["g1_kitchen_pick_and_place  |  10 Lightwheel RoboCasa styles",
                                                             "HOPE / YCB / Objaverse objects, randomized poses each reset"],
                      kicker="Added", seconds=3.0), "New: 10 kitchen styles"))
    # One camera move, the kitchen style changes every 3 s.
    parts.append(orbit("s_orbit" if robot else "s_orbit_norobot", "r" if robot else "o"))
    parts.append((card("c03", "New: keyboard teleop, no headset", ["g1_keyboard device emits the same 23-D action as XR teleop",
                                                                  "WASD walk  |  arrows + T/G wrists  |  K grasp  |  Z/X squat"],
                      kicker="Added", seconds=3.0), "New: keyboard teleop"))
    if os.path.exists(f"{CLIPS}/t20_teleop.mp4"):
        parts.append(segment("s_t20", f"{CLIPS}/t20_teleop.mp4", speed=1.25))
    os.makedirs(OUT, exist_ok=True)
    concat(parts, os.path.join(OUT, "g1_env_intro.mp4" if robot else "g1_env_intro_norobot.mp4"))


def auto_data():
    parts = [(card("c10", "Automatic demo generation",
                  ["Scripted experts read privileged state and emit the teleop action",
                   "Every episode: new materials and lighting  |  every run: new kitchen and object",
                   "Only successful episodes are written to HDF5"], kicker="Added", seconds=4.0), "Overview")]
    names = sorted(glob.glob(f"{CLIPS}/e*_*.mp4"))
    demo_count = 0
    for p in names:
        if not os.path.exists(p.replace(".mp4", ".json")):
            continue
        meta = json.load(open(p.replace(".mp4", ".json")))
        for ev in meta["events"]:
            if ev.get("outcome") == "success":
                ev["demo_index"] = demo_count
                demo_count += 1
        base = os.path.basename(p)[:-4]
        for ev in meta["events"]:
            s, e = ev["start_frame"], ev["end_frame"]
            n_action = e - s
            speed = max(1.0, n_action / FPS / 10.0)  # every episode shown in about 10 s
            label = f"{meta['label']}  ep {ev['episode'] + 1}: {'success' if ev['outcome'] == 'success' else 'failed'}"
            parts.append((segment(f"s_{base}_{ev['episode']}a", p, start=s, end=e, speed=round(speed, 2), fade=False),
                          {"label": label, "outcome": "success" if ev["outcome"] == "success" else "fail"}))
            parts.append(badge_hold(f"s_{base}_{ev['episode']}b", p, e - 1, ev["outcome"] == "success",
                                    ev.get("demo_index")))
    parts.append((card("c19", "Same format as human demos",
                      ["actions (23-D), robot & object states, head-camera RGB per step",
                       "-> replay, Mimic augmentation, LeRobot conversion, GR00T fine-tuning"],
                      kicker="Output", seconds=3.5), "Output format"))
    os.makedirs(OUT, exist_ok=True)
    concat(parts, os.path.join(OUT, "g1_auto_data.mp4"), target_mb=14.3)


SELECTED_ORDER = ["f04_ketchup", "f05_soup_toaster", "f06_mug", "f02_water_bottle", "f03_duck", "f01_pear"]
"""Clips of ``run_selected.sh`` in video order; clips without a successful episode are left out."""

SELECTED_SOURCE = {"f04_ketchup": "e12_modern2"}
"""Selection entries filmed by an earlier clip instead (the original ketchup run, without arrows)."""


def first_success(meta):
    return next((ev for ev in meta["events"] if ev.get("outcome") == "success"), None)


def auto_data_selected():
    """Auto data video, one successful episode per object, each preceded by the object/destination selection."""
    import selection_cards

    parts = [(card("c10", "Automatic demo generation",
                  ["Pick any object (also straight from Objaverse) and where to put it",
                   "Scripted experts read privileged state and emit the teleop action",
                   "Only successful episodes are written to HDF5"], kicker="Added", seconds=4.0), "Overview")]
    demo = 0
    for name in SELECTED_ORDER:
        src = f"{CLIPS}/{SELECTED_SOURCE.get(name, name)}.mp4"
        if not os.path.exists(src.replace(".mp4", ".json")):
            continue
        meta = json.load(open(src.replace(".mp4", ".json")))
        ev = first_success(meta)
        if ev is None:
            print(f"skip {name}: no successful episode")
            continue
        category, _, obj, dst, _ = selection_cards.SELECTIONS[name]
        sel = selection_cards.segment(name, "Choose the object and where to put it")
        parts.append((sel, {"label": f"{obj} -> {dst}", "outcome": "success"}))
        s, e = ev["start_frame"], ev["end_frame"]
        speed = max(1.0, (e - s) / FPS / 11.0)
        parts.append(segment(f"s_{name}a", src, start=s, end=e, speed=round(speed, 2), fade=False, demos=demo))
        parts.append(badge_hold(f"s_{name}b", src, e - 1, True, demo))
        demo += 1
    box = f"{CLIPS}/e20_galileo.mp4"
    meta = json.load(open(box.replace(".mp4", ".json")))
    ev = first_success(meta)
    if ev is not None:
        s, e = ev["start_frame"], ev["end_frame"]
        parts.append((segment("s_e20_sel_a", box, start=s, end=e, speed=round(max(1.0, (e - s) / FPS / 11.0), 2),
                              fade=False, demos=demo), {"label": "brown_box -> bin (warehouse, two hands)", "outcome": "success"}))
        parts.append(badge_hold("s_e20_sel_b", box, e - 1, True, demo))
    parts.append((card("c19", "Same format as human demos",
                      ["actions (23-D), robot & object states, head-camera RGB per step",
                       "-> replay, Mimic augmentation, LeRobot conversion, GR00T fine-tuning"],
                      kicker="Output", seconds=3.5), "Output format"))
    os.makedirs(OUT, exist_ok=True)
    concat(parts, os.path.join(OUT, "g1_auto_data.mp4"), target_mb=14.3)


if __name__ == "__main__":
    os.makedirs(SEG, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "intro"):
        env_intro()
    if which == "intro_norobot":
        env_intro(robot=False)
    if which in ("all", "auto"):
        auto_data_selected()
    if which == "auto_v1":
        auto_data()  # first version: every episode of the e* clips, failures included
