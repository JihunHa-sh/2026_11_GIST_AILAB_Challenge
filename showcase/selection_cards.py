"""Terminal + 3D-preview segments that show a person choosing the object (from Objaverse) and the destination.

Each segment types the real commands (import, then run with --object / --destination) into a terminal panel,
prints the real importer output, and spins a textured preview of the downloaded Objaverse model next to it.

    python selection_cards.py      # writes segments/sel_<clip>.mp4 for the clips in SELECTIONS
"""

import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import showcase_lib as sl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
SEG = os.path.join(HERE, "segments")
FPS = 25
W, H = 1280, 720
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
BG = (16, 22, 28)
TERM = (10, 14, 18)
PROMPT = (118, 185, 0)
TEXT = (220, 228, 232)
OUT_TEXT = (150, 165, 172)
HILITE = (255, 204, 51)

# clip -> (objaverse category or None, import log, local object name, destination, extra flags shown)
SELECTIONS = {
    "f01_pear": ("pear", None, "objaverse_pear_3", "plate_small_vomp_robolab", ""),
    "f02_water_bottle": ("water_bottle", "import_water_bottle.log", "objaverse_water_bottle_2", "bowl_ycb_robolab", ""),
    "f03_duck": ("duck", "import_duck.log", "objaverse_duck_0", "wooden_bowl_hot3d_robolab", ""),
    "f04_ketchup": (None, None, "ketchup_bottle_hope_robolab", "bowl_ycb_robolab", ""),
    "f05_soup_toaster": (None, None, "alphabet_soup_can_hope_robolab", "toaster_oven", "--destination_open"),
    "f06_mug": ("mug", None, "objaverse_mug_0", "serving_bowl_vomp_robolab", ""),
}


def shorten(line):
    line = line.replace(os.path.expanduser("~"), "~")
    line = re.sub(r"~/.cache/isaaclab_arena/local_objects/\S+/", "~/.cache/.../", line)
    return line


def importer_lines(category, log, name, size):
    cmd = f"python -m isaaclab_arena_g1.tools.objaverse_import --category {category} --size {size}"
    out = []
    if log and os.path.exists(os.path.join(LOGS, log)):
        for line in open(os.path.join(LOGS, log)):
            if f"{name}:" in line:
                out.append(shorten(line.strip()))
    if not out:
        # Same line the importer prints, rebuilt from the local registry entry.
        reg = json.load(open(os.path.expanduser("~/.cache/isaaclab_arena/local_objects/objects.json")))
        uid = reg[name]["source"].split(":", 1)[1]
        out = [f"[objaverse_import] {name}: {uid} -> ~/.cache/.../{name}.usd"]
    return cmd, out


def run_lines(name, destination, extra):
    flags = f"--object {name} --destination {destination}" + (f" {extra}" if extra else "")
    return ("python isaaclab_arena/scripts/record_policy_demos.py ... \\",
            f"    g1_kitchen_pick_and_place {flags}")


def wrap(text, font, width, draw):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) > width and cur:
            lines.append(cur)
            cur = "    " + w
        else:
            cur = trial
    lines.append(cur)
    return lines


class Terminal:
    def __init__(self, box):
        self.box = box  # x0, y0, x1, y1
        self.font = ImageFont.truetype(MONO, 17)
        self.bold = ImageFont.truetype(MONO_BOLD, 17)
        self.entries = []  # (kind, text, highlight words)

    def render(self, img, typed_chars=None):
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = self.box
        d.rounded_rectangle(self.box, 12, fill=TERM, outline=(48, 60, 68), width=2)
        for i, c in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
            d.ellipse([x0 + 16 + 22 * i, y0 + 14, x0 + 28 + 22 * i, y0 + 26], fill=c)
        d.text((x0 + 90, y0 + 9), "terminal", font=self.font, fill=OUT_TEXT)
        y = y0 + 46
        for k, (kind, text, hl) in enumerate(self.entries):
            shown = text if typed_chars is None or k != len(self.entries) - 1 else text[:typed_chars]
            prefix = "$ " if kind == "cmd" else ""
            for j, line in enumerate(wrap(prefix + shown, self.font, x1 - x0 - 40, d)):
                xx = x0 + 20
                if kind == "cmd" and j == 0 and line.startswith("$ "):
                    d.text((xx, y), "$ ", font=self.bold, fill=PROMPT)
                    xx += d.textlength("$ ", font=self.bold)
                    line = line[2:]
                color = TEXT if kind == "cmd" else OUT_TEXT
                # highlight the user's choices
                pos = 0
                for word in re.split(r"(\s+)", line):
                    fill = HILITE if word.strip() in hl else color
                    d.text((xx + pos, y), word, font=self.bold if fill == HILITE else self.font, fill=fill)
                    pos += d.textlength(word, font=self.font)
                y += 26
            y += 6


def load_preview_mesh(name):
    """Decimated, colored preview mesh of a local Objaverse object (colors from its glTF textures)."""
    import objaverse

    reg = json.load(open(os.path.expanduser("~/.cache/isaaclab_arena/local_objects/objects.json")))
    uid = reg[name]["source"].split(":", 1)[1]
    glb = objaverse.load_objects(uids=[uid])[uid]
    scene = trimesh.load(glb, force="scene")
    meshes = []
    for geom in scene.dump(concatenate=False):
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        try:
            colors = geom.visual.to_color().vertex_colors[:, :3]
        except Exception:
            colors = np.full((len(geom.vertices), 3), 200)
        m = trimesh.Trimesh(geom.vertices, geom.faces, vertex_colors=np.c_[colors, np.full(len(colors), 255)],
                            process=False)
        meshes.append(m)
    full = trimesh.util.concatenate(meshes)
    full.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    face_colors = full.visual.vertex_colors[full.faces][:, :, :3].mean(axis=1) / 255.0
    small = full.simplify_quadric_decimation(face_count=min(4000, len(full.faces))) if len(full.faces) > 4000 else full
    _, _, tri = trimesh.proximity.closest_point(full, small.triangles_center)
    small_colors = face_colors[tri]
    small.apply_translation(-small.bounds.mean(axis=0))
    small.apply_scale(1.0 / (small.bounds[1] - small.bounds[0]).max())
    return small, small_colors


def preview_frame(mesh, colors, azim):
    fig = plt.figure(figsize=(4.6, 4.6), dpi=100, facecolor="#0b1014")
    ax = fig.add_axes([0, 0, 1, 1], projection="3d", facecolor="#0b1014")
    ax.set_axis_off()
    light = np.array([0.35, -0.55, 0.75])
    lam = np.clip(mesh.face_normals @ (light / np.linalg.norm(light)), 0, 1) * 0.55 + 0.45
    ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors=np.clip(colors * lam[:, None], 0, 1),
                                         edgecolor="none"))
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.set_box_aspect((1, 1, 1), zoom=1.15)
    ax.view_init(elev=18, azim=azim)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def segment(clip, step_label):
    category, log, name, destination, extra = SELECTIONS[clip]
    enc = sl.Encoder(os.path.join(SEG, f"sel_{clip}.mp4"), fps=FPS)
    title_font = ImageFont.truetype(sl.FONT_BOLD, 30)
    sub_font = ImageFont.truetype(sl.FONT, 20)
    has_preview = category is not None
    term = Terminal((60, 140, 760 if has_preview else 1220, 640))
    script = []
    if category is not None:
        cmd, out = importer_lines(category, log, name, {"pear": 0.12, "water_bottle": 0.18, "duck": 0.10,
                                                        "mug": 0.11}[category])
        script.append(("cmd", cmd, {category}))
        script += [("out", line, {name}) for line in out]
    run1, run2 = run_lines(name, destination, extra)
    script.append(("cmd", run1, set()))
    script.append(("out", run2, {name, destination}))
    mesh = colors = None
    if has_preview:
        mesh, colors = load_preview_mesh(name)

    def base():
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((60, 46), step_label, font=title_font, fill=(240, 244, 246))
        sub = ("Objaverse object chosen by category, then the object and destination set on the command line"
               if has_preview else "Object and destination set on the command line")
        d.text((60, 90), sub, font=sub_font, fill=(170, 182, 190))
        return img

    frame_i = 0

    def write(img, show_preview):
        nonlocal frame_i
        if has_preview and show_preview:
            pv = preview_frame(mesh, colors, -60 + frame_i * 2.4)
            img.paste(Image.fromarray(pv), (790, 160))
            d = ImageDraw.Draw(img)
            d.text((800, 130), f"downloaded: {name}", font=sub_font, fill=HILITE)
        enc.write(np.array(img))
        frame_i += 1

    preview_on = False
    for k, (kind, text, hl) in enumerate(script):
        term.entries.append((kind, text, hl))
        if kind == "cmd":
            for c in range(0, len(text) + 1, 2):
                img = base()
                term.render(img, typed_chars=c)
                write(img, preview_on)
            for _ in range(6):
                img = base()
                term.render(img)
                write(img, preview_on)
        else:
            if k > 0 and script[k - 1][0] == "cmd" and category is not None and not preview_on:
                preview_on = True
            for _ in range(8):
                img = base()
                term.render(img)
                write(img, preview_on)
    for _ in range(int(1.6 * FPS)):
        img = base()
        term.render(img)
        write(img, preview_on)
    enc.close()
    return os.path.join(SEG, f"sel_{clip}.mp4")


if __name__ == "__main__":
    import sys

    os.makedirs(SEG, exist_ok=True)
    for i, clip in enumerate(sys.argv[1:] or SELECTIONS):
        print(segment(clip, "Choose the object and where to put it"))
