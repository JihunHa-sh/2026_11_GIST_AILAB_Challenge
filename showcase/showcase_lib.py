"""Helpers for the showcase videos: a free-flying scene camera, per-episode visual randomization, overlays."""

import glob
import math
import os
import subprocess
import types

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
TEX_DIR = os.path.join(HERE, "textures")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
WIDTH, HEIGHT = 1280, 720


# ---------------------------------------------------------------- scene camera


def add_showcase_camera(env_cfg, width=WIDTH, height=HEIGHT, focal_length=14.0):
    """Add a free camera to the scene config; move it every step with :func:`look_at`."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    env_cfg.scene.showcase_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ShowcaseCam",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal_length, clipping_range=(0.05, 40.0)),
    )


def look_at(env, eye, target):
    import torch

    cam = env.unwrapped.scene["showcase_cam"]
    dev = env.unwrapped.device
    cam.set_world_poses_from_view(
        torch.tensor([eye], dtype=torch.float32, device=dev), torch.tensor([target], dtype=torch.float32, device=dev)
    )


def grab(env, name="showcase_cam"):
    rgb = env.unwrapped.scene[name].data.output["rgb"]
    rgb = rgb.torch if hasattr(rgb, "torch") else rgb
    arr = rgb[0, ..., :3].cpu().numpy()
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr.copy()


def orbit_eye(center, radius, height, angle):
    return (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle), center[2] + height)


# ---------------------------------------------------------------- headless keyboard stub


def stub_keyboard():
    """Headless Kit has no app window: stub the OS keyboard so the g1_keyboard device can be built."""
    import carb
    import omni

    omni.appwindow = types.SimpleNamespace(
        get_default_app_window=lambda: types.SimpleNamespace(get_keyboard=lambda: "stub_keyboard")
    )
    carb.input = types.SimpleNamespace(
        KeyboardEventType=types.SimpleNamespace(KEY_PRESS="press", KEY_RELEASE="release"),
        acquire_input_interface=lambda: types.SimpleNamespace(
            subscribe_to_keyboard_events=lambda kb, fn: "stub_sub",
            unsubscribe_to_keyboard_events=lambda kb, sub: None,
        ),
    )


def press(device, name, down=True):
    device._on_keyboard_event(
        types.SimpleNamespace(input=types.SimpleNamespace(name=name), type="press" if down else "release")
    )


# ---------------------------------------------------------------- visual randomization


def make_procedural_textures(n=12, seed=0):
    os.makedirs(TEX_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = sorted(glob.glob(os.path.join(TEX_DIR, "proc_*.png")))
    if len(paths) >= n:
        return paths
    for k in range(n):
        size = 256
        c1 = tuple(int(v) for v in rng.integers(30, 255, 3))
        c2 = tuple(int(v) for v in rng.integers(30, 255, 3))
        img = Image.new("RGB", (size, size), c1)
        d = ImageDraw.Draw(img)
        kind = k % 4
        if kind == 0:  # stripes
            w = int(rng.integers(12, 40))
            for x in range(-size, size * 2, 2 * w):
                d.polygon([(x, 0), (x + w, 0), (x + w + size, size), (x + size, size)], fill=c2)
        elif kind == 1:  # checker
            w = int(rng.integers(16, 64))
            for i in range(0, size, w):
                for j in range(0, size, w):
                    if (i // w + j // w) % 2:
                        d.rectangle([i, j, i + w, j + w], fill=c2)
        elif kind == 2:  # dots
            w = int(rng.integers(20, 48))
            for i in range(0, size, w):
                for j in range(0, size, w):
                    r = w // 3
                    d.ellipse([i + w // 2 - r, j + w // 2 - r, i + w // 2 + r, j + w // 2 + r], fill=c2)
        else:  # noise blobs
            arr = rng.random((16, 16))
            arr = np.array(Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)) / 255.0
            a = np.array(c1)[None, None] * (1 - arr[..., None]) + np.array(c2)[None, None] * arr[..., None]
            img = Image.fromarray(a.astype(np.uint8))
        p = os.path.join(TEX_DIR, f"proc_{k:02d}.png")
        img.save(p)
    return sorted(glob.glob(os.path.join(TEX_DIR, "proc_*.png")))


def texture_pool():
    pool = make_procedural_textures()
    for name in ("gray_wood_grain_0", "gray_wood_planks_0", "marble_3_0", "light_gray_0"):
        pool += glob.glob(os.path.expanduser(f"~/.cache/lightwheel_sdk/floorplan/*/textures/{name}.png"))[:2]
    return pool


_MAT_COUNT = [0]


def randomize_prim_material(prim_path, rng, pool, p_texture=0.6):
    """Bind a fresh UsdPreviewSurface (random color or texture) to every mesh under ``prim_path``."""
    import omni.usd
    from pxr import Gf, Sdf, UsdShade

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return f"missing {prim_path}"
    _MAT_COUNT[0] += 1
    mat_path = f"/World/ShowcaseLooks/Mat_{_MAT_COUNT[0]}"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rng.uniform(0.25, 0.8)))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(rng.choice([0.0, 0.0, 0.6])))
    desc = ""
    if rng.random() < p_texture:
        tex_path = str(rng.choice(pool))
        st = UsdShade.Shader.Define(stage, mat_path + "/StReader")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, mat_path + "/Tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_path)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        desc = "texture " + os.path.basename(tex_path)
    else:
        color = Gf.Vec3f(*[float(v) for v in rng.uniform(0.05, 0.95, 3)])
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        desc = "color"
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    from pxr import Usd

    # Bind for the "full" (render) purpose only, so physics-purpose friction materials are left untouched.
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() in ("Mesh", "Xform", "Scope") or prim == root:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mat, UsdShade.Tokens.strongerThanDescendants, UsdShade.Tokens.full
            )
    return desc


def recolor_prim(prim_path, rng, pool, p_texture=0.6):
    """Change the existing visual material(s) under ``prim_path`` in place: a random texture or a random color.

    Only shader inputs are edited, so material bindings (and with them physics materials) stay untouched.
    """
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdShade

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return f"missing {prim_path}"
    use_texture = rng.random() < p_texture
    tex_path = str(rng.choice(pool))
    color = Gf.Vec3f(*[float(v) for v in rng.uniform(0.05, 0.95, 3)])
    edited = []
    shader_prims = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.GetTypeName() == "Shader"]
    if not shader_prims:
        # Materials can live elsewhere (e.g. a kitchen's shared Looks scope): follow the meshes' bindings.
        materials = {}
        from pxr import UsdGeom

        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            # Any geometric primitive: RoboCasa counter tops are Cube prims, not meshes.
            if not prim.IsA(UsdGeom.Gprim):
                continue
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if material:
                materials[str(material.GetPath())] = material.GetPrim()
        for mat_prim in materials.values():
            shader_prims += [p for p in Usd.PrimRange(mat_prim) if p.GetTypeName() == "Shader"]
        edited.append(f"{len(materials)} bound material(s)")
    for prim in shader_prims:
        if prim.GetTypeName() != "Shader":
            continue
        if prim.IsInstanceProxy():
            edited.append("instance-proxy(skip)")
            continue
        sh = UsdShade.Shader(prim)
        mdl = prim.GetAttribute("info:mdl:sourceAsset")
        shader_id = sh.GetIdAttr().Get() if sh.GetIdAttr() else None
        if mdl and mdl.Get() is not None:
            if use_texture:
                sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex_path))
                sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
            else:
                sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(""))
                sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(color)
                sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
            edited.append("mdl")
        elif shader_id == "UsdUVTexture":
            if use_texture:
                sh.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex_path))
            sh.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(1, 1, 1, 1) if use_texture else Gf.Vec4f(0, 0, 0, 1)
            )
            sh.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(0, 0, 0, 0) if use_texture else Gf.Vec4f(color[0], color[1], color[2], 0)
            )
            edited.append("uvtex")
        elif shader_id == "UsdPreviewSurface":
            inp = sh.GetInput("diffuseColor")
            if not (inp and inp.HasConnectedSource()):
                sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
                edited.append("preview")
    kind = ("texture " + os.path.basename(tex_path)) if use_texture else "color"
    return f"{kind} [{', '.join(sorted(set(edited))) or 'no shader found'}]"


class SelectionMarker:
    """Floating arrow (visual only, no physics) that marks a user-selected object or destination in the scene."""

    def __init__(self, name, color):
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = omni.usd.get_context().get_stage()
        self.path = f"/World/ShowcaseMarkers/{name}"
        root = UsdGeom.Xform.Define(stage, self.path)
        self.translate = root.AddTranslateOp()
        mat = UsdShade.Material.Define(stage, self.path + "/Look")
        shader = UsdShade.Shader.Define(stage, self.path + "/Look/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[0.6 * c for c in color]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        tip = UsdGeom.Cone.Define(stage, self.path + "/Tip")  # apex points down
        tip.CreateRadiusAttr(0.032)
        tip.CreateHeightAttr(0.06)
        tip.CreateAxisAttr("Z")
        tip.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.03))
        tip.AddScaleOp().Set(Gf.Vec3f(1, 1, -1))
        shaft = UsdGeom.Cylinder.Define(stage, self.path + "/Shaft")
        shaft.CreateRadiusAttr(0.011)
        shaft.CreateHeightAttr(0.07)
        shaft.CreateAxisAttr("Z")
        shaft.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.095))
        for prim in (tip, shaft):
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)
        self.imageable = UsdGeom.Imageable(root)
        self.visible = True

    def place(self, pos, t=0.0):
        """Put the arrow tip at ``pos`` (world), bobbing slowly with time ``t`` [s]."""
        from pxr import Gf

        self.translate.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]) + 0.012 * math.sin(3.0 * t)))

    def show(self, visible):
        if visible != self.visible:
            (self.imageable.MakeVisible if visible else self.imageable.MakeInvisible)()
            self.visible = visible


def randomize_light(rng):
    import omni.usd
    from pxr import Gf

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/Light")
    if not prim.IsValid():
        return "no light"
    intensity = float(rng.uniform(700, 2600))
    temp_color = rng.choice(
        [(1.0, 0.85, 0.7), (1.0, 0.95, 0.9), (0.85, 0.9, 1.0), (0.75, 0.75, 0.75), (1.0, 0.8, 0.85), (0.8, 1.0, 0.85)]
    )
    prim.GetAttribute("inputs:intensity").Set(intensity)
    prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*[float(c) for c in temp_color]))
    return f"light {intensity:.0f}"


# ---------------------------------------------------------------- overlays + encoding


def _font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def overlay(frame, lines_tl=(), lines_br=(), banner=None, badge=None):
    """Draw HUD text on a frame (top-left lines, bottom-right lines, top banner, colored badge)."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    if banner:
        f = _font(30, True)
        tw = d.textlength(banner, font=f)
        d.rounded_rectangle([W / 2 - tw / 2 - 18, 16, W / 2 + tw / 2 + 18, 64], 10, fill=(0, 0, 0, 150))
        d.text((W / 2 - tw / 2, 22), banner, font=f, fill=(255, 255, 255))
    y = 80 if banner else 18
    for i, line in enumerate(lines_tl):
        f = _font(22, i == 0)
        tw = d.textlength(line, font=f)
        d.rounded_rectangle([14, y - 4, 30 + tw, y + 28], 7, fill=(0, 0, 0, 140))
        d.text((22, y), line, font=f, fill=(255, 255, 255))
        y += 36
    y = H - 18 - 36 * len(lines_br)
    for line in lines_br:
        f = _font(22)
        tw = d.textlength(line, font=f)
        d.rounded_rectangle([W - 30 - tw, y - 4, W - 14, y + 28], 7, fill=(0, 0, 0, 140))
        d.text((W - 22 - tw, y), line, font=f, fill=(255, 255, 255))
        y += 36
    if badge:
        text, color = badge
        f = _font(34, True)
        tw = d.textlength(text, font=f)
        d.rounded_rectangle([W / 2 - tw / 2 - 22, H / 2 - 36, W / 2 + tw / 2 + 22, H / 2 + 30], 12, fill=color)
        d.text((W / 2 - tw / 2, H / 2 - 28), text, font=f, fill=(255, 255, 255))
    return np.array(img)


def compose_split(main, inset, inset_label="Robot head camera (recorded)"):
    """Place ``inset`` (head camera) in the bottom-left corner of ``main``."""
    img = Image.fromarray(main)
    ins = Image.fromarray(inset).resize((384, 288))
    d = ImageDraw.Draw(img, "RGBA")
    x0, y0 = 16, img.size[1] - 288 - 16
    d.rectangle([x0 - 4, y0 - 34, x0 + 388, y0 + 292], fill=(0, 0, 0, 170))
    img.paste(ins, (x0, y0))
    d.text((x0 + 4, y0 - 30), inset_label, font=_font(18, True), fill=(255, 255, 255))
    return np.array(img)


class Encoder:
    """Pipe RGB frames into ffmpeg (H.264, yuv420p) for web playback."""

    def __init__(self, path, fps=25, size=(WIDTH, HEIGHT)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", path,
            ],
            stdin=subprocess.PIPE,
        )
        self.n = 0

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.n += 1

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
