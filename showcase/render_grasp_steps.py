"""Render the grasp-planning steps of a captured episode (``capture_grasp.py``) as an mp4.

Re-runs ``G1GraspPlanner`` on the recorded inputs with every sample logged, then animates: object mesh, convex
hull, sampled closing axes, antipodal and wrap checks, support-collision and orientation filters, ranking, the cuRobo
pre-grasp checks of the top candidates, and finally the sim footage with the chosen grasp drawn over it.

    python render_grasp_steps.py clips/g00_mug        # writes segments/grasp_steps.mp4
"""

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import showcase_lib as sl  # noqa: E402

from isaaclab_arena_g1.policy import g1_grasp_planner as gp  # noqa: E402

FPS = 25
BG = "#10161c"
PANEL = "#151d24"
INK = "#e8eef0"
MUTED = "#8c9aa3"
ACCENT = "#76b900"
OK = "#3dbb8c"
BAD = "#ff6b5e"
WRAP = "#59a8ff"
GOLD = "#ffcc33"
SAMPLE = "#9fb3bf"

# --------------------------------------------------------------------------- planner with every sample logged


class LoggedPlanner(gp.G1GraspPlanner):
    """``G1GraspPlanner`` that also records every sampled candidate and why it was dropped."""

    def plan(self, *args, **kwargs):
        self.samples = []
        self.dropped = {}
        self.valid = []
        return super().plan(*args, **kwargs)

    def _evaluate(self, mesh, center, axis, approach, side, kind):
        grasp = super()._evaluate(mesh, center, axis, approach, side, kind)
        reason, contacts = "ok", None
        if grasp is None:
            reason, contacts = self._pinch_reason(mesh, center, axis)
        else:
            contacts = self._contacts(mesh, center, axis)
        self.samples.append({"type": "pinch", "kind": kind, "center": center.copy(), "axis": axis.copy(),
                             "approach": approach.copy(), "reason": reason, "contacts": contacts, "grasp": grasp})
        return grasp

    def _evaluate_power(self, mesh, center, axis, approach, side):
        grasp = super()._evaluate_power(mesh, center, axis, approach, side)
        self.samples.append({"type": "power", "kind": "power", "center": center.copy(), "axis": axis.copy(),
                             "approach": approach.copy(), "reason": "ok" if grasp is not None else "size",
                             "contacts": None, "grasp": grasp})
        return grasp

    def _contacts(self, mesh, center, axis):
        origins = np.array([center - 0.5 * axis, center + 0.5 * axis])
        locations, ray_ids, tris = mesh.ray.intersects_location(origins, np.array([axis, -axis]), multiple_hits=False)
        if len(set(ray_ids.tolist())) < 2:
            return None
        hit = {int(r): (loc, mesh.face_normals[t]) for loc, r, t in zip(locations, ray_ids, tris)}
        return [(hit[0][0], hit[0][1]), (hit[1][0], hit[1][1])]

    def _pinch_reason(self, mesh, center, axis):
        p = self.params
        contacts = self._contacts(mesh, center, axis)
        if contacts is None:
            return "miss", None
        (p_t, n_t), (p_f, n_f) = contacts
        width = float(np.dot(p_f - p_t, axis))
        if not p.min_width <= width <= p.max_width:
            return "width", contacts
        return "friction", contacts


def run_planner(hull, call):
    planner = LoggedPlanner(trimesh.Trimesh(hull.vertices, hull.faces))
    grasps = planner.plan(np.array(call["object_pos"]), np.array(call["object_quat_xyzw"]), call["side"],
                          call["support_height"], np.array(call["pelvis_pos"][:2]), robot_yaw=call["heading"])
    # Classify the post-filters the same way ``plan`` does.
    by_id = {id(s["grasp"]): s for s in planner.samples if s["grasp"] is not None}
    kept = {id(g) for g in grasps}
    stats = planner.last_stats
    return planner, grasps, by_id, kept, stats


# --------------------------------------------------------------------------- geometry helpers


def quat_matrix(q):
    x, y, z, w = q
    return trimesh.transformations.quaternion_matrix([w, x, y, z])[:3, :3]


BOX_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
BOX_FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]


def hand_boxes(grasp, side="right"):
    """The three open-hand boxes (palm, fingers, thumb) of the planner's hand model, in the world."""
    _, _, pts = gp.hand_model(side)
    R = quat_matrix(grasp.wrist_quat_xyzw)
    world = pts @ R.T + grasp.wrist_pos
    return [world[8 * i:8 * i + 8] for i in range(3)]


def box_segments(boxes):
    return [(b[i], b[j]) for b in boxes for i, j in BOX_EDGES]


def box_polys(boxes):
    return [[b[k] for k in f] for b in boxes for f in BOX_FACES]


def object_world_mesh(v, f, call):
    T = np.eye(4)
    T[:3, :3] = quat_matrix(call["object_quat_xyzw"])
    T[:3, 3] = call["object_pos"]
    m = trimesh.Trimesh(v, f, process=False)
    m.apply_transform(T)
    return m


def shaded_polys(mesh, base=(0.82, 0.84, 0.86), light=(0.4, -0.6, 0.7), max_faces=12000):
    faces = mesh.faces
    if len(faces) > max_faces:
        faces = faces[np.linspace(0, len(faces) - 1, max_faces).astype(int)]
        normals = mesh.face_normals[np.linspace(0, len(mesh.faces) - 1, max_faces).astype(int)]
    else:
        normals = mesh.face_normals
    lam = np.clip(normals @ (np.array(light) / np.linalg.norm(light)), 0, 1) * 0.65 + 0.3
    colors = np.clip(np.outer(lam, base), 0, 1)
    return mesh.vertices[faces], colors


# --------------------------------------------------------------------------- figure


class Scene3D:
    def __init__(self, center, span=0.24):
        self.fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
        self.ax = self.fig.add_axes([-0.03, -0.08, 0.72, 1.16], projection="3d", facecolor=BG)
        self.panel = self.fig.add_axes([0.66, 0.0, 0.34, 1.0], facecolor=PANEL)
        self.panel.set_axis_off()
        self.center = np.asarray(center)
        self.span = span

    def reset(self, azim, elev=24, span=None):
        ax = self.ax
        ax.cla()
        ax.set_facecolor(BG)
        ax.set_axis_off()
        self.span = span or self.span
        c, s = self.center, self.span
        ax.set_xlim(c[0] - s, c[0] + s)
        ax.set_ylim(c[1] - s, c[1] + s)
        ax.set_zlim(c[2] - s * 0.75, c[2] + s * 0.75)
        ax.set_box_aspect((1, 1, 0.75), zoom=1.25)
        ax.view_init(elev=elev, azim=azim)
        self.panel.cla()
        self.panel.set_axis_off()
        self.panel.set_xlim(0, 1)
        self.panel.set_ylim(0, 1)

    def steps_panel(self, steps, current, counts, note=""):
        p = self.panel
        p.text(0.06, 0.93, "GRASP PLANNING", color=ACCENT, fontsize=12, fontweight="bold", family="DejaVu Sans")
        y = 0.85
        for i, (title, sub) in enumerate(steps):
            active = i == current
            done = i < current
            color = INK if active else (MUTED if done else "#4d5a62")
            p.text(0.06, y, f"{i + 1}", color=ACCENT if active else color, fontsize=13, fontweight="bold")
            p.text(0.13, y, title, color=color, fontsize=12.5, fontweight="bold" if active else "normal")
            if active and sub:
                p.text(0.13, y - 0.035, sub, color=MUTED, fontsize=10, va="top", wrap=True, linespacing=1.4)
                y -= 0.035 * (sub.count("\n") + 1)
            if counts.get(i):
                p.text(0.95, y, counts[i], color=color, fontsize=10.5, ha="right", family="DejaVu Sans Mono")
            y -= 0.062
        if note:
            p.text(0.06, 0.04, note, color=INK, fontsize=11, va="bottom", linespacing=1.5)

    def support(self, vertices, alpha=0.35):
        top = vertices[:, 2].max()
        c, s = self.center, self.span
        x0, x1 = max(vertices[:, 0].min(), c[0] - s), min(vertices[:, 0].max(), c[0] + s)
        y0, y1 = max(vertices[:, 1].min(), c[1] - s), min(vertices[:, 1].max(), c[1] + s)
        quad = [[(x0, y0, top), (x1, y0, top), (x1, y1, top), (x0, y1, top)]]
        self.ax.add_collection3d(Poly3DCollection(quad, facecolor="#6b7a84", edgecolor="#8c9aa3", alpha=alpha,
                                                  linewidths=0.6))

    def mesh(self, polys, colors, alpha=1.0):
        pc = Poly3DCollection(polys, facecolors=colors, edgecolor="none", alpha=alpha)
        self.ax.add_collection3d(pc)

    def wire(self, mesh, color, alpha=0.8, lw=0.6):
        segs = mesh.vertices[mesh.edges_unique]
        self.ax.add_collection3d(Line3DCollection(segs, colors=color, alpha=alpha, linewidths=lw))

    def lines(self, segs, color, alpha=0.6, lw=1.0):
        if len(segs):
            self.ax.add_collection3d(Line3DCollection(segs, colors=color, alpha=alpha, linewidths=lw))

    def points(self, pts, color, size=18, alpha=1.0):
        if len(pts):
            pts = np.asarray(pts)
            self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color, s=size, alpha=alpha, depthshade=False)

    def hand(self, grasp, color, alpha=0.25, lw=1.2):
        boxes = hand_boxes(grasp)
        self.ax.add_collection3d(Poly3DCollection(box_polys(boxes), facecolor=color, edgecolor="none", alpha=alpha))
        self.lines(box_segments(boxes), color, alpha=min(1.0, alpha * 3), lw=lw)

    def image(self):
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]
        return np.ascontiguousarray(buf)


def pregrasp(grasp, distance=0.08, behind=0.06):
    """The expert's two pre-grasp poses (see ``_pregrasp_poses``): out along the approach, then back along the
    fingers."""
    fingers = quat_matrix(grasp.wrist_quat_xyzw)[:, 0]
    beside = gp.Grasp(grasp.wrist_pos - distance * grasp.approach, grasp.wrist_quat_xyzw, grasp.approach,
                      grasp.width, grasp.score, grasp.kind)
    back = gp.Grasp(beside.wrist_pos - behind * fingers, grasp.wrist_quat_xyzw, grasp.approach, grasp.width,
                    grasp.score, grasp.kind)
    return back, beside


# --------------------------------------------------------------------------- sim overlay


def project(points, eye, target, fx, w, h):
    eye, target = np.asarray(eye), np.asarray(target)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = np.asarray(points) - eye
    z = d @ f
    return np.stack([w / 2 + fx * (d @ r) / z, h / 2 - fx * (d @ u) / z], axis=1), z


def draw_hand_overlay(frame, grasp, cam, fx, color=(255, 204, 51), alpha=1.0):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    a = int(255 * alpha)
    for box in hand_boxes(grasp):
        uv, z = project(box, cam["eye"], cam["target"], fx, frame.shape[1], frame.shape[0])
        if (z <= 0).any():
            continue
        for i, j in BOX_EDGES:
            d.line([tuple(uv[i]), tuple(uv[j])], fill=(*color, a), width=3)
    return np.array(img)


# --------------------------------------------------------------------------- main


def main(base):
    meta = json.load(open(base + ".json"))
    data = np.load(base + ".npz")
    calls = [c for c in meta["grasp_calls"] if c["phase"] == "reach_side"]
    call = calls[-1]
    tries = [a for a in meta["arm_calls"] if a["step"] == call["step"] and a["phase"] == "reach_side"]
    hull = trimesh.Trimesh(data["hull_v"], data["hull_f"])
    planner, grasps, by_id, kept, stats = run_planner(hull, call)
    rec = call["grasps"]
    assert len(grasps) == len(rec) and all(
        np.allclose(g.wrist_pos, r["wrist_pos"], atol=1e-6) for g, r in zip(grasps, rec)
    ), "re-run of the grasp planner does not match the recorded grasps"
    chosen_index = next((i for i, t in enumerate(tries) if t["ok"]), None)
    print(f"samples {len(planner.samples)}, valid grasps {len(grasps)}, stats {stats}, "
          f"curobo tries {[t['ok'] for t in tries]}, chosen {chosen_index}")

    vis = object_world_mesh(data["mesh_v"], data["mesh_f"], call)
    hull_w = object_world_mesh(data["hull_v"], data["hull_f"], call)
    support_v = np.asarray(data["support_vertices"])
    center = hull_w.bounds.mean(axis=0) + np.array([0.0, 0.0, 0.03])
    polys, colors = shaded_polys(vis)
    hull_polys, hull_colors = shaded_polys(hull_w, base=(0.55, 0.75, 0.85))

    samples = planner.samples
    pinch = [s for s in samples if s["type"] == "pinch"]
    power = [s for s in samples if s["type"] == "power"]
    n_pinch_ok = sum(s["reason"] == "ok" for s in pinch)
    n_power_ok = sum(s["reason"] == "ok" for s in power)
    reasons = {k: sum(s["reason"] == k for s in pinch) for k in ("miss", "width", "friction")}
    passed = [s for s in samples if s["grasp"] is not None]
    support_rej = stats["support"]
    orient_rej = stats["power_orientation"]
    n_after_filters = stats["candidates"] - support_rej - orient_rej

    steps = [
        ("Object mesh", "Visual mesh of the object\n(Objaverse import)"),
        ("Convex hull", "Plan on the hull: the simulated\ncollider is at least this large"),
        ("Sample candidates", "Closing axis every 15 deg x height\nx approach tilt 0/30/60 deg + top"),
        ("Antipodal pinch check", "Ray-cast both contacts; normals inside\nfriction cone (mu 0.8), width 1.2-10 cm"),
        ("Wrap (power) check", "Palm on the object, fingers curl\naround a 3-8.5 cm section"),
        ("Collision & direction filter", "Open hand must clear the counter;\nwraps only straight-on"),
        ("Rank", "Contact quality + height + hand\ndirection relative to the robot"),
        ("cuRobo pre-grasp plan", "Best candidate whose pre-grasp pose\nhas a collision-free arm path"),
        ("Execute in Isaac Sim", "Pre-grasp -> slide in -> close -> lift"),
    ]
    counts = {}
    sc = Scene3D(center)
    enc = sl.Encoder(os.path.join(HERE, "segments", "grasp_steps.mp4"), fps=FPS)
    t_global = [0]

    def azim():
        return -60 + 18 * math.sin(t_global[0] / FPS * 0.35) + t_global[0] / FPS * 4.0

    chapters = []

    def emit(draw, current, seconds, note="", span=(0.12, 0.12)):
        if not chapters or chapters[-1]["step"] != current:
            chapters.append({"t": round(enc.n / FPS, 2), "step": current, "label": steps[current][0]})
        n = int(seconds * FPS)
        for k in range(n):
            u = k / max(n - 1, 1)
            e = u * u * (3 - 2 * u)
            sc.reset(azim(), span=span[0] + (span[1] - span[0]) * e)
            draw(k / max(n - 1, 1))
            sc.steps_panel(steps, current, counts, note)
            enc.write(sc.image())
            t_global[0] += 1

    # 0: the scene frame where planning happened
    frames = meta["frames"]
    plan_frame = next(i for i, f in enumerate(frames) if f["phase"] == "reach_side")
    import subprocess

    def sim_frame(i):
        out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", base + ".mp4", "-vf", f"select=eq(n\\,{i})",
                              "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        return np.frombuffer(out, np.uint8).reshape(meta["height"], meta["width"], 3).copy()

    chapters.append({"t": 0.0, "step": -1, "label": "Scene"})
    still = sim_frame(max(plan_frame - 1, 0))
    for k in range(int(2.5 * FPS)):
        enc.write(sl.overlay(still, ["Grasp planning on the object mesh", f"object: {meta['object']}"],
                             banner=None))

    # 1 mesh
    def d_mesh(u):
        sc.support(support_v, alpha=0.25)
        sc.mesh(polys, colors)
    emit(d_mesh, 0, 3.0, f"{len(vis.faces)} triangles")

    # 2 hull
    def d_hull(u):
        sc.support(support_v, alpha=0.25)
        sc.mesh(polys, colors, alpha=1.0 - 0.6 * u)
        sc.mesh(hull_polys, hull_colors, alpha=0.35 * u)
        sc.wire(hull_w, "#7fd3ff", alpha=0.9 * u)
    emit(d_hull, 1, 3.0, f"hull: {len(hull_w.faces)} triangles")

    # 3 sampling: closing axes appear as the yaw sweeps
    seg_len = 0.07

    def axis_seg(s):
        return (s["center"] - seg_len * s["axis"], s["center"] + seg_len * s["axis"])

    counts[2] = f"{len(samples)}"

    def d_sample(u):
        sc.support(support_v, alpha=0.25)
        sc.mesh(hull_polys, hull_colors, alpha=0.35)
        n = int(len(samples) * min(1.0, u * 1.15))
        sc.lines([axis_seg(s) for s in samples[:n]], SAMPLE, alpha=0.18, lw=0.6)
        sc.points([s["center"] for s in samples[:n]], SAMPLE, size=3, alpha=0.5)
    emit(d_sample, 2, 4.0, "each line = one closing axis\n(thumb <-> finger pads)")

    # 4 antipodal: three worked examples, then all pinch samples coloured
    examples = []
    for reason in ("ok", "width", "friction"):
        cand = [s for s in pinch if s["reason"] == reason and s["contacts"] is not None and s["kind"] == "side"]
        if cand:
            examples.append(cand[len(cand) // 2])
    counts[3] = f"{n_pinch_ok} ok"
    mu = 0.8

    def cone_segs(point, normal, length=0.035, n=10):
        normal = -normal / np.linalg.norm(normal)
        a = np.cross(normal, [0, 0, 1.0])
        if np.linalg.norm(a) < 1e-3:
            a = np.cross(normal, [1.0, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(normal, a)
        half = math.atan(mu)
        segs = []
        for k in range(n):
            phi = 2 * math.pi * k / n
            d = math.cos(half) * normal + math.sin(half) * (math.cos(phi) * a + math.sin(phi) * b)
            segs.append((point, point - length * d))
        return segs

    def d_anti(u):
        sc.support(support_v, alpha=0.25)
        sc.mesh(hull_polys, hull_colors, alpha=0.3)
        if u < 0.6:
            k = min(int(u / 0.6 * len(examples)), len(examples) - 1)
            ex = examples[k]
            color = OK if ex["reason"] == "ok" else BAD
            verdict = {"ok": "PASS: both normals inside the friction cone", "width": "FAIL: section too wide for the hand",
                       "friction": "FAIL: a normal is outside the friction cone"}[ex["reason"]]
            sc.ax.text2D(0.08, 0.9, f"example {k + 1}/{len(examples)}  -  {verdict}", transform=sc.ax.transAxes,
                         color=color, fontsize=13, fontweight="bold")
            sc.lines([(ex["center"] - 0.12 * ex["axis"], ex["center"] + 0.12 * ex["axis"])], color, alpha=0.9, lw=1.5)
            for p_c, n_c in ex["contacts"]:
                sc.points([p_c], color, size=40)
                sc.lines([(p_c, p_c + 0.04 * n_c)], INK, alpha=0.9, lw=1.5)
                sc.lines(cone_segs(p_c, n_c), color, alpha=0.5, lw=0.8)
        else:
            v = (u - 0.6) / 0.4
            n = int(len(pinch) * min(1.0, v * 1.3))
            ok = [axis_seg(s) for s in pinch[:n] if s["reason"] == "ok"]
            bad = [axis_seg(s) for s in pinch[:n] if s["reason"] != "ok"]
            sc.lines(bad, BAD, alpha=0.08, lw=0.5)
            sc.lines(ok, OK, alpha=0.45, lw=0.8)

    def anti_note():
        return (f"pinch: {n_pinch_ok} ok, {reasons['friction']} friction, {reasons['width']} width, "
                  f"{reasons['miss']} miss\n" + "white = surface normal, cone = friction cone")
    emit(d_anti, 3, 5.0, anti_note())

    # 5 power
    counts[4] = f"{n_power_ok} ok"

    def d_power(u):
        sc.support(support_v, alpha=0.25)
        sc.mesh(hull_polys, hull_colors, alpha=0.3)
        sc.lines([axis_seg(s) for s in pinch if s["reason"] == "ok"], OK, alpha=0.25, lw=0.6)
        sc.lines([axis_seg(s) for s in power if s["reason"] == "ok"], WRAP, alpha=0.6, lw=1.0)
        ex = [s for s in power if s["reason"] == "ok"]
        if ex:
            sc.hand(ex[len(ex) // 2]["grasp"], WRAP, alpha=0.18 * min(1.0, u * 3))
    emit(d_power, 4, 3.0, span=(0.12, 0.2), note= f"wrap: {n_power_ok} sections fit the curled fingers\nblue = wrap candidates")

    # 6 filters: support collisions
    support_height = call["support_height"]
    hits, clear = [], []
    for s in passed:
        g = s["grasp"]
        low = min(b[:, 2].min() for b in hand_boxes(g))
        (hits if low < support_height + (0.0 if g.kind == "power" else 0.015) else clear).append(g)
    counts[5] = f"{n_after_filters} left"

    def d_filter(u):
        sc.support(support_v, alpha=0.55)
        sc.mesh(hull_polys, hull_colors, alpha=0.3)
        show_hits = hits[:: max(1, len(hits) // 4)][:4]
        for g in show_hits:
            sc.hand(g, BAD, alpha=0.10)
        for g in clear[:: max(1, len(clear) // 3)][:3]:
            sc.hand(g, OK, alpha=0.12)
    emit(d_filter, 5, 4.0, span=(0.2, 0.2), note=f"{support_rej} hands hit the counter (red)\n{orient_rej} wraps not straight-on\n"
                           f"{n_after_filters} candidates left")

    # 7 ranking: top 10
    counts[6] = f"top {len(grasps)}"
    def d_rank(u):
        sc.support(support_v, alpha=0.4)
        sc.mesh(polys, colors, alpha=0.9)
        n = max(1, int(len(grasps) * min(1.0, u * 1.4)))
        for i in reversed(range(n)):
            if i:
                sc.lines(box_segments(hand_boxes(grasps[i])), WRAP if grasps[i].kind == "power" else SAMPLE,
                         alpha=0.35, lw=0.7)
            else:
                sc.hand(grasps[i], GOLD, alpha=0.2, lw=1.8)
    top = "\n".join(f"#{i + 1} {g.kind:6s} score {g.score:.2f}" for i, g in enumerate(grasps[:3]))
    emit(d_rank, 6, 4.0, top + "\n(gold = best)", span=(0.2, 0.2))

    # 8 cuRobo checks
    counts[7] = "-"
    for i, t in enumerate(tries):
        g = grasps[i]
        res = "collision-free path" if t["ok"] else f"no path ({t['failure'] or 'failed'})"
        counts[7] = f"#{i + 1} {'ok' if t['ok'] else 'fail'}"

        def d_try(u, g=g, t=t):
            sc.support(support_v, alpha=0.4)
            sc.mesh(polys, colors, alpha=0.9)
            behind, beside = pregrasp(g)
            sc.lines(box_segments(hand_boxes(behind)), GOLD, alpha=0.5, lw=0.9)
            sc.lines([(behind.wrist_pos, beside.wrist_pos), (beside.wrist_pos, g.wrist_pos)], GOLD, alpha=0.9, lw=1.6)
            sc.hand(g, OK if (t["ok"] and u > 0.4) else (BAD if u > 0.4 else GOLD), alpha=0.22, lw=1.8)
        emit(d_try, 7, 3.0, f"candidate #{i + 1} ({g.kind})\npre-grasp: 8 cm out, 6 cm back (thin box)\n"
                            f"-> cuRobo to the pre-grasp: {res}", span=(0.2, 0.2))
    if chosen_index is not None:
        chosen = grasps[chosen_index]

        def d_chosen(u):
            sc.support(support_v, alpha=0.4)
            sc.mesh(polys, colors)
            sc.hand(chosen, GOLD, alpha=0.3, lw=2.0)
        counts[7] = f"#{chosen_index + 1} chosen"
        emit(d_chosen, 7, 2.0, span=(0.2, 0.2), note=f"chosen: #{chosen_index + 1} {chosen.kind} grasp\n-> load it into the scene")
    else:
        chosen = grasps[0]

    # 9 sim footage with the chosen grasp drawn over it
    names = {"reach_side": "pre-grasp pose (cuRobo arm path)", "reach_forward": "slide forward beside the object",
             "contact": "move in along the approach", "close": "close the hand", "lift": "lift"}
    end = len(frames)
    for i, f in enumerate(frames):
        if i > plan_frame and f["phase"] not in names:
            end = min(i + 25, len(frames))
            break
    tmp = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", base + ".mp4", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-"], capture_output=True).stdout
    video = np.frombuffer(tmp, np.uint8).reshape(-1, meta["height"], meta["width"], 3)
    fx = meta["fx"]
    chapters.append({"t": round(enc.n / FPS, 2), "step": 8, "label": steps[8][0]})
    for i in range(plan_frame, min(end, len(video))):
        f = frames[i]
        frame = video[i].copy()
        if f["phase"] in ("reach_side", "reach_forward", "contact"):
            frame = draw_hand_overlay(frame, chosen, f, fx, alpha=0.9 if f["phase"] != "contact" else 0.5)
        caption = names.get(f["phase"], f["phase"])
        frame = sl.overlay(frame, ["9  Execute in Isaac Sim", f"phase: {f['phase']}  -  {caption}"],
                           ["yellow = planned grasp (open-hand model)"])
        enc.write(frame)
    enc.close()
    with open(os.path.join(HERE, "segments", "grasp_steps_chapters.json"), "w") as f:
        json.dump(chapters, f, indent=1)
    print(f"wrote segments/grasp_steps.mp4 ({enc.n} frames)")


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "segments"), exist_ok=True)
    main(os.path.join(HERE, sys.argv[1]) if not os.path.isabs(sys.argv[1]) else sys.argv[1])
