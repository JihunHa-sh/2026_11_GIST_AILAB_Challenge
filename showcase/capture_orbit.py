"""Capture one slice of the kitchen-style orbit: robot standing (or hidden), camera on a path shared by every style.

The camera path is a function of a global time, so slices of different kitchen styles line up and
``assemble.py`` can crossfade them into one continuous orbit in which the kitchen changes.
Slice ``ORBIT_INDEX`` covers global time [i * ORBIT_SEG, (i + 1) * ORBIT_SEG + ORBIT_FADE).
Env vars: SHOWCASE_OUT, SHOWCASE_LABEL, ORBIT_HIDE_ROBOT (1 hides the robot), ORBIT_INDEX, ORBIT_COUNT, ORBIT_SEG, ORBIT_FADE. The rest of the CLI
is the usual Arena env CLI.
"""

import json
import math
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import showcase_lib as sl  # noqa: E402

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser  # noqa: E402
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext  # noqa: E402
from isaaclab_arena_environments.cli import (  # noqa: E402
    get_arena_builder_from_cli,
    get_isaaclab_arena_environments_cli_parser,
)

OUT = os.environ["SHOWCASE_OUT"]
LABEL = os.environ.get("SHOWCASE_LABEL", "")
INDEX = int(os.environ.get("ORBIT_INDEX", "0"))
COUNT = int(os.environ.get("ORBIT_COUNT", "10"))
SEG = float(os.environ.get("ORBIT_SEG", "3.0"))
FADE = float(os.environ.get("ORBIT_FADE", "1.0"))
HIDE_ROBOT = os.environ.get("ORBIT_HIDE_ROBOT", "0") == "1"
FPS = 25

# Orbit inside the one_wall kitchen (room x 0..5.5, y -3..0, wall at y = 0). With the robot hidden the orbit
# frames the whole kitchen wall; with the robot it centers on the robot's start pose (x 3.95, y -1.4) and the
# counter in front of it.
if HIDE_ROBOT:
    CENTER, RADIUS, TARGET_Y, PAN = (2.6, -0.3), 2.75, -0.15, 0.6
    ANGLE_START, ANGLE_END = math.radians(-136.0), math.radians(-44.0)
else:
    CENTER, RADIUS, TARGET_Y, PAN = (3.6, -0.95), 2.2, -0.85, 0.35
    ANGLE_START, ANGLE_END = math.radians(-150.0), math.radians(-55.0)


def camera(g):
    """Eye and target at global time ``g`` seconds."""
    total = COUNT * SEG + FADE
    u = min(max(g / total, 0.0), 1.0)
    u = u - math.sin(2 * math.pi * u) / (2 * math.pi) * 0.35  # gentle ease in/out, never stops
    ang = ANGLE_START + (ANGLE_END - ANGLE_START) * u
    eye = (CENTER[0] + RADIUS * math.cos(ang), CENTER[1] + RADIUS * math.sin(ang), 1.65 + 0.15 * math.sin(math.pi * u))
    eye = (min(max(eye[0], 0.3), 5.2), min(max(eye[1], -2.8), -0.8), eye[2])
    tgt = (CENTER[0] + PAN * math.cos(ang + math.pi), TARGET_Y, 0.95)
    return eye, tgt


def hide_robot(env):
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    path = env.unwrapped.scene["robot"].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    path = path.replace("env_.*", "env_0")
    prim = stage.GetPrimAtPath(path)
    assert prim.IsValid(), f"robot prim not found at {path}"
    UsdGeom.Imageable(prim).MakeInvisible()
    print(f"[orbit] hid {path}", flush=True)


def render(unwrapped):
    with torch.inference_mode():
        unwrapped.sim.render()
        unwrapped.scene.update(unwrapped.physics_dt)


def run(parser):
    parser = get_isaaclab_arena_environments_cli_parser(parser)
    args, hydra = parser.parse_known_args(sys.argv[1:])
    builder = get_arena_builder_from_cli(args, hydra_overrides=hydra)
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    env_cfg.recorders = {}
    sl.add_showcase_camera(env_cfg)
    env = builder.make_registered(env_cfg, env_kwargs)
    unwrapped = env.unwrapped
    with torch.inference_mode():
        env.reset()
    robot_pos = unwrapped.scene["robot"].data.root_pos_w[0].tolist()
    print(f"[orbit] robot root at {[round(v, 3) for v in robot_pos]}", flush=True)
    if HIDE_ROBOT:
        hide_robot(env)
    # Physics stays paused, so the robot holds its reset pose (the arms-down start); only rendering advances.
    g0 = INDEX * SEG
    n_frames = int(round((SEG + FADE) * FPS))
    # Let textures stream in with the camera parked on the first pose.
    for _ in range(80):
        sl.look_at(env, *camera(g0))
        render(unwrapped)
    enc = sl.Encoder(OUT, fps=FPS)
    for k in range(n_frames):
        sl.look_at(env, *camera(g0 + k / FPS))
        render(unwrapped)
        enc.write(sl.overlay(sl.grab(env), lines_tl=[LABEL] if LABEL else ()))
    enc.close()
    with open(OUT.replace(".mp4", ".json"), "w") as f:
        json.dump({"label": LABEL, "mode": "orbit", "fps": FPS, "index": INDEX, "hide_robot": HIDE_ROBOT, "seg": SEG, "fade": FADE,
                   "frames": enc.n}, f, indent=1)
    print(f"[orbit] wrote {OUT} ({enc.n} frames)", flush=True)
    env.close()


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.set_defaults(device="cpu")
    args, _ = parser.parse_known_args(sys.argv[1:])
    with SimulationAppContext(args):
        try:
            run(parser)
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
