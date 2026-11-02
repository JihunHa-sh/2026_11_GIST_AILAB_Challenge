"""Capture one showcase clip (mp4 + json) in an Arena G1 environment.

Modes (set with the SHOWCASE_MODE env var, the rest of the CLI is the usual Arena env CLI):
  tour    robot stands, camera sweeps around it                        (SHOWCASE_SECONDS)
  teleop  scripted g1_keyboard key presses with an on-screen key HUD
  expert  scripted expert episodes with per-episode material/light randomization (SHOWCASE_EPISODES)
Other env vars: SHOWCASE_OUT (mp4 path), SHOWCASE_LABEL, SHOWCASE_SEED, SHOWCASE_EXPERT=one_hand|bimanual.
"""

import json
import math
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import showcase_lib as sl  # noqa: E402

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser  # noqa: E402
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext  # noqa: E402
from isaaclab_arena_environments.cli import (  # noqa: E402
    get_arena_builder_from_cli,
    get_isaaclab_arena_environments_cli_parser,
)

MODE = os.environ.get("SHOWCASE_MODE", "tour")
OUT = os.environ.get("SHOWCASE_OUT", os.path.join(sl.HERE, "clips", f"{MODE}.mp4"))
LABEL = os.environ.get("SHOWCASE_LABEL", "")
SEED = int(os.environ.get("SHOWCASE_SEED", "0"))
FPS = 25  # one frame every 2 control steps at 50 Hz -> real time


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.set_defaults(device="cpu")
    args, _ = parser.parse_known_args(sys.argv[1:])
    with SimulationAppContext(args):
        try:
            run(parser)
        except Exception:
            traceback.print_exc()


class Clip:
    def __init__(self, env):
        self.env = env
        self.enc = sl.Encoder(OUT, fps=FPS)
        self.meta = {"label": LABEL, "mode": MODE, "fps": FPS, "events": []}
        self.step_i = 0

    def frame(self, tl=(), br=(), banner=None, badge=None, inset=True):
        main = sl.grab(self.env)
        if inset:
            main = sl.compose_split(main, sl.grab(self.env, "robot_head_cam"))
        self.enc.write(sl.overlay(main, tl, br, banner, badge))

    def close(self):
        self.enc.close()
        self.meta["frames"] = self.enc.n
        with open(OUT.replace(".mp4", ".json"), "w") as f:
            json.dump(self.meta, f, indent=1)
        print(f"[showcase] wrote {OUT} ({self.enc.n} frames)", flush=True)


def build(parser):
    parser = get_isaaclab_arena_environments_cli_parser(parser)
    args, hydra = parser.parse_known_args(sys.argv[1:])
    builder = get_arena_builder_from_cli(args, hydra_overrides=hydra)
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    env_cfg.recorders = {}
    sl.add_showcase_camera(env_cfg)
    env = builder.make_registered(env_cfg, env_kwargs)
    return args, env


def pelvis_pose(env):
    p = env.unwrapped.arena_world.get_body_pose_w("robot", "pelvis")[0].cpu().numpy().astype(float)
    x, y, z, w = p[3:7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return p[:3], yaw


def chase_eye(pos, yaw, t, radius=1.9, height=1.05, swing=0.75):
    """Camera behind-and-beside the robot, swinging slowly from side to side."""
    ang = yaw + math.pi + swing * math.sin(t * 0.35)
    eye = (pos[0] + radius * math.cos(ang), pos[1] + radius * math.sin(ang), 0.75 + height)
    tgt = (pos[0] + 0.55 * math.cos(yaw), pos[1] + 0.55 * math.sin(yaw), 0.85)
    return eye, tgt


def clamp_to_room(eye, args):
    """Keep the camera inside the one_wall kitchen room (x 0..5.5, y -3..0); other scenes are left alone."""
    if getattr(args, "kitchen_background", None) is None or "one_wall" not in str(args.kitchen_background):
        return eye
    return (min(max(eye[0], 0.25), 5.25), min(max(eye[1], -2.8), -0.75), eye[2])


def run(parser):
    args, env = build(parser)
    sl.stub_keyboard()
    if MODE == "expert":
        run_expert(args, env)
    else:
        run_keyboard(args, env)
    env.close()


# ------------------------------------------------------------------ tour / teleop


TELEOP_SCRIPT = [  # (seconds, keys held, caption)
    (1.0, [], "Keyboard teleop: g1_keyboard device"),
    (1.6, ["T"], "T  raise both wrists"),
    (2.6, ["W"], "W  walk forward"),
    (1.2, [], ""),
    (1.6, ["A"], "A  turn left"),
    (1.6, ["D"], "D  turn right"),
    (1.4, ["E"], "E  side-step right"),
    (1.4, ["Q"], "Q  side-step left"),
    (1.2, ["UP"], "UP  reach forward"),
    (1.0, ["RIGHT"], "RIGHT  squeeze hands together (both arms)"),
    (0.8, ["LEFT"], "LEFT  spread hands"),
    (0.3, ["KEY_2"], "2  select right arm"),
    (1.0, ["K"], "K  close right hand"),
    (1.0, ["K"], "K  open right hand"),
    (1.6, ["Z"], "Z  squat"),
    (1.6, ["X"], "X  stand up"),
    (1.0, ["SPACE"], "SPACE  wrists back home"),
    (1.0, [], ""),
]
ONE_SHOT = {"K", "KEY_2", "SPACE"}


def run_keyboard(args, env):
    from isaaclab.devices.teleop_device_factory import create_teleop_device

    device = create_teleop_device("g1_keyboard", env.unwrapped.cfg.teleop_devices.devices, {})
    device._scene_list_printed = True
    with torch.inference_mode():
        env.reset()
    clip = Clip(env)
    pos0, yaw0 = pelvis_pose(env)
    scene = getattr(args, "kitchen_background", None) or env.unwrapped.cfg.__class__.__name__
    obj = getattr(args, "object", "")

    def step_and_capture(n_steps, tl, eye_fn, br=()):
        for k in range(n_steps):
            eye, tgt = eye_fn(clip.step_i)
            sl.look_at(env, clamp_to_room(eye, args), tgt)
            with torch.inference_mode():
                env.step(device.advance().repeat(1, 1))
            if k % 2 == 1:
                clip.frame(tl=tl, br=br)
            clip.step_i += 1

    if MODE == "tour":
        seconds = float(os.environ.get("SHOWCASE_SECONDS", "5"))
        # let the scene settle with the camera parked
        n = int(seconds * 50)
        center = (pos0[0] + 0.45 * math.cos(yaw0), pos0[1] + 0.45 * math.sin(yaw0), 0.8)

        galileo = getattr(args, "kitchen_background", None) is None
        if galileo:
            center = (pos0[0] + 0.45 * math.cos(yaw0), pos0[1] + 0.45 * math.sin(yaw0), 0.0)

        def eye_fn(i):
            u = i / max(n, 1)
            if galileo:
                # Sweep along the open side only; directly behind the robot is a wall.
                ang = yaw0 + math.pi - 1.5 + 0.8 * u
                r = 2.2 - 0.3 * u
                return (center[0] + r * math.cos(ang), center[1] + r * math.sin(ang), 0.95 - 0.25 * u), center
            ang = yaw0 + math.pi + 1.1 * (u - 0.5) * 2
            r = 2.3 - 0.4 * u
            return (center[0] + r * math.cos(ang), center[1] + r * math.sin(ang), 1.75 - 0.35 * u), center

        for _ in range(30):
            eye, tgt = eye_fn(0)
            sl.look_at(env, clamp_to_room(eye, args), tgt)
            with torch.inference_mode():
                env.step(device.advance().repeat(1, 1))
        lines = [LABEL] if LABEL else [scene]
        if obj:
            lines.append(f"object: {obj}")
        step_and_capture(n, lines, eye_fn)
    else:
        for seconds, keys, caption in TELEOP_SCRIPT:
            for key in keys:
                sl.press(device, key, True)
                if key in ONE_SHOT:
                    sl.press(device, key, False)
            n = int(seconds * 50)

            def eye_fn(i):
                pos, yaw = pelvis_pose(env)
                return chase_eye(pos, yaw, i / 50.0, radius=2.0, height=1.0, swing=0.5)

            held = [k for k in keys if k not in ONE_SHOT]
            hud = ["Keyboard teleop (g1_keyboard)"] + ([caption] if caption else [])
            step_and_capture(n, hud, eye_fn, br=[f"keys: {' + '.join(keys) if keys else '-'}"])
            for key in held:
                sl.press(device, key, False)
    clip.close()


# ------------------------------------------------------------------ expert


def run_expert(args, env):
    unwrapped = env.unwrapped
    rng = np.random.default_rng(SEED)
    pool = sl.texture_pool()
    which = os.environ.get("SHOWCASE_EXPERT", "one_hand")
    obj = getattr(args, "object", None)
    dest = getattr(args, "destination", None)
    if which == "bimanual":
        from isaaclab_arena_g1.policy.g1_scripted_bimanual_pick_and_place_policy import (
            G1ScriptedBimanualPickAndPlacePolicy as P,
            G1ScriptedBimanualPickAndPlacePolicyCfg as C,
        )
    else:
        from isaaclab_arena_g1.policy.g1_scripted_one_hand_pick_and_place_policy import (
            G1ScriptedOneHandPickAndPlacePolicy as P,
            G1ScriptedOneHandPickAndPlacePolicyCfg as C,
        )
    cfg_kwargs = {"motion_seed": SEED}
    if obj:
        cfg_kwargs["pick_up_object_name"] = obj
    if dest:
        cfg_kwargs["destination_name"] = dest
    policy = P(C(**cfg_kwargs))
    episodes = int(os.environ.get("SHOWCASE_EPISODES", "2"))
    start_count = int(os.environ.get("SHOWCASE_DEMO_COUNT", "0"))
    tm = unwrapped.termination_manager
    rigid = list(unwrapped.scene.rigid_objects.keys())
    # User-selected object (blue) and destination (green) are marked with floating arrows.
    marked = os.environ.get("SHOWCASE_MARKERS", "0") == "1"
    world = unwrapped.arena_world
    dest_name = dest or "bowl_ycb_robolab"
    if marked:
        pick_marker = sl.SelectionMarker("Pick", (0.15, 0.5, 1.0))
        place_marker = sl.SelectionMarker("Place", (0.2, 0.9, 0.3))
    scene = LABEL or getattr(args, "kitchen_background", "") or "scene"
    clip = Clip(env)
    recorded = start_count

    with torch.inference_mode():
        obs, _ = env.reset()
    for ep in range(episodes):
        # The policy's reset clears tensors created inside inference mode, so it must run there too.
        with torch.inference_mode():
            policy.reset()
        looks = []
        for name in rigid:
            if name == obj and os.environ.get("SHOWCASE_RECOLOR_OBJECT", "1") == "0":
                continue  # keep the user-chosen object's own look
            looks.append(f"{name}: {sl.recolor_prim(f'/World/envs/env_0/{name}', rng, pool)}")
        kitchen = getattr(args, "kitchen_background", None)
        if kitchen and os.environ.get("SHOWCASE_RECOLOR_COUNTER", "1") == "1":
            counter = f"/World/envs/env_0/{kitchen}/counter_right_main_group"
            looks.append(f"counter: {sl.recolor_prim(counter, rng, pool, p_texture=0.7)}")
        print("[showcase] looks:", looks, flush=True)
        light = sl.randomize_light(rng)
        clip.meta["events"].append({"episode": ep, "start_frame": clip.enc.n, "looks": looks, "light": light})
        # Hold a moment on the freshly randomized scene.
        steps = 0
        outcome = None
        while outcome is None:
            pos, yaw = pelvis_pose(env)
            if getattr(args, "kitchen_background", None) is None:
                # Galileo: a wall sits right behind the start pose, so film shelf and bin from a fixed wide view.
                eye, tgt = (-0.6, 2.3, 1.15), (0.1, -0.7, -0.45)
            else:
                eye, tgt = chase_eye(pos, yaw, clip.step_i / 50.0)
            sl.look_at(env, clamp_to_room(eye, args), tgt)
            if marked:
                phase_now = policy.phase.value if policy.phase is not None else "settle"
                t_sim = steps * unwrapped.step_dt
                obj_v = world.get_vertices_w(obj)[0].cpu().numpy()
                dst_v = np.asarray(policy._vertices_w(unwrapped, dest_name))  # also handles articulations
                pick_marker.place((*obj_v[:, :2].mean(axis=0), obj_v[:, 2].max() + 0.09), t_sim)
                place_marker.place((*dst_v[:, :2].mean(axis=0), dst_v[:, 2].max() + 0.09), t_sim + 1.0)
                pick_marker.show(phase_now in ("settle", "approach", "reach_side", "reach_forward", "contact"))
                place_marker.show(phase_now not in ("release", "open_away", "retract", "stand", "lower", "done"))
            with torch.inference_mode():
                action = policy.get_action(env, obs)
                obs, _, term, trunc, _ = env.step(action)
            steps += 1
            done = bool(term.any()) or bool(trunc.any())
            if done:
                fired = [n for n, v in zip(tm.active_terms, tm._last_episode_dones[0].tolist()) if v]
                outcome = "success" if "success" in fired else "fail:" + ",".join(fired)
            if steps % 2 == 0 or done:
                phase = policy.phase.value if policy.phase is not None else "-"
                tl = [
                    f"Auto data generation  |  episode {ep + 1}",
                    f"scene: {scene}",
                    f"object: {obj or 'brown_box'}",
                    f"phase: {phase}",
                ]
                if marked:
                    tl[2:3] = [f"pick (blue arrow): {obj}", f"place (green arrow): {dest_name}"]
                br = [f"t = {steps * unwrapped.step_dt:5.1f} s", f"recorded demos: {recorded}"]
                clip.frame(tl=tl, br=br)
            clip.step_i += 1
        if outcome == "success":
            recorded += 1
            badge = (f"SUCCESS  ->  saved demo_{recorded - 1} to HDF5", (20, 140, 60, 220))
        else:
            badge = ("FAILED  ->  attempt discarded", (170, 40, 40, 220))
        clip.meta["events"][-1].update(outcome=outcome, end_frame=clip.enc.n, sim_seconds=steps * unwrapped.step_dt)
        print(f"[showcase] episode {ep}: {outcome} after {steps * unwrapped.step_dt:.1f}s", flush=True)
        # Freeze-frame with the outcome badge for 1.5 s (the env already reset, so reuse the last frame).
        last = sl.grab(env)
        for _ in range(int(1.5 * FPS)):
            clip.enc.write(sl.overlay(sl.compose_split(last, sl.grab(env, "robot_head_cam")), badge=badge))
        if outcome == "success" and os.environ.get("SHOWCASE_STOP_AFTER_SUCCESS", "0") == "1":
            break
    clip.meta["recorded"] = recorded
    clip.close()


if __name__ == "__main__":
    main()
