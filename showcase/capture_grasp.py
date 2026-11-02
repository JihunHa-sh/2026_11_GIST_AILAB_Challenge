"""Capture a one-hand expert grasp for the grasp-planning video: sim frames plus everything the planners decided.

Runs scripted one-hand episodes until one lifts the object. For that episode it writes
  <out>.mp4   frames of the episode (chase camera while walking, close-up camera on the object from REACH_SIDE on)
  <out>.json  per-frame phase and camera pose, each grasp-planner call (its inputs and returned grasps) and each
              cuRobo pre-grasp plan attempt (candidate index, success, failure reason), plus the chosen grasp
  <out>.npz   object mesh (visual and convex hull, object frame) and support-surface vertices
``render_grasp_steps.py`` re-runs the grasp planner on the recorded inputs to draw the planning steps.
Env vars: SHOWCASE_OUT, SHOWCASE_SEED, SHOWCASE_EPISODES (max tries). The rest is the usual Arena env CLI.
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

OUT = os.environ["SHOWCASE_OUT"]
SEED = int(os.environ.get("SHOWCASE_SEED", "0"))
MAX_EPISODES = int(os.environ.get("SHOWCASE_EPISODES", "4"))
FPS = 25
CLOSE_UP_PHASES = ("reach_side", "reach_forward", "contact", "close", "lift")
FOCAL_LENGTH = 14.0
HORIZONTAL_APERTURE = 20.955


def main():
    parser = get_isaaclab_arena_cli_parser()
    parser.set_defaults(device="cpu")
    args, _ = parser.parse_known_args(sys.argv[1:])
    with SimulationAppContext(args):
        try:
            run(parser)
        except Exception:
            traceback.print_exc()


def pelvis_pose(env):
    p = env.unwrapped.arena_world.get_body_pose_w("robot", "pelvis")[0].cpu().numpy().astype(float)
    x, y, z, w = p[3:7]
    return p[:3], math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def close_up(obj_pos):
    """Camera on the object from the robot's right-front side (the robot faces +y, the right hand is at +x).

    Kept within 0.3 m of the object in x: a fridge stands right of the counter in the one_wall kitchens."""
    o = np.asarray(obj_pos, dtype=float)
    return tuple(o + np.array([0.28, -0.72, 0.36])), tuple(o + np.array([-0.06, -0.05, -0.04]))


def chase(pos, yaw):
    ang = yaw + math.pi - 0.5
    eye = (pos[0] + 1.9 * math.cos(ang), pos[1] + 1.9 * math.sin(ang), 1.7)
    tgt = (pos[0] + 0.55 * math.cos(yaw), pos[1] + 0.55 * math.sin(yaw), 0.85)
    return (min(max(eye[0], 0.25), 5.25), min(max(eye[1], -2.8), -0.75), eye[2]), tgt


def grasp_dict(g):
    return {"wrist_pos": g.wrist_pos.tolist(), "wrist_quat_xyzw": g.wrist_quat_xyzw.tolist(),
            "approach": g.approach.tolist(), "width": float(g.width), "score": float(g.score), "kind": g.kind}


class Recorder:
    """Wraps the expert's grasp planner and arm planner and logs what they return."""

    def __init__(self, policy):
        self.policy = policy
        self.grasp_calls = []
        self.arm_calls = []
        self.step = 0
        expert = policy.expert
        scene_planner = expert.grasp_planner
        self.hull = scene_planner.planner.mesh
        plan_grasps = scene_planner.plan

        def logged_grasp_plan(state, side, heading=None):
            grasps = plan_grasps(state, side, heading=heading)
            planner = scene_planner.planner
            support_height = float(np.asarray(state.support_vertices)[:, 2].max())
            self.support_vertices = np.asarray(state.support_vertices)
            self.grasp_calls.append({
                "step": self.step, "phase": policy.phase.value, "side": side,
                "heading": None if heading is None else float(heading),
                "object_pos": np.asarray(state.object_pos).tolist(),
                "object_quat_xyzw": np.asarray(state.object_quat_xyzw).tolist(),
                "support_height": support_height,
                "pelvis_pos": np.asarray(state.pelvis_pos).tolist(),
                "pelvis_quat_xyzw": np.asarray(state.pelvis_quat_xyzw).tolist(),
                "stats": dict(planner.last_stats),
                "grasps": [grasp_dict(g) for g in grasps],
            })
            return grasps

        scene_planner.plan = logged_grasp_plan
        if expert.planner is not None:
            plan_arm = expert.planner.plan

            def logged_arm_plan(state, goals, ignore, sides=("left", "right")):
                trajectory = plan_arm(state, goals, ignore, sides)
                inner = getattr(expert.planner, "planner", None)
                self.arm_calls.append({
                    "step": self.step, "phase": policy.phase.value, "sides": list(sides),
                    "ok": trajectory is not None,
                    "failure": "" if trajectory is not None else str(getattr(inner, "last_failure", "")),
                    "waypoints": 0 if trajectory is None else int(len(trajectory)),
                    "goal": {k: [np.asarray(v[0]).tolist(), np.asarray(v[1]).tolist()] for k, v in goals.items()},
                })
                return trajectory

            expert.planner.plan = logged_arm_plan


def run(parser):
    parser = get_isaaclab_arena_environments_cli_parser(parser)
    args, hydra = parser.parse_known_args(sys.argv[1:])
    builder = get_arena_builder_from_cli(args, hydra_overrides=hydra)
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    env_cfg.recorders = {}
    sl.add_showcase_camera(env_cfg, focal_length=FOCAL_LENGTH)
    env = builder.make_registered(env_cfg, env_kwargs)
    unwrapped = env.unwrapped

    from isaaclab_arena_g1.policy.g1_scripted_one_hand_pick_and_place_policy import (
        G1ScriptedOneHandPickAndPlacePolicy as P,
        G1ScriptedOneHandPickAndPlacePolicyCfg as C,
    )

    obj = args.object
    policy = P(C(pick_up_object_name=obj, motion_seed=SEED))
    obj_data = unwrapped.scene[obj].data

    def object_pos():
        return obj_data.root_pose_w.torch[0, :3].cpu().numpy().astype(float)

    with torch.inference_mode():
        obs, _ = env.reset()

    for ep in range(MAX_EPISODES):
        with torch.inference_mode():
            policy.reset()
        tmp = OUT.replace(".mp4", f"_try{ep}.mp4")
        enc = sl.Encoder(tmp, fps=FPS)
        frames = []
        recorder = None
        start_z = float(object_pos()[2])
        lift_end = None
        steps = 0
        done = False
        cam_fixed = None
        while not done:
            with torch.inference_mode():
                action = policy.get_action(env, obs)
            if recorder is None and policy.expert is not None and policy.expert.grasp_planner is not None:
                recorder = Recorder(policy)
            phase = policy.phase.value if policy.phase is not None else "-"
            pos, yaw = pelvis_pose(env)
            if phase in CLOSE_UP_PHASES or cam_fixed is not None and phase not in ("settle", "approach"):
                if cam_fixed is None:
                    cam_fixed = close_up(object_pos())
                eye, tgt = cam_fixed
            else:
                eye, tgt = chase(pos, yaw)
            sl.look_at(env, eye, tgt)
            with torch.inference_mode():
                obs, _, term, trunc, _ = env.step(action)
            steps += 1
            if recorder is not None:
                recorder.step = steps
            done = bool(term.any()) or bool(trunc.any())
            if steps % 2 == 0 and not done:
                frame = sl.grab(env)
                enc.write(frame)
                frames.append({"phase": phase, "step": steps, "eye": list(map(float, eye)), "target": list(map(float, tgt)),
                               "object_pos": object_pos().tolist()})
            if phase in ("back_off", "carry", "place_over", "release", "retract") and lift_end is None:
                lift_end = steps
            if lift_end is not None and steps - lift_end > 50:
                break
            if phase.startswith("retry"):
                break
        enc.close()
        z = float(object_pos()[2])
        lifted = lift_end is not None and z > start_z + 0.04
        print(f"[grasp] try {ep}: lifted={lifted} steps={steps} dz={z - start_z:.3f} "
              f"grasp_calls={0 if recorder is None else len(recorder.grasp_calls)} "
              f"arm_calls={0 if recorder is None else len(recorder.arm_calls)}", flush=True)
        if lifted and recorder is not None and recorder.grasp_calls:
            os.replace(tmp, OUT)
            from isaaclab_arena.assets.registries import AssetRegistry
            from isaaclab_arena.utils.usd.helpers import extract_trimesh_from_usd_path

            asset = AssetRegistry().get_asset_by_name(obj)
            mesh = extract_trimesh_from_usd_path(asset.usd_path, tuple(asset.scale))
            np.savez(OUT.replace(".mp4", ".npz"), mesh_v=mesh.vertices, mesh_f=mesh.faces,
                     hull_v=recorder.hull.vertices, hull_f=recorder.hull.faces,
                     support_vertices=recorder.support_vertices)
            fx = FOCAL_LENGTH / HORIZONTAL_APERTURE * sl.WIDTH
            meta = {"object": obj, "kitchen": args.kitchen_background, "fps": FPS, "width": sl.WIDTH,
                    "height": sl.HEIGHT, "fx": fx, "frames": frames, "grasp_calls": recorder.grasp_calls,
                    "arm_calls": recorder.arm_calls, "chosen_kind": policy.expert.chosen_grasp_kind,
                    "step_dt": unwrapped.step_dt}
            with open(OUT.replace(".mp4", ".json"), "w") as f:
                json.dump(meta, f)
            print(f"[grasp] wrote {OUT} ({len(frames)} frames)", flush=True)
            break
        os.remove(tmp)
        with torch.inference_mode():
            obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    main()
