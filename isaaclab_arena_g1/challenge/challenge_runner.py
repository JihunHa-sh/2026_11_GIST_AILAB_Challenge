# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Score a policy on the g1_fast_transport challenge: one episode per seed, timed in simulated and wall-clock time.

Example::

    python isaaclab_arena_g1/challenge/challenge_runner.py \\
        --policy_type isaaclab_arena_g1.challenge.template_policy.G1FastTransportTemplatePolicy \\
        --num_seeds 10 --team_name my_team g1_fast_transport

Each seed fixes the robot's start offset, so every team is scored on the same episodes. The results are printed as a
table and written to ``<output_dir>/<team>_<timestamp>/results.json`` and ``results.csv``; ``--video`` also writes
one viewport video per episode (rendering slows the run down, so wall-clock times are only comparable without it).

Score (lower is better): the mean simulated time over all episodes, where a failed episode counts as the time limit.
Teams are ranked by it; ties are broken by the success count, then by the mean position error.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import time
from typing import TYPE_CHECKING

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.policy_runner import get_policy_cls
from isaaclab_arena.evaluation.policy_runner_cli import (
    add_policy_cli_args,
    add_policy_runner_arguments,
    build_policy_from_cli,
)
from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    from isaaclab_arena.policy.policy_base import PolicyBase

CHALLENGE_ENVIRONMENT = "g1_fast_transport"
TERMINATION_REASONS = ("success", "robot_fell", "object_dropped", "policy_gave_up", "time_out")
"""Termination terms of the challenge, in the order they are reported when several fire on the same step."""

VIDEO_FPS = 25
VIDEO_CAMERA = "challenge_video_cam"
VIDEO_SIZE = (960, 540)


def add_challenge_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the challenge runner's own command-line options."""
    group = parser.add_argument_group("Challenge runner")
    group.add_argument(
        "--num_seeds", type=int, default=10, help="Number of episodes, seeds first_seed, first_seed+1..."
    )
    group.add_argument("--first_seed", type=int, default=0, help="Seed of the first episode.")
    group.add_argument(
        "--seeds", type=str, default=None, help="Explicit comma-separated seeds, e.g. 3,17,42; overrides --num_seeds."
    )
    group.add_argument("--team_name", type=str, default="team", help="Name written into the results.")
    group.add_argument(
        "--challenge_output_dir", type=str, default="challenge_results", help="Directory the results are written to."
    )
    group.add_argument("--video", action="store_true", help="Record one viewport video per episode.")
    group.add_argument(
        "--video_eye", type=float, nargs=3, default=None, help="Video camera position (env frame); default: viewport."
    )
    group.add_argument(
        "--video_lookat", type=float, nargs=3, default=None, help="Video camera target (env frame); default: viewport."
    )


def add_video_camera(env_cfg) -> None:
    """Add a fixed RGB scene camera for the episode videos (the viewport cannot be read back in this Isaac Lab)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    # The camera must live under the environment namespace, or it renders blank.
    setattr(
        env_cfg.scene,
        VIDEO_CAMERA,
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/ChallengeVideoCam",
            update_period=0.0,
            width=VIDEO_SIZE[0],
            height=VIDEO_SIZE[1],
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.05, 40.0)),
        ),
    )


def aim_video_camera(env, eye_e=None, lookat_e=None) -> None:
    """Point the video camera from ``eye_e`` toward ``lookat_e`` (environment frame), by default like the viewport."""
    import torch

    unwrapped = env.unwrapped
    # Arena moves the task's viewer pose into the Kit visualizer config.
    viewer = unwrapped.cfg.sim.default_visualizer_cfg
    eye_e = viewer.eye if eye_e is None else eye_e
    lookat_e = viewer.lookat if lookat_e is None else lookat_e
    origin = unwrapped.scene.env_origins[0]
    eye = torch.tensor([eye_e], dtype=torch.float32, device=unwrapped.device) + origin
    target = torch.tensor([lookat_e], dtype=torch.float32, device=unwrapped.device) + origin
    unwrapped.scene[VIDEO_CAMERA].set_world_poses_from_view(eye, target)


def grab_video_frame(env):
    """Return the video camera's latest RGB image as a (H, W, 3) uint8 array."""
    import numpy as np

    rgb = env.unwrapped.scene[VIDEO_CAMERA].data.output["rgb"]
    rgb = rgb.torch if hasattr(rgb, "torch") else rgb
    frame = rgb[0, ..., :3].cpu().numpy()
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return frame.copy()


def _object_quat(env):
    from isaaclab_arena_g1.challenge.fast_transport_task import FAST_TRANSPORT_STATE_ATTR

    object_name = getattr(env.unwrapped, FAST_TRANSPORT_STATE_ATTR).object_name
    return env.unwrapped.scene[object_name].data.root_pose_w.torch[0, 3:7]


def yaw_error_deg(env, state) -> float:
    """Object yaw minus goal yaw [deg], wrapped to [-180, 180)."""
    from isaaclab.utils.math import yaw_quat

    def yaw_of(q):
        x, y, z, w = yaw_quat(q).tolist()
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    error = yaw_of(_object_quat(env)) - yaw_of(state.goal_quat[0])
    return math.degrees((error + math.pi) % (2.0 * math.pi) - math.pi)


def pelvis_xy_yaw(env) -> str:
    """Pelvis x, y [m] and heading [deg] in the environment frame, formatted."""
    robot = env.unwrapped.scene["robot"]
    pose = robot.data.body_link_pose_w.torch[0, robot.body_names.index("pelvis")]
    x, y = (pose[:2] - env.unwrapped.scene.env_origins[0, :2]).tolist()
    qx, qy, qz, qw = pose[3:7].tolist()
    yaw = math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    return f"({x:5.2f}, {y:5.2f}, {yaw:6.1f} deg)"


def object_dz(env, state) -> float:
    """Object height minus goal height [m]."""
    object_z = (
        env.unwrapped.scene[state.object_name].data.root_pose_w.torch[0, 2] - env.unwrapped.scene.env_origins[0, 2]
    )
    return float(object_z - state.goal_pos_e[0, 2])


def tilt_deg(env) -> float:
    """Angle between the object's up axis and the vertical [deg]."""
    x, y, z, w = _object_quat(env).tolist()
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


def episode_seeds(args_cli: argparse.Namespace) -> list[int]:
    """Return the seeds to run, one episode each."""
    if args_cli.seeds is not None:
        return [int(seed) for seed in args_cli.seeds.split(",") if seed.strip()]
    return list(range(args_cli.first_seed, args_cli.first_seed + args_cli.num_seeds))


def termination_reason(env) -> str:
    """Name of the termination term that ended the last step's episode."""
    manager = env.unwrapped.termination_manager
    for name in TERMINATION_REASONS:
        if name in manager.active_terms and bool(manager.get_term(name)[0]):
            return name
    for name in manager.active_terms:
        if bool(manager.get_term(name)[0]):
            return name
    return "unknown"


def run_episode(env, policy: PolicyBase, seed: int, video_path: str | None, video_view: tuple | None = None) -> dict:
    """Run one episode from ``seed`` and return its result row.

    Policies with a ``phase`` attribute (the scripted experts) get their phase changes printed with the sim time.
    ``video_view`` is an optional (eye, lookat) pair for the video camera.
    """
    import numpy as np
    import torch

    from isaaclab_arena_g1.challenge.fast_transport_task import FAST_TRANSPORT_STATE_ATTR, START_OFFSET_SEED_ATTR

    unwrapped = env.unwrapped
    setattr(unwrapped, START_OFFSET_SEED_ATTR, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # The environment's buffers are inference tensors once it has stepped, so resets run in inference mode too.
    with torch.inference_mode():
        obs, _ = env.reset(seed=seed)
        policy.reset()
    policy.set_task_description(unwrapped.get_language_instruction())
    state = getattr(unwrapped, FAST_TRANSPORT_STATE_ATTR)
    start_offset = state.start_offset[0].tolist()

    frames = []
    render_every = max(1, round(1.0 / (VIDEO_FPS * unwrapped.step_dt)))
    steps = 0
    last_phase = None
    policy_time = 0.0
    wall_start = time.perf_counter()
    while True:
        if video_path is not None:
            # Camera poses are written outside inference mode, every step, or the camera keeps its spawn pose.
            aim_video_camera(env, *(video_view or (None, None)))
        with torch.inference_mode():
            policy_start = time.perf_counter()
            actions = policy.get_action(env, obs)
            policy_time += time.perf_counter() - policy_start
            obs, _, terminated, truncated, _ = env.step(actions)
        steps += 1
        phase = getattr(policy, "phase", None)
        if phase is not None and phase != last_phase:
            print(
                f"[challenge]   t={steps * unwrapped.step_dt:6.2f} s  phase {getattr(phase, 'value', phase):<14}"
                f" object-goal error {100.0 * float(state.position_error[0]):6.1f} cm"
                f" {math.degrees(float(state.orientation_error[0])):6.1f} deg"
                f" (yaw {yaw_error_deg(env, state):6.1f}, tilt {tilt_deg(env):5.1f})"
                f" dz {100.0 * object_dz(env, state):5.1f} cm | pelvis {pelvis_xy_yaw(env)}",
                flush=True,
            )
            last_phase = phase
        done = bool(terminated[0]) or bool(truncated[0])
        # After the last step the environment has already reset, so its frame would show the next start.
        if video_path is not None and not done and steps % render_every == 0:
            frames.append(grab_video_frame(env))
        if done:
            break
    wall_time = time.perf_counter() - wall_start

    reason = termination_reason(env)
    if video_path is not None and frames:
        import imageio

        imageio.mimsave(video_path, frames, fps=VIDEO_FPS)
    return {
        "seed": seed,
        "success": reason == "success",
        "termination": reason,
        "sim_time_s": round(steps * unwrapped.step_dt, 3),
        "steps": steps,
        "wall_time_s": round(wall_time, 3),
        "policy_time_s": round(policy_time, 3),
        "final_position_error_m": round(float(state.position_error[0]), 4),
        "final_orientation_error_deg": round(math.degrees(float(state.orientation_error[0])), 2),
        "start_offset_x_m": round(start_offset[0], 4),
        "start_offset_y_m": round(start_offset[1], 4),
        "video": None if video_path is None else os.path.basename(video_path),
    }


def summarize(rows: list[dict], time_limit_s: float) -> dict:
    """Aggregate the episode rows into the challenge score."""
    successes = [row for row in rows if row["success"]]
    penalized = [row["sim_time_s"] if row["success"] else time_limit_s for row in rows]
    finite_errors = [row["final_position_error_m"] for row in rows if not math.isnan(row["final_position_error_m"])]
    return {
        "episodes": len(rows),
        "successes": len(successes),
        "success_rate": round(len(successes) / len(rows), 3) if rows else 0.0,
        "score_s": round(sum(penalized) / len(penalized), 3) if rows else None,
        "mean_success_sim_time_s": (
            round(sum(row["sim_time_s"] for row in successes) / len(successes), 3) if successes else None
        ),
        "best_sim_time_s": min((row["sim_time_s"] for row in successes), default=None),
        "mean_final_position_error_m": round(sum(finite_errors) / len(finite_errors), 4) if finite_errors else None,
        "total_wall_time_s": round(sum(row["wall_time_s"] for row in rows), 1),
        "time_limit_s": time_limit_s,
    }


def print_table(rows: list[dict], summary: dict, team_name: str) -> None:
    """Print the episode table and the summary."""
    header = f"{'seed':>5} {'result':>15} {'sim[s]':>8} {'wall[s]':>8} {'pos err[cm]':>11} {'rot err[deg]':>12}"
    print(f"\n=== g1_fast_transport | team: {team_name} ===")
    print(header)
    print("-" * len(header))
    for row in rows:
        position_error_cm = row["final_position_error_m"] * 100.0
        print(
            f"{row['seed']:>5} {row['termination']:>15} {row['sim_time_s']:>8.2f} {row['wall_time_s']:>8.1f}"
            f" {position_error_cm:>11.1f} {row['final_orientation_error_deg']:>12.1f}"
        )
    print("-" * len(header))
    print(
        f"success {summary['successes']}/{summary['episodes']} | score {summary['score_s']} s (lower is better;"
        f" failure = {summary['time_limit_s']:.0f} s) | best {summary['best_sim_time_s']} s | mean success time"
        f" {summary['mean_success_sim_time_s']} s\n"
    )


def write_results(output_dir: str, rows: list[dict], summary: dict, metadata: dict) -> None:
    """Write ``results.json`` (metadata, summary, episodes) and ``results.csv`` (episodes)."""
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({**metadata, "summary": summary, "episodes": rows}, f, indent=2)
    with open(os.path.join(output_dir, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Score a policy on the fast-transport challenge."""
    args_parser = get_isaaclab_arena_cli_parser()
    add_challenge_arguments(args_parser)
    args_cli, _ = args_parser.parse_known_args()
    if args_cli.video:
        args_cli.enable_cameras = True

    with SimulationAppContext(args_cli):
        add_policy_runner_arguments(args_parser)
        args_cli, _ = args_parser.parse_known_args()
        assert args_cli.policy_type is not None, "--policy_type is required."
        policy_cls = get_policy_cls(args_cli.policy_type)

        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_parser = add_policy_cli_args(args_parser, policy_cls)
        args_cli, hydra_overrides = args_parser.parse_known_args()
        assert_hydra_overrides(hydra_overrides, args_parser)
        if args_cli.video:
            args_cli.enable_cameras = True
        assert args_cli.num_envs == 1, "The challenge runs one environment at a time (the WBC Pink action needs it)."

        arena_builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)
        assert (
            arena_builder.arena_env.name == CHALLENGE_ENVIRONMENT
        ), f"challenge_runner scores the {CHALLENGE_ENVIRONMENT} environment, got {arena_builder.arena_env.name}."
        env_cfg, env_kwargs = arena_builder.compose_manager_cfg()
        if args_cli.video:
            add_video_camera(env_cfg)
        env = arena_builder.make_registered(env_cfg, env_kwargs)
        policy = build_policy_from_cli(policy_cls, args_cli)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(args_cli.challenge_output_dir, f"{args_cli.team_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        rows = []
        for seed in episode_seeds(args_cli):
            video_path = os.path.join(output_dir, f"seed_{seed:04d}.mp4") if args_cli.video else None
            row = run_episode(env, policy, seed, video_path, (args_cli.video_eye, args_cli.video_lookat))
            rows.append(row)
            print(
                f"[challenge] seed {seed}: {row['termination']} after {row['sim_time_s']:.2f} s sim"
                f" ({row['wall_time_s']:.1f} s wall)",
                flush=True,
            )

        time_limit_s = float(env.unwrapped.max_episode_length_s)
        summary = summarize(rows, time_limit_s)
        metadata = {
            "challenge": CHALLENGE_ENVIRONMENT,
            "team_name": args_cli.team_name,
            "policy_type": args_cli.policy_type,
            "timestamp": timestamp,
            "seeds": [row["seed"] for row in rows],
            "control_dt_s": env.unwrapped.step_dt,
            "video": bool(args_cli.video),
        }
        print_table(rows, summary, args_cli.team_name)
        write_results(output_dir, rows, summary, metadata)
        print(f"[challenge] Results written to {os.path.abspath(output_dir)}")

        if policy.is_remote:
            policy.shutdown_remote(kill_server=args_cli.remote_kill_on_exit)
        env.close()


if __name__ == "__main__":
    main()
