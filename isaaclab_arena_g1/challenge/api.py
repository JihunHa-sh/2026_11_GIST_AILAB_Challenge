# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Everything a g1_fast_transport policy can read, and helpers to build its action.

Call these from your policy's ``get_action(env, observation)``. Conventions, for every function here:

- Tensors have a leading environment dimension (always 1 in the challenge): a pose is ``(1, 7)``.
- Poses are ``(x, y, z, qx, qy, qz, qw)``: position [m] and quaternion in (x, y, z, w) order.
- "env frame" is the simulation world shifted to this environment's origin: z = 0 is the kitchen floor, the counter
  front edge is at y = -0.65, the robot starts facing +y. With one environment it equals the world frame.
- "pelvis frame" is attached to the robot's pelvis (x forward, y left, z up); the action's wrist targets use it.
- Angles are radians, times seconds.

Reading the simulator's state (object pose, robot links, furniture) is allowed; changing it is not.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym

ACTION_DIM = 23
"""Size of the ``g1_wbc_pink`` action."""

SIDES = ("left", "right")

REST_WRIST_POS = {"left": (0.03, 0.20, -0.07), "right": (0.03, -0.20, -0.07)}
"""Wrist positions [m] in the pelvis frame with the arms hanging at the sides."""

REST_WRIST_QUAT_XYZW = (0.0, math.sin(math.pi / 4.0), 0.0, math.cos(math.pi / 4.0))
"""Wrist orientation (pelvis frame) with the fingers pointing down."""

STANDING_PELVIS_HEIGHT = 0.74
"""Pelvis height command [m] for normal standing and walking."""


# ---------------------------------------------------------------------------------------------------------------------
# The challenge observation (no simulator access needed).
# ---------------------------------------------------------------------------------------------------------------------


def read_challenge_observation(observation: dict) -> dict[str, torch.Tensor]:
    """Split ``observation["challenge"]`` (shape (1, 22)) into named tensors, all in the env frame.

    Returns:
        ``object_pos`` (1, 3), ``object_quat`` (1, 4), ``goal_pos`` (1, 3), ``goal_quat`` (1, 4), ``pelvis_pos``
        (1, 3), ``pelvis_quat`` (1, 4) and ``time`` (1,), the simulated time since the episode started [s].
    """
    obs = observation["challenge"]
    return {
        "object_pos": obs[:, 0:3],
        "object_quat": obs[:, 3:7],
        "goal_pos": obs[:, 7:10],
        "goal_quat": obs[:, 10:14],
        "pelvis_pos": obs[:, 14:17],
        "pelvis_quat": obs[:, 17:21],
        "time": obs[:, 21],
    }


# ---------------------------------------------------------------------------------------------------------------------
# Object and goal.
# ---------------------------------------------------------------------------------------------------------------------


def object_name(env: gym.Env) -> str:
    """Scene name of the object to carry (``bbq_sauce_bottle_hope_robolab`` by default)."""
    return _state(env).object_name or _task_object_name(env)


def object_pose(env: gym.Env) -> torch.Tensor:
    """Object root pose in the env frame, (1, 7)."""
    return _pose_e(env, env.unwrapped.scene[object_name(env)].data.root_pose_w.torch)


def object_velocity(env: gym.Env) -> torch.Tensor:
    """Object linear [m/s] and angular [rad/s] velocity in the world frame, (1, 6)."""
    data = env.unwrapped.scene[object_name(env)].data
    return torch.cat([data.root_lin_vel_w.torch, data.root_ang_vel_w.torch], dim=-1)


def object_corners(env: gym.Env) -> torch.Tensor:
    """Corners of the object's (oriented) bounding box in the env frame, (1, 8, 3)."""
    return _corners_e(env, object_name(env))


def goal_pose(env: gym.Env) -> torch.Tensor:
    """Goal pose of the object's root in the env frame, (1, 7): upright on the desk, turned 180 deg from the start."""
    state = _state(env)
    return torch.cat([state.goal_pos_e, state.goal_quat], dim=-1)


def goal_error(env: gym.Env) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance [m] and rotation angle [rad] between the object and the goal, each (1,).

    The episode succeeds once both stay below 0.03 m and 15 deg, with the object nearly still, for 0.5 s.
    """
    from isaaclab.utils.math import quat_error_magnitude

    pose = object_pose(env)
    goal = goal_pose(env)
    return (pose[:, :3] - goal[:, :3]).norm(dim=-1), quat_error_magnitude(pose[:, 3:7], goal[:, 3:7])


# ---------------------------------------------------------------------------------------------------------------------
# Robot.
# ---------------------------------------------------------------------------------------------------------------------


def pelvis_pose(env: gym.Env) -> torch.Tensor:
    """Pelvis pose in the env frame, (1, 7)."""
    return body_pose(env, "pelvis")


def pelvis_velocity(env: gym.Env) -> torch.Tensor:
    """Pelvis linear [m/s] and angular [rad/s] velocity in the world frame, (1, 6)."""
    robot = env.unwrapped.scene["robot"]
    index = robot.body_names.index("pelvis")
    return robot.data.body_link_vel_w.torch[:, index]


def body_pose(env: gym.Env, body_name: str) -> torch.Tensor:
    """Pose of one robot link in the env frame, (1, 7); see ``body_names`` for the names (e.g. ``right_hand_palm_link``)."""
    robot = env.unwrapped.scene["robot"]
    return _pose_e(env, robot.data.body_link_pose_w.torch[:, robot.body_names.index(body_name)])


def body_names(env: gym.Env) -> list[str]:
    """Names of the robot's 53 links."""
    return list(env.unwrapped.scene["robot"].body_names)


def wrist_pose_in_pelvis(env: gym.Env, side: str) -> torch.Tensor:
    """Current pose of the ``left`` or ``right`` wrist (``*_wrist_yaw_link``) in the pelvis frame, (1, 7).

    This is the frame of the action's wrist targets, so it tells how far the hand still is from its target.
    """
    assert side in SIDES, f"side must be one of {SIDES}, got {side}"
    return env_to_pelvis_pose(env, body_pose(env, f"{side}_wrist_yaw_link"))


def joint_positions(env: gym.Env) -> torch.Tensor:
    """Joint positions [rad] of the robot's 43 joints, (1, 43); order as in ``joint_names``."""
    return env.unwrapped.scene["robot"].data.joint_pos.torch


def joint_velocities(env: gym.Env) -> torch.Tensor:
    """Joint velocities [rad/s], (1, 43); order as in ``joint_names``."""
    return env.unwrapped.scene["robot"].data.joint_vel.torch


def joint_names(env: gym.Env) -> list[str]:
    """Names of the robot's 43 joints (legs, waist, arms, Dex3 fingers)."""
    return list(env.unwrapped.scene["robot"].joint_names)


# ---------------------------------------------------------------------------------------------------------------------
# Furniture.
# ---------------------------------------------------------------------------------------------------------------------


def desk_corners(env: gym.Env) -> torch.Tensor:
    """Corners of the desk's bounding box (top and legs, floor to top) in the env frame, (1, 8, 3)."""
    return _corners_e(env, "transport_table")


def counter_top_corners(env: gym.Env) -> torch.Tensor:
    """Corners of the kitchen counter top the object starts on, in the env frame, (1, 8, 3)."""
    return _corners_e(env, "kitchen_counter_top")


# ---------------------------------------------------------------------------------------------------------------------
# Time.
# ---------------------------------------------------------------------------------------------------------------------


def elapsed_time(env: gym.Env) -> torch.Tensor:
    """Simulated time since the episode started [s], (1,); this is what the challenge scores."""
    unwrapped = env.unwrapped
    return unwrapped.episode_length_buf.float() * unwrapped.step_dt


def control_dt(env: gym.Env) -> float:
    """Time between two ``get_action`` calls [s] (0.02, i.e. 50 Hz)."""
    return float(env.unwrapped.step_dt)


def time_limit(env: gym.Env) -> float:
    """Episode time limit [s] (90)."""
    return float(env.unwrapped.max_episode_length_s)


# ---------------------------------------------------------------------------------------------------------------------
# Frames and angles.
# ---------------------------------------------------------------------------------------------------------------------


def env_to_pelvis_points(env: gym.Env, points_e: torch.Tensor) -> torch.Tensor:
    """Express env-frame points (1, ..., 3) in the current pelvis frame, e.g. to turn an object position into a
    wrist target."""
    from isaaclab.utils.math import quat_apply_inverse

    pelvis = pelvis_pose(env)
    shape = points_e.shape
    flat = points_e.reshape(shape[0], -1, 3)
    quat = pelvis[:, None, 3:7].expand(-1, flat.shape[1], -1)
    return quat_apply_inverse(quat, flat - pelvis[:, None, :3]).reshape(shape)


def env_to_pelvis_pose(env: gym.Env, pose_e: torch.Tensor) -> torch.Tensor:
    """Express an env-frame pose (1, 7) in the current pelvis frame."""
    from isaaclab.utils.math import quat_inv, quat_mul

    pelvis = pelvis_pose(env)
    position = env_to_pelvis_points(env, pose_e[:, :3])
    return torch.cat([position, quat_mul(quat_inv(pelvis[:, 3:7]), pose_e[:, 3:7])], dim=-1)


def yaw_of(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Heading (rotation about the vertical) of (x, y, z, w) quaternions [rad]."""
    x, y, z, w = quat_xyzw.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------------------------------------------------
# Building the action.
# ---------------------------------------------------------------------------------------------------------------------


def make_action(
    env: gym.Env,
    left_wrist_pos=REST_WRIST_POS["left"],
    left_wrist_quat=REST_WRIST_QUAT_XYZW,
    right_wrist_pos=REST_WRIST_POS["right"],
    right_wrist_quat=REST_WRIST_QUAT_XYZW,
    left_hand_closed: bool = False,
    right_hand_closed: bool = False,
    walk_forward: float = 0.0,
    walk_left: float = 0.0,
    turn_rate: float = 0.0,
    pelvis_height: float = STANDING_PELVIS_HEIGHT,
    torso_rpy=(0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Build the 23-D ``g1_wbc_pink`` action, (1, 23).

    The arms follow wrist targets in the pelvis frame (positions [m], (x, y, z, w) quaternions) through inverse
    kinematics; the legs follow a learned walking policy driven by ``walk_forward`` / ``walk_left`` [m/s] and
    ``turn_rate`` [rad/s] (pelvis frame) and ``pelvis_height`` [m]. Defaults: standing still, arms down, hands open.
    Inputs may be floats/tuples or tensors.

    Layout: [0] left hand closed, [1] right hand closed, [2:5] left wrist pos, [5:9] left wrist quat,
    [9:12] right wrist pos, [12:16] right wrist quat, [16:19] walk forward/left/turn, [19] pelvis height,
    [20:23] torso roll/pitch/yaw.
    """
    device = env.unwrapped.device

    def vec(value, size):
        return torch.as_tensor(value, dtype=torch.float32, device=device).reshape(-1)[:size]

    action = torch.zeros(1, ACTION_DIM, device=device)
    action[0, 0] = float(left_hand_closed)
    action[0, 1] = float(right_hand_closed)
    action[0, 2:5] = vec(left_wrist_pos, 3)
    action[0, 5:9] = vec(left_wrist_quat, 4)
    action[0, 9:12] = vec(right_wrist_pos, 3)
    action[0, 12:16] = vec(right_wrist_quat, 4)
    action[0, 16:19] = torch.stack([vec(walk_forward, 1)[0], vec(walk_left, 1)[0], vec(turn_rate, 1)[0]])
    action[0, 19] = vec(pelvis_height, 1)[0]
    action[0, 20:23] = vec(torso_rpy, 3)
    return action


# ---------------------------------------------------------------------------------------------------------------------
# Internals.
# ---------------------------------------------------------------------------------------------------------------------


def _state(env: gym.Env):
    from isaaclab_arena_g1.challenge.fast_transport_task import FAST_TRANSPORT_STATE_ATTR

    state = getattr(env.unwrapped, FAST_TRANSPORT_STATE_ATTR, None)
    assert state is not None, "Not a g1_fast_transport environment, or it has not been reset yet."
    return state


def _task_object_name(env: gym.Env) -> str:
    # Before the first success check runs, find the object as the scene's only graspable rigid object.
    unwrapped = env.unwrapped
    candidates = [name for name in unwrapped.scene.rigid_objects.keys() if name != "transport_table"]
    assert len(candidates) == 1, f"Cannot tell the object to carry among {candidates}."
    return candidates[0]


def _pose_e(env: gym.Env, pose_w: torch.Tensor) -> torch.Tensor:
    return torch.cat([pose_w[..., :3] - env.unwrapped.scene.env_origins, pose_w[..., 3:7]], dim=-1)


def _corners_e(env: gym.Env, name: str) -> torch.Tensor:
    unwrapped = env.unwrapped
    return unwrapped.arena_world.get_vertices_w(name) - unwrapped.scene.env_origins[:, None, :]
