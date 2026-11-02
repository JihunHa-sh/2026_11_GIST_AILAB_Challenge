# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The two policy layers of the g1_fast_transport challenge and the fixed data passed between them.

Every control step (0.02 s) the main program runs::

    observation --> HighLevelPolicy.compute_command --> Command --> LowLevelPolicy.compute_joint_targets
                                                                     (+ RobotState)        |
                                                         43 joint position targets  <------+--> robot

- ``Command`` (fixed): what the robot should do right now: walk velocity, pelvis height, torso orientation, both
  wrist targets in the pelvis frame and both grips. A high level may plan however it likes inside.
- Low-level output (fixed): position targets for the robot's 43 joints, in ``api.joint_names`` order, tracked by
  the environment's joint PD controllers (fixed gains and torque limits for everyone).

The default low level (``DEFAULT_LOW_LEVEL``) is the environment's built-in controller: a learned walking policy
for the legs, Pink inverse kinematics for the arms and a grip mapping for the Dex3 hands.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gymnasium as gym

COMMAND_DIM = 23
"""Size of a command as a flat tensor (the ``g1_wbc_pink`` action)."""

NUM_JOINTS = 43
"""Number of G1 joints a low level commands."""

DEFAULT_LOW_LEVEL = "default"
"""Value of ``low_level.class`` in a team file that selects the built-in low level."""


@dataclass
class Command:
    """What the high level asks the robot to do this step. Every field has a leading environment dimension (1).

    Poses are in the pelvis frame (x forward, y left, z up); quaternions are (x, y, z, w).
    """

    walk_forward: torch.Tensor
    """Forward walking velocity [m/s], (1,)."""
    walk_left: torch.Tensor
    """Sideways walking velocity, positive to the left [m/s], (1,)."""
    turn_rate: torch.Tensor
    """Turning rate, positive counterclockwise [rad/s], (1,)."""
    pelvis_height: torch.Tensor
    """Pelvis height above the floor [m], (1,); about 0.74 standing."""
    torso_rpy: torch.Tensor
    """Torso roll, pitch, yaw relative to the pelvis [rad], (1, 3)."""
    left_wrist_pos: torch.Tensor
    """Left wrist (``left_wrist_yaw_link``) target position [m], (1, 3)."""
    left_wrist_quat: torch.Tensor
    """Left wrist target orientation, (1, 4)."""
    right_wrist_pos: torch.Tensor
    """Right wrist target position [m], (1, 3)."""
    right_wrist_quat: torch.Tensor
    """Right wrist target orientation, (1, 4)."""
    left_grip: torch.Tensor
    """Left hand: 0 open, 1 closed, (1,)."""
    right_grip: torch.Tensor
    """Right hand: 0 open, 1 closed, (1,)."""

    def to_tensor(self) -> torch.Tensor:
        """Flatten to the 23-D layout of the built-in controller, (1, 23)."""
        return torch.cat(
            [
                self.left_grip.reshape(-1, 1),
                self.right_grip.reshape(-1, 1),
                self.left_wrist_pos,
                self.left_wrist_quat,
                self.right_wrist_pos,
                self.right_wrist_quat,
                self.walk_forward.reshape(-1, 1),
                self.walk_left.reshape(-1, 1),
                self.turn_rate.reshape(-1, 1),
                self.pelvis_height.reshape(-1, 1),
                self.torso_rpy,
            ],
            dim=-1,
        )

    @classmethod
    def from_tensor(cls, action: torch.Tensor) -> Command:
        """Read a command from the 23-D layout, (1, 23); e.g. from ``api.make_action``."""
        assert action.shape[-1] == COMMAND_DIM, f"A command has {COMMAND_DIM} values, got {action.shape[-1]}."
        return cls(
            left_grip=action[:, 0],
            right_grip=action[:, 1],
            left_wrist_pos=action[:, 2:5],
            left_wrist_quat=action[:, 5:9],
            right_wrist_pos=action[:, 9:12],
            right_wrist_quat=action[:, 12:16],
            walk_forward=action[:, 16],
            walk_left=action[:, 17],
            turn_rate=action[:, 18],
            pelvis_height=action[:, 19],
            torso_rpy=action[:, 20:23],
        )


@dataclass
class RobotState:
    """The robot's own (proprioceptive) state, the input of a low level besides the command."""

    joint_pos: torch.Tensor
    """Joint positions [rad], (1, 43), in ``joint_names`` order."""
    joint_vel: torch.Tensor
    """Joint velocities [rad/s], (1, 43)."""
    joint_names: list[str]
    """Names of the 43 joints; also the order of the low level's output."""
    default_joint_pos: torch.Tensor
    """Joint positions of the robot's nominal standing pose [rad], (1, 43)."""
    pelvis_quat: torch.Tensor
    """Pelvis orientation in the world (x, y, z, w), (1, 4), as an IMU would estimate it."""
    pelvis_lin_vel: torch.Tensor
    """Pelvis linear velocity in the pelvis frame [m/s], (1, 3)."""
    pelvis_ang_vel: torch.Tensor
    """Pelvis angular velocity in the pelvis frame [rad/s], (1, 3)."""
    projected_gravity: torch.Tensor
    """Gravity direction in the pelvis frame (unit vector), (1, 3); (0, 0, -1) when upright."""
    dt: float = 0.02
    """Control period [s]."""
    extras: dict = field(default_factory=dict)
    """Room for more signals later; empty now."""

    @classmethod
    def from_env(cls, env: gym.Env) -> RobotState:
        """Read the robot's state from the simulator."""
        from isaaclab.utils.math import quat_apply_inverse

        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        pelvis = robot.body_names.index("pelvis")
        pose = robot.data.body_link_pose_w.torch[:, pelvis]
        velocity = robot.data.body_link_vel_w.torch[:, pelvis]
        quat = pose[:, 3:7]
        gravity = torch.zeros_like(pose[:, :3])
        gravity[:, 2] = -1.0
        return cls(
            joint_pos=robot.data.joint_pos.torch.clone(),
            joint_vel=robot.data.joint_vel.torch.clone(),
            joint_names=list(robot.joint_names),
            default_joint_pos=robot.data.default_joint_pos.torch.clone(),
            pelvis_quat=quat.clone(),
            pelvis_lin_vel=quat_apply_inverse(quat, velocity[:, :3]),
            pelvis_ang_vel=quat_apply_inverse(quat, velocity[:, 3:6]),
            projected_gravity=quat_apply_inverse(quat, gravity),
            dt=float(unwrapped.step_dt),
        )


class HighLevelPolicy:
    """Base class of a high level: plans the task and outputs a ``Command`` every step.

    The constructor takes the ``high_level.options`` of the team file as keyword arguments. Read anything you need
    with ``isaaclab_arena_g1.challenge.api`` (reading only); keep per-episode state on ``self`` and clear it in
    ``reset``. An optional ``phase`` attribute (e.g. the state of your state machine) is printed when it changes.
    """

    def __init__(self, **options):
        self.options = options

    def setup(self, env: gym.Env) -> None:
        """Called once after the environment is built, before the first episode."""

    def reset(self) -> None:
        """Called at the start of every episode."""

    def compute_command(self, env: gym.Env, observation: dict) -> Command:
        """Return this step's command."""
        raise NotImplementedError


class LowLevelPolicy:
    """Base class of a low level: turns a ``Command`` into position targets for the 43 joints every step.

    The constructor takes the ``low_level.options`` of the team file as keyword arguments. By the rules a low level
    uses only the command and the robot's own state (``RobotState``); ``setup`` may read the robot model (joint
    names, limits) from the environment once.
    """

    def __init__(self, **options):
        self.options = options

    def setup(self, env: gym.Env) -> None:
        """Called once after the environment is built, before the first episode."""

    def reset(self) -> None:
        """Called at the start of every episode."""

    def compute_joint_targets(self, command: Command, state: RobotState) -> torch.Tensor:
        """Return joint position targets [rad], (1, 43), in ``state.joint_names`` order."""
        raise NotImplementedError
