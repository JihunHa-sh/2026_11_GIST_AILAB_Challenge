# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Example team low level: replace ``compute_joint_targets`` with your controller (e.g. your own walking policy).

Use it by setting ``low_level.class`` in team.yaml to ``isaaclab_arena_g1.challenge.policies.example_team.low_level.MyLowLevel``.

Input: the high level's ``Command`` and the robot's own state ``RobotState`` (joint positions/velocities, pelvis
orientation and velocities, gravity direction). Output: position targets [rad] for the 43 joints, (1, 43), in
``state.joint_names`` order, every 0.02 s. The environment's joint PD controllers (same for everyone) track them.
"""

from __future__ import annotations

import torch

from isaaclab_arena_g1.challenge.interfaces import Command, LowLevelPolicy, RobotState
from isaaclab_arena_g1.challenge.low_level.wbc import WbcLowLevel


class MyLowLevel(LowLevelPolicy):
    """Starts as the built-in controller; swap in your own parts."""

    def setup(self, env) -> None:
        # Load your trained network here, e.g. torch.jit.load(...). This example reuses the built-in controller.
        self.builtin = WbcLowLevel()
        self.builtin.setup(env)
        self.leg_joint_ids = None

    def reset(self) -> None:
        self.builtin.reset()

    def compute_joint_targets(self, command: Command, state: RobotState) -> torch.Tensor:
        targets = self.builtin.compute_joint_targets(command, state)
        # Example of replacing one part: overwrite the leg joints with your walking policy's output.
        # if self.leg_joint_ids is None:
        #     self.leg_joint_ids = [i for i, name in enumerate(state.joint_names) if "hip" in name or "knee" in name
        #                           or "ankle" in name]
        # targets[:, self.leg_joint_ids] = my_walking_policy(command, state)
        return targets
