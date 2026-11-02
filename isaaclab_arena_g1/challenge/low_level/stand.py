# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smallest possible low level: hold the starting pose and ignore the command. A starting point for your own."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena_g1.challenge.interfaces import Command, LowLevelPolicy, RobotState

if TYPE_CHECKING:
    import gymnasium as gym


class HoldPoseLowLevel(LowLevelPolicy):
    """Hold every joint at its position from the first step of the episode; the robot neither walks nor reaches."""

    def setup(self, env: gym.Env) -> None:
        self.hold: torch.Tensor | None = None

    def reset(self) -> None:
        self.hold = None

    def compute_joint_targets(self, command: Command, state: RobotState) -> torch.Tensor:
        if self.hold is None:
            self.hold = state.joint_pos.clone()
        return self.hold
