# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Starting point for a team's high level: walks up to the counter and stands there (it never picks anything up)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena_g1.challenge import api
from isaaclab_arena_g1.challenge.interfaces import Command, HighLevelPolicy

if TYPE_CHECKING:
    import gymnasium as gym


class TemplateHighLevel(HighLevelPolicy):
    """Walk straight toward the object until ``stop_distance`` from it, then stand with the arms down.

    Options: ``stop_distance`` [m] (default 0.55), ``max_forward_vel`` [m/s] (default 0.4).
    """

    def __init__(self, stop_distance: float = 0.55, max_forward_vel: float = 0.4, **options):
        super().__init__(**options)
        self.stop_distance = stop_distance
        self.max_forward_vel = max_forward_vel
        self.phase = "walk"
        """Shown by the runner when it changes."""

    def reset(self) -> None:
        self.phase = "walk"

    def compute_command(self, env: gym.Env, observation: dict) -> Command:
        # 1. Read: the object position, in the pelvis frame.
        object_pos = api.read_challenge_observation(observation)["object_pos"]
        object_in_pelvis = api.env_to_pelvis_points(env, object_pos)
        # 2. Decide: a P controller on the remaining distance.
        remaining = float(object_in_pelvis[0, 0]) - self.stop_distance
        walk_forward = min(max(1.5 * remaining, 0.1), self.max_forward_vel) if remaining > 0.02 else 0.0
        self.phase = "walk" if walk_forward > 0.0 else "stand"
        # 3. Command: arms down, hands open, walking.
        return Command.from_tensor(api.make_action(env, walk_forward=walk_forward))
