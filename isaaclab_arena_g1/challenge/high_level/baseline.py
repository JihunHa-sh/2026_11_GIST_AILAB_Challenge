# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The baseline as a high level: the scripted one-hand expert of baseline_policy.py, emitting commands.

Inside, it is a task state machine (approach, grasp, lift, back off, carry, place) with three planners:

- walking: a P controller toward waypoints (the counter, a point in front of the desk, the desk);
  in: pelvis pose and the waypoint; out: walk forward/left velocity and turn rate.
- grasping: a hand pose wrapped around the object's bounding box from the side;
  in: the object's bounding-box corners in the pelvis frame; out: the right wrist grasp pose.
- arm moves: straight minimum-jerk moves between wrist poses, plus wrist corrections of the object's tilt and yaw
  before letting go; in: current and target wrist pose; out: the wrist target of each step.

Options (team file ``high_level.options``) are fields of ``G1FastTransportBaselinePolicyCfg``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_arena_g1.challenge.interfaces import Command, HighLevelPolicy

if TYPE_CHECKING:
    import gymnasium as gym


class BaselineHighLevel(HighLevelPolicy):
    """Scripted baseline high level (privileged object and goal poses)."""

    def __init__(self, **options):
        super().__init__(**options)
        from isaaclab_arena_g1.challenge.baseline_policy import (
            G1FastTransportBaselinePolicy,
            G1FastTransportBaselinePolicyCfg,
        )

        self.policy = G1FastTransportBaselinePolicy(G1FastTransportBaselinePolicyCfg(**options))

    @property
    def phase(self):
        """Current state of the state machine."""
        return self.policy.phase

    def reset(self) -> None:
        self.policy.reset()

    def compute_command(self, env: gym.Env, observation: dict) -> Command:
        return Command.from_tensor(self.policy.get_action(env, observation))
