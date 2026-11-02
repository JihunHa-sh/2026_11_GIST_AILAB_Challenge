# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Example team high level: starts as the baseline, which completes the task; change it part by part.

Input: anything read with ``api`` (object, goal, robot, furniture; read only) and the observation.
Output: a ``Command`` every 0.02 s: walk velocity, pelvis height, torso angles, both wrist targets (pelvis frame),
both grips. ``Command.from_tensor(api.make_action(env, ...))`` builds one with defaults for what you leave out.
"""

from __future__ import annotations

from isaaclab_arena_g1.challenge import api  # noqa: F401  (used in the commented examples)
from isaaclab_arena_g1.challenge.high_level.baseline import BaselineHighLevel
from isaaclab_arena_g1.challenge.interfaces import Command, HighLevelPolicy


class MyHighLevel(HighLevelPolicy):
    """Runs the baseline high level; replace its command with your own planning where you like."""

    def __init__(self, **options):
        # Team file options arrive here as keyword arguments; the baseline takes the fields of
        # G1FastTransportBaselinePolicyCfg (e.g. max_forward_vel, place_clearance).
        super().__init__(**options)
        self.baseline = BaselineHighLevel(**options)

    @property
    def phase(self):
        """Printed by the runner when it changes; here the baseline's state machine phase."""
        return self.baseline.phase

    def setup(self, env) -> None:
        # Once, before the first episode: load models, build planners, read the object's size, ...
        self.baseline.setup(env)

    def reset(self) -> None:
        # Every episode start: clear per-episode state.
        self.baseline.reset()

    def compute_command(self, env, observation) -> Command:
        command = self.baseline.compute_command(env, observation)
        # Example of changing one part: walk faster while carrying.
        # if self.phase is not None and self.phase.value == "carry":
        #     command.walk_forward = command.walk_forward * 1.5
        # Or build the whole command yourself:
        # command = Command.from_tensor(api.make_action(env, walk_forward=0.3))
        return command
