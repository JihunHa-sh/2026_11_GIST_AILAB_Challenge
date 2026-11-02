# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The built-in whole-body controller, written as a challenge low level.

It is the same controller as the default low level (a learned walking policy for the legs, Pink inverse kinematics
for the arms, a grip mapping for the hands), but runs here, in the team's low-level slot, and sends all 43 joint
targets through the ``g1_joint_position`` embodiment. Start from it to replace one part, e.g. your own walking
policy for the legs while keeping the arm IK.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena_g1.challenge.interfaces import Command, LowLevelPolicy, RobotState

if TYPE_CHECKING:
    import gymnasium as gym


class WbcLowLevel(LowLevelPolicy):
    """Built-in walking policy + arm IK + hand grip, as a low level that outputs 43 joint position targets."""

    def setup(self, env: gym.Env) -> None:
        """Build the controller on the environment's robot."""
        from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action_cfg import G1DecoupledWBCPinkActionCfg

        cfg = G1DecoupledWBCPinkActionCfg(asset_name="robot", joint_names=[".*"])
        # The controller reads only the robot's own state (joint states, pelvis orientation and velocity).
        self.controller = cfg.class_type(cfg, env.unwrapped)

    def reset(self) -> None:
        """Restart the IK solver, the walking policy's history and the controller flags."""
        controller = self.controller
        controller.upperbody_controller.body_ik_solver.initialize()
        controller.upperbody_controller.in_warmup = True
        controller.get_wbc_policy.lower_body_policy.reset(torch.arange(controller.num_envs, device=controller.device))
        controller._is_navigating = False
        controller._navigation_goal_reached = False
        controller._num_navigation_subgoals_reached = -1

    def compute_joint_targets(self, command: Command, state: RobotState) -> torch.Tensor:
        """Run the walking policy and the arm IK on the command."""
        self.controller.process_actions(command.to_tensor())
        return self.controller.processed_actions.clone()
