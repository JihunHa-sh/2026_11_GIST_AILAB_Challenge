# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_g1.policy.g1_bimanual_pick_and_place_expert import (
    BimanualPickAndPlaceParams,
    G1BimanualPickAndPlaceExpert,
)
from isaaclab_arena_g1.policy.g1_scripted_expert_policy import G1ScriptedExpertPolicyCfg, G1ScriptedExpertPolicyMixin


@dataclass
class G1ScriptedBimanualPickAndPlacePolicyCfg(G1ScriptedExpertPolicyCfg):
    """Configure the scripted G1 expert that carries a box with both hands."""

    grasp_distance: float = 0.40
    """Horizontal distance from the pelvis to the box center when grasping [m]."""

    squeeze_depth: float = 0.02
    """How far each palm target goes inside the box side, which sets the squeeze force [m]."""

    close_hands: bool = True
    """Whether to close the fingers on the box after squeezing it."""

    lift_height: float = 0.08
    """How high the box is lifted off its support before walking [m]."""

    backoff_distance: float = 0.3
    """How far the robot walks backward after lifting, to clear the shelf before turning [m]."""

    place_wrist_x: float = 0.27
    """Forward wrist position, in the pelvis frame, at which the box is lowered onto the destination [m]."""

    place_height: float = 0.62
    """Base height command while placing the box [m]."""


@register_policy
class G1ScriptedBimanualPickAndPlacePolicy(
    G1ScriptedExpertPolicyMixin, PolicyBase[G1ScriptedBimanualPickAndPlacePolicyCfg]
):
    """Scripted expert that walks the G1 to a box, carries it with both hands and puts it on a destination.

    The expert reads privileged object poses from the simulator, so it needs neither training nor a
    teleoperator. It drives the ``g1_wbc_pink`` embodiments with their 23-D action. Run it with
    ``policy_runner.py`` to check its success rate, or with ``record_policy_demos.py`` to record its
    successful episodes as demonstrations.
    """

    name = "g1_scripted_bimanual_pick_and_place"
    expert_cls = G1BimanualPickAndPlaceExpert
    params_cls = BimanualPickAndPlaceParams
