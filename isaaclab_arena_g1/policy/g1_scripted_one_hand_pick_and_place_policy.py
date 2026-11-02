# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_g1.policy.g1_one_hand_pick_and_place_expert import (
    G1OneHandPickAndPlaceExpert,
    OneHandPickAndPlaceParams,
)
from isaaclab_arena_g1.policy.g1_scripted_expert_policy import G1ScriptedExpertPolicyCfg, G1ScriptedExpertPolicyMixin


@dataclass
class G1ScriptedOneHandPickAndPlacePolicyCfg(G1ScriptedExpertPolicyCfg):
    """Configure the scripted G1 expert that moves a small object into a container with one hand."""

    pick_up_object_name: str = "ranch_dressing_hope_robolab"
    """Scene name of the object to pick up; match the environment's ``--object``."""

    destination_name: str = "bowl_ycb_robolab"
    """Scene name of the container; match the environment's ``--destination``."""

    hand: str = "right"
    """Hand that grasps the object, ``"left"`` or ``"right"``."""

    raised_wrist_height: float = 0.30
    """Wrist height above the pelvis while walking up to the counter, so the hands pass over its edge [m]."""

    grasp_forward: float = 0.30
    """Forward distance from the pelvis to the object center when grasping [m]."""

    grasp_lateral: float = 0.17
    """Sideways distance from the pelvis to the object center, toward the grasping hand [m]."""

    min_grasp_height: float = 0.10
    """Minimum height of the wrist above the object bottom in bounding-box grasps [m]."""

    thumb_clearance: float = 0.10
    """Forward distance from the wrist to the object's near face in bounding-box grasps [m]."""

    idle_arm_down: bool = True
    """Keep the arm that does not grasp hanging at the side; otherwise it is held raised and tucked in."""

    use_motion_planner: bool = True
    """Plan the free-space arm moves around the counter, object and bowl with cuRobo."""

    use_grasp_planner: bool = True
    """Plan antipodal grasps on the object mesh, so objects other than upright bottles can be picked."""

    support_name: str | None = "kitchen_counter_top"
    """Scene name of the counter the object stands on, an obstacle for planned moves."""

    lift_height: float = 0.20
    """How high the object is lifted off its support before carrying it [m]."""


@register_policy
class G1ScriptedOneHandPickAndPlacePolicy(
    G1ScriptedExpertPolicyMixin, PolicyBase[G1ScriptedOneHandPickAndPlacePolicyCfg]
):
    """Scripted expert that walks the G1 to a counter and moves an object into a container with one hand.

    The expert reads privileged object poses from the simulator. It drives the ``g1_wbc_pink`` embodiments
    with their 23-D action, for example in ``g1_kitchen_pick_and_place``. Run it with ``policy_runner.py``
    to check its success rate, or with ``record_policy_demos.py`` to record its successful episodes.
    """

    name = "g1_scripted_one_hand_pick_and_place"
    expert_cls = G1OneHandPickAndPlaceExpert
    params_cls = OneHandPickAndPlaceParams
