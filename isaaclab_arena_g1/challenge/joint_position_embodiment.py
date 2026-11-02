# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""G1 embodiment whose action is a position target for each of its 43 joints, for challenge teams' own low levels.

The targets go to the same joint PD actuators as the built-in whole-body controller's (``g1_wbc_pink``), so every
low level drives the same motors. Import this module to register the ``g1_joint_position`` embodiment.
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.embodiments.g1.g1 import G1EmbodimentBase, G1WBCJointObservationsCfg
from isaaclab_arena.terms.events import reset_all_articulation_joints
from isaaclab_arena.utils.pose import Pose

G1_JOINT_POSITION_EMBODIMENT = "g1_joint_position"


@configclass
class G1JointPositionActionCfg:
    """Absolute position targets [rad] for all 43 joints, in the articulation's joint order."""

    g1_action: ActionTermCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False
    )


@configclass
class G1JointPositionEventCfg:
    """Reset events: joints back to their initial state; no controller state to reset."""

    reset_all = EventTerm(func=reset_all_articulation_joints, mode="reset")
    apply_high_friction_to_g1_fingers: EventTerm | None = None


class G1JointPositionEmbodiment(G1EmbodimentBase):
    """G1 driven by joint position targets for all joints (no built-in walking or arm controller)."""

    name = G1_JOINT_POSITION_EMBODIMENT

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.action_config = G1JointPositionActionCfg()
        self.observation_config = G1WBCJointObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.observation_config.wbc.concatenate_terms = self.concatenate_observation_terms
        self.event_config = G1JointPositionEventCfg()


def register_joint_position_embodiment() -> None:
    """Register ``g1_joint_position`` once."""
    if not AssetRegistry().is_registered(G1_JOINT_POSITION_EMBODIMENT, ensure_loaded=False):
        register_asset(G1JointPositionEmbodiment)
