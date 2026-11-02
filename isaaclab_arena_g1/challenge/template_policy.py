# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Starting point for a g1_fast_transport challenge policy.

Copy this file to ``isaaclab_arena_g1/challenge/policies/<team>_policy.py``, rename the two classes and ``name``,
and fill in ``get_action``. Run it with::

    python isaaclab_arena_g1/challenge/challenge_runner.py \\
        --policy_type isaaclab_arena_g1.challenge.policies.<team>_policy.<YourPolicyClass> \\
        --team_name <team> --num_seeds 3 g1_fast_transport

The template walks the G1 up to the counter and stands there with its arms down; it never picks anything up, so it
scores 0. Everything a policy can read, and ``make_action`` to build the 23-D action, is in ``api.py``; see API.md.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from dataclasses import dataclass
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena_g1.challenge import api


def split_challenge_observation(observation: GymSpacesDict) -> dict[str, torch.Tensor]:
    """Split ``observation["challenge"]`` into named tensors; same as ``api.read_challenge_observation``."""
    return api.read_challenge_observation(observation)


def rest_action(num_envs: int, device: torch.device | str, pelvis_height: float = api.STANDING_PELVIS_HEIGHT):
    """Actions that keep the G1 standing still, hands open and arms down, shape (num_envs, 23)."""
    action = torch.zeros(num_envs, api.ACTION_DIM, device=device)
    action[:, 2:5] = torch.tensor(api.REST_WRIST_POS["left"], device=device)
    action[:, 5:9] = torch.tensor(api.REST_WRIST_QUAT_XYZW, device=device)
    action[:, 9:12] = torch.tensor(api.REST_WRIST_POS["right"], device=device)
    action[:, 12:16] = torch.tensor(api.REST_WRIST_QUAT_XYZW, device=device)
    action[:, 19] = pelvis_height
    return action


@dataclass
class G1FastTransportTemplatePolicyCfg(PolicyCfg):
    """Configure the template challenge policy.

    Every field becomes a command-line option of challenge_runner.py, e.g. ``--stop_distance 0.5``.
    """

    stop_distance: float = 0.55
    """Distance in front of the object at which the template stops walking [m]."""

    max_forward_vel: float = 0.4
    """Maximum forward walking speed [m/s]."""


@register_policy
class G1FastTransportTemplatePolicy(PolicyBase[G1FastTransportTemplatePolicyCfg]):
    """Template challenge policy: walks up to the object and stands. Replace ``get_action`` with your own."""

    name = "g1_fast_transport_template"
    """Unique policy name; change it in your copy."""

    def __init__(self, config: G1FastTransportTemplatePolicyCfg):
        """Initialize the policy.

        Args:
            config: Typed policy configuration.
        """
        super().__init__(config)
        # Keep per-episode state (phase of your state machine, ...) on self and clear it in reset().

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """Return the next action, (1, 23); called every 0.02 s of simulated time."""
        # 1. Read what you need: from the observation...
        object_pos = api.read_challenge_observation(observation)["object_pos"]
        # ...or straight from the simulator, e.g. api.goal_pose(env), api.wrist_pose_in_pelvis(env, "right").

        # 2. Decide: walk straight toward the object until stop_distance from it (a P controller).
        object_in_pelvis = api.env_to_pelvis_points(env, object_pos)
        remaining = object_in_pelvis[0, 0] - self.config.stop_distance
        walk_forward = float((1.5 * remaining).clamp(0.1, self.config.max_forward_vel)) if remaining > 0.02 else 0.0

        # 3. Build the action: arms down, hands open, walking command.
        return api.make_action(env, walk_forward=walk_forward)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Called at the start of every episode; clear per-episode state here."""
        pass
