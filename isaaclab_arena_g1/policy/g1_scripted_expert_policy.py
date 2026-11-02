# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from dataclasses import dataclass, fields
from enum import Enum
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.policy.policy_base import PolicyCfg
from isaaclab_arena_g1.policy.g1_scripted_expert_base import (
    SceneState,
    ScriptedExpertParams,
    ScriptedLocomanipExpertBase,
)


@dataclass
class G1ScriptedExpertPolicyCfg(PolicyCfg):
    """Configuration fields shared by the scripted G1 experts."""

    pick_up_object_name: str = "brown_box"
    """Scene name of the object to pick up; match the environment's ``--object``."""

    destination_name: str = "blue_sorting_bin"
    """Scene name of the destination; match the environment's ``--destination``."""

    ready_wrist_raise: float = 0.08
    """How far above the home pose the empty hands are carried at the start, to clear furniture edges [m]."""

    stand_height: float = 0.74
    """Base height command while walking [m]; use 0.72 with the AGILE embodiments."""

    max_forward_vel: float = 0.4
    """Maximum forward walking speed [m/s]."""

    randomize_motion: bool = True
    """Vary distances, speeds and pauses between episodes so the demonstrations differ."""

    motion_seed: int = 0
    """Seed of the motion randomization."""

    use_motion_planner: bool = False
    """Plan the free-space arm moves around obstacles with cuRobo (needs ``nvidia-curobo`` and a CUDA device);
    otherwise every arm move is a straight minimum-jerk move."""

    planner_time_dilation: float = 0.5
    """Fraction of the arm velocity and acceleration limits planned moves use; lower is slower and smoother."""

    support_name: str | None = None
    """Scene name of the furniture the object stands on, an obstacle for planned moves, or None."""

    destination_interior_site: str | None = "reg_int"
    """Name of the USD site prim marking a front-opening destination's cavity (Lightwheel fixtures such as the
    microwave define ``reg_int``); when the destination has one, the object is slid in through the opening."""

    use_grasp_planner: bool = False
    """Plan grasps on the object's collision mesh, for experts that support it (the one-hand expert); otherwise
    grasps follow the object's bounding box."""


class G1ScriptedExpertPolicyMixin:
    """Policy logic shared by the scripted G1 experts: build the expert, feed it privileged state, return actions.

    Combine with ``PolicyBase[Cfg]`` (listed after this mixin) and set ``expert_cls`` and ``params_cls``.
    Configuration fields whose names match fields of ``params_cls`` are copied into the expert parameters.
    """

    expert_cls: type[ScriptedLocomanipExpertBase]
    """State machine that computes the actions."""

    params_cls: type[ScriptedExpertParams]
    """Parameters of the state machine."""

    config: G1ScriptedExpertPolicyCfg

    def __init__(self, config: G1ScriptedExpertPolicyCfg):
        """Initialize the policy.

        Args:
            config: Typed policy configuration.
        """
        super().__init__(config)
        param_names = {field.name for field in fields(self.params_cls)}
        self.params = self.params_cls(
            **{field.name: getattr(config, field.name) for field in fields(config) if field.name in param_names}
        )
        self.rng = np.random.default_rng(config.motion_seed) if config.randomize_motion else None
        # Built on the first step, once the environment's control period is known.
        self.expert: ScriptedLocomanipExpertBase | None = None
        self._ended_episode = False
        self._gave_up_flags: torch.Tensor | None = None
        self._local_boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._interior_local: dict[str, np.ndarray | None] = {}

    @property
    def phase(self) -> Enum | None:
        """Current phase of the scripted motion, or None before the first step."""
        return None if self.expert is None else self.expert.phase

    @property
    def gave_up(self) -> bool:
        """Whether the current attempt timed out, so the episode cannot succeed any more."""
        return self.expert is not None and self.expert.failed

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        """Compute the next 23-D action from the privileged scene state."""
        unwrapped = env.unwrapped
        assert unwrapped.num_envs == 1, "The scripted expert drives a single environment, like the WBC Pink action."
        if self.expert is None:
            self.expert = self.expert_cls(self.params, step_dt=unwrapped.step_dt, rng=self.rng)
            if self.config.use_motion_planner:
                from isaaclab_arena_g1.policy.g1_curobo_arm_planner import (
                    G1ArmMotionPlanner,
                    G1SceneArmPlanner,
                    g1_urdf_path,
                )

                self.expert.planner = G1SceneArmPlanner(
                    G1ArmMotionPlanner(
                        g1_urdf_path(), step_dt=unwrapped.step_dt, time_dilation=self.config.planner_time_dilation
                    )
                )

            if self.config.use_grasp_planner:
                from isaaclab_arena.assets.registries import AssetRegistry
                from isaaclab_arena.utils.usd.helpers import extract_trimesh_from_usd_path
                from isaaclab_arena_g1.policy.g1_grasp_planner import G1SceneGraspPlanner
                from isaaclab_arena_g1.tools.local_objects import register_local_objects

                register_local_objects()
                asset_type = AssetRegistry().get_asset_by_name(self.config.pick_up_object_name)
                mesh = extract_trimesh_from_usd_path(asset_type.usd_path, tuple(asset_type.scale))
                self.expert.grasp_planner = G1SceneGraspPlanner(mesh)

        world = unwrapped.arena_world
        pelvis_pose = world.get_body_pose_w("robot", "pelvis")[0].cpu().numpy().astype(np.float64)
        object_pose = None
        if self.config.use_grasp_planner:
            object_data = unwrapped.scene[self.config.pick_up_object_name].data
            object_pose = object_data.root_pose_w.torch[0].cpu().numpy().astype(np.float64)
        state = SceneState(
            pelvis_pos=pelvis_pose[:3],
            pelvis_quat_xyzw=pelvis_pose[3:7],
            object_vertices=world.get_vertices_w(self.config.pick_up_object_name)[0].cpu().numpy().astype(np.float64),
            destination_vertices=self._vertices_w(unwrapped, self.config.destination_name),
            arm_joint_pos=self._commanded_arm_joint_pos(unwrapped),
            destination_interior_vertices=self._interior_w(unwrapped, self.config.destination_name),
            object_pos=object_pose[:3] if object_pose is not None else None,
            object_quat_xyzw=object_pose[3:7] if object_pose is not None else None,
            support_vertices=(
                None
                if self.config.support_name is None
                else world.get_vertices_w(self.config.support_name)[0].cpu().numpy().astype(np.float64)
            ),
        )
        action = self.expert.compute_action(state)
        if self.expert.failed and not self._ended_episode:
            # The attempt cannot succeed any more. Environments with a ``policy_gave_up`` failure term (such as
            # g1_kitchen_pick_and_place) end the episode on this flag; elsewhere the robot stands until time-out.
            print(f"[{self.name}] Giving up: {self.expert.failure_reason}.", flush=True)
            flags = getattr(unwrapped, "policy_gave_up_buf", None)
            if flags is None:
                flags = torch.zeros(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
                unwrapped.policy_gave_up_buf = flags
            flags[:] = True
            self._gave_up_flags = flags
            self._ended_episode = True
        return torch.as_tensor(action, dtype=torch.float32, device=unwrapped.device).unsqueeze(0)

    def _vertices_w(self, unwrapped, name: str) -> np.ndarray:
        """Bounding-box corners of a scene object in the world; articulations (e.g. a microwave), which the arena
        world does not measure, use their asset's local box at their root pose."""
        if name not in unwrapped.scene.articulations:
            return unwrapped.arena_world.get_vertices_w(name)[0].cpu().numpy().astype(np.float64)
        if name not in self._local_boxes:
            from isaaclab_arena.assets.registries import AssetRegistry

            box = AssetRegistry().get_asset_by_name(name)().get_bounding_box()
            self._local_boxes[name] = (box.min_point[0].cpu().numpy(), box.max_point[0].cpu().numpy())
        import itertools

        low, high = self._local_boxes[name]
        corners = np.array(list(itertools.product(*zip(low, high))), dtype=np.float64)
        root = unwrapped.scene[name].data.root_pose_w.torch[0].cpu().numpy().astype(np.float64)
        from isaaclab_arena_g1.policy.g1_scripted_expert_base import quat_xyzw_to_matrix

        return corners @ quat_xyzw_to_matrix(root[3:7]).T + root[:3]

    def _interior_w(self, unwrapped, name: str) -> np.ndarray | None:
        """World corners of an articulated destination's cavity site, or None when it has none."""
        site = self.config.destination_interior_site
        if site is None or name not in unwrapped.scene.articulations:
            return None
        if name not in self._interior_local:
            self._interior_local[name] = self._find_site_corners_in_root(unwrapped, name, site)
        corners = self._interior_local[name]
        if corners is None:
            return None
        from isaaclab_arena_g1.policy.g1_scripted_expert_base import quat_xyzw_to_matrix

        root = unwrapped.scene[name].data.root_pose_w.torch[0].cpu().numpy().astype(np.float64)
        return corners @ quat_xyzw_to_matrix(root[3:7]).T + root[:3]

    @staticmethod
    def _find_site_corners_in_root(unwrapped, name: str, site: str) -> np.ndarray | None:
        """Corners of the site's unit box (scaled by its transform) in the destination's root frame, read from USD.

        The site is fixed to the root, so its USD transform relative to the root stays valid when the physics moves
        the destination.
        """
        import itertools

        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        import re

        # The configured path is a regex over environments (env_.*, env_[^/]+); take environment 0.
        root_path = unwrapped.scene[name].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        root_path = re.sub(r"env_(\.\*|\[\^/\]\+)", "env_0", root_path)
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim:
            print(
                f"[G1ScriptedExpertPolicy] Destination prim {root_path} not found; the destination is filled from"
                " above."
            )
            return None
        # Lightwheel fixtures are instanceable; their sites are only reachable through instance proxies.
        prims = Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies())
        site_prim = next((prim for prim in prims if prim.GetName() == site), None)
        if site_prim is None:
            print(f"[G1ScriptedExpertPolicy] No '{site}' site under {root_path}; the destination is filled from above.")
        if site_prim is None:
            return None
        cache = UsdGeom.XformCache()
        T_W_root = cache.GetLocalToWorldTransform(root_prim)
        T_W_site = cache.GetLocalToWorldTransform(site_prim)
        T_root_site = T_W_site * T_W_root.GetInverse()  # Row-vector convention: site -> world -> root.
        corners = []
        for unit in itertools.product((-1.0, 1.0), repeat=3):
            point = T_root_site.Transform(Gf.Vec3d(*unit))
            corners.append([point[0], point[1], point[2]])
        return np.array(corners, dtype=np.float64)

    def _commanded_arm_joint_pos(self, unwrapped) -> dict[str, float] | None:
        """Joint positions of the last IK solution, which realize the commanded wrist poses; None before the first."""
        if not self.config.use_motion_planner:
            return None
        action_term = unwrapped.action_manager.get_term("g1_action")
        ik_joints = getattr(action_term, "_last_ik_target_robot_joints", None)
        if ik_joints is None:
            return None
        return dict(zip(action_term.robot_model.joint_names, np.asarray(ik_joints, dtype=np.float64).tolist()))

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Restart the scripted motion for a new episode."""
        self._ended_episode = False
        if self._gave_up_flags is not None:
            # A caller that resets the environment itself (record_policy_demos) must not carry the flag over.
            self._gave_up_flags[:] = False
        if self.expert is not None:
            self.expert.reset()
