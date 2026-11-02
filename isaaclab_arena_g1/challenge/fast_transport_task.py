# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task of the G1 fast-transport challenge: carry an object from a kitchen counter to a goal pose on a table.

The goal is shown by a translucent, collision-free copy of the object. An episode succeeds once the object rests
within a position and an orientation tolerance of the goal for a while, and fails when the robot falls, the object
drops toward the floor or the (simulated) time runs out.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena_environments.g1_kitchen_pick_and_place_environment import policy_gave_up

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv


FAST_TRANSPORT_STATE_ATTR = "fast_transport_state"
"""Attribute of the environment holding the per-env goal pose and the last measured errors."""

START_OFFSET_SEED_ATTR = "fast_transport_start_seed"
"""Attribute of the environment a runner sets to an int so the next reset draws the robot's start offset from that
seed; when unset, the offset comes from torch's global random generator."""

PELVIS_BODY_NAME = "pelvis"


@dataclass
class FastTransportGoal:
    """Goal pose of the object in the environment frame (relative to the env origin)."""

    position: tuple[float, float, float]
    """Goal position of the object's root [m]."""

    quat_xyzw: tuple[float, float, float, float]
    """Goal orientation of the object's root."""


class FastTransportState:
    """Per-environment goal and error buffers, created on first use and stored on the environment."""

    def __init__(self, env: ManagerBasedRLEnv, goal: FastTransportGoal):
        import torch

        self.goal_pos_e = torch.tensor(goal.position, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
        """Goal position per environment, in the environment frame."""
        self.goal_quat = torch.tensor(goal.quat_xyzw, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)
        """Goal orientation (x, y, z, w) per environment."""
        self.position_error = torch.full((env.num_envs,), float("nan"), device=env.device)
        """Distance from the object to the goal at the last termination check [m]."""
        self.orientation_error = torch.full((env.num_envs,), float("nan"), device=env.device)
        """Angle between the object's and the goal's orientation at the last termination check [rad]."""
        self.start_offset = torch.zeros(env.num_envs, 2, device=env.device)
        """Robot start offset (x, y) drawn at the last reset [m]."""
        self.object_name: str | None = None
        """Scene name of the object to carry, recorded by the success predicate."""

    @staticmethod
    def get(env: ManagerBasedRLEnv, goal: FastTransportGoal) -> FastTransportState:
        """Return the state stored on ``env``, creating it on the first call."""
        state = getattr(env, FAST_TRANSPORT_STATE_ATTR, None)
        if state is None:
            state = FastTransportState(env, goal)
            setattr(env, FAST_TRANSPORT_STATE_ATTR, state)
        return state


# ---------------------------------------------------------------------------------------------------------------------
# Measurements, predicates and failure terms.
# ---------------------------------------------------------------------------------------------------------------------


def object_goal_errors(env: ManagerBasedRLEnv, object_name: str, goal: FastTransportGoal) -> tuple:
    """Return the object's position error [m] and orientation error [rad] to the goal, per environment."""
    from isaaclab.utils.math import quat_error_magnitude

    state = FastTransportState.get(env, goal)
    pose_w = env.scene[object_name].data.root_pose_w.torch
    position_e = pose_w[:, :3] - env.scene.env_origins
    position_error = (position_e - state.goal_pos_e).norm(dim=-1)
    orientation_error = quat_error_magnitude(pose_w[:, 3:7], state.goal_quat)
    return position_error, orientation_error


def object_at_goal(
    env: ManagerBasedRLEnv,
    object_name: str,
    goal: FastTransportGoal,
    position_tolerance: float,
    orientation_tolerance_rad: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Predicate: the object is within the position and orientation tolerances of the goal and nearly still.

    Also records the errors on the environment, so a runner can report the final error of each episode.
    """
    position_error, orientation_error = object_goal_errors(env, object_name, goal)
    state = FastTransportState.get(env, goal)
    state.object_name = object_name
    state.position_error[:] = position_error
    state.orientation_error[:] = orientation_error
    speed = env.scene[object_name].data.root_lin_vel_w.torch.norm(dim=-1)
    return (
        (position_error < position_tolerance)
        & (orientation_error < orientation_tolerance_rad)
        & (speed < velocity_threshold)
    )


def robot_fell(env: ManagerBasedRLEnv, minimum_pelvis_height: float, maximum_tilt_rad: float) -> torch.Tensor:
    """Failure term: the pelvis dropped below ``minimum_pelvis_height`` or tilted more than ``maximum_tilt_rad``."""
    import torch

    robot = env.scene["robot"]
    pelvis_id = robot.body_names.index(PELVIS_BODY_NAME)
    pelvis_pose = robot.data.body_link_pose_w.torch[:, pelvis_id]
    from isaaclab.utils.math import quat_apply

    up = torch.zeros_like(pelvis_pose[:, :3])
    up[:, 2] = 1.0
    pelvis_up = quat_apply(pelvis_pose[:, 3:7], up)
    tilt = torch.acos(pelvis_up[:, 2].clamp(-1.0, 1.0))
    return (pelvis_pose[:, 2] < minimum_pelvis_height) | (tilt > maximum_tilt_rad)


def object_dropped(env: ManagerBasedRLEnv, object_name: str, minimum_height: float) -> torch.Tensor:
    """Failure term: the object's root fell below ``minimum_height`` above the floor (world z)."""
    return env.scene[object_name].data.root_pose_w.torch[:, 2] < minimum_height


# ---------------------------------------------------------------------------------------------------------------------
# Observations, all in the environment frame (positions relative to the env origin, quaternions x, y, z, w).
# ---------------------------------------------------------------------------------------------------------------------


def object_pose_e(env: ManagerBasedRLEnv, object_name: str) -> torch.Tensor:
    """Object root pose (x, y, z, qx, qy, qz, qw) in the environment frame, shape (num_envs, 7)."""
    import torch

    pose_w = env.scene[object_name].data.root_pose_w.torch
    return torch.cat([pose_w[:, :3] - env.scene.env_origins, pose_w[:, 3:7]], dim=-1)


def goal_pose_e(env: ManagerBasedRLEnv, goal: FastTransportGoal) -> torch.Tensor:
    """Goal pose (x, y, z, qx, qy, qz, qw) of the object in the environment frame, shape (num_envs, 7)."""
    import torch

    state = FastTransportState.get(env, goal)
    return torch.cat([state.goal_pos_e, state.goal_quat], dim=-1)


def pelvis_pose_e(env: ManagerBasedRLEnv) -> torch.Tensor:
    """G1 pelvis pose (x, y, z, qx, qy, qz, qw) in the environment frame, shape (num_envs, 7)."""
    import torch

    robot = env.scene["robot"]
    pelvis_id = robot.body_names.index(PELVIS_BODY_NAME)
    pose_w = robot.data.body_link_pose_w.torch[:, pelvis_id]
    return torch.cat([pose_w[:, :3] - env.scene.env_origins, pose_w[:, 3:7]], dim=-1)


def elapsed_sim_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Simulated time since the episode started [s], shape (num_envs, 1)."""
    return (env.episode_length_buf.float() * env.step_dt).unsqueeze(-1)


# ---------------------------------------------------------------------------------------------------------------------
# Reset events.
# ---------------------------------------------------------------------------------------------------------------------


def randomize_robot_start(env: ManagerBasedRLEnv, env_ids: torch.Tensor, xy_half_range: float) -> None:
    """Reset event: shift the G1's root by a uniform random (x, y) offset of at most ``xy_half_range``.

    Runs after the embodiment's own pose reset, so the shift is applied to the configured start pose. The heading
    is unchanged. When the environment carries ``fast_transport_start_seed`` (an int), the offsets come from a
    generator seeded with it, so a runner reproduces a start pose from its seed.
    """
    import torch

    if env_ids is None or len(env_ids) == 0 or xy_half_range <= 0.0:
        return
    seed = getattr(env, START_OFFSET_SEED_ATTR, None)
    if seed is None:
        unit = torch.rand(len(env_ids), 2, device=env.device)
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        unit = torch.rand(len(env_ids), 2, generator=generator).to(env.device)
    offset = (2.0 * unit - 1.0) * xy_half_range
    robot = env.scene["robot"]
    root_pose = robot.data.root_pose_w.torch[env_ids].clone()
    root_pose[:, :2] += offset
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    state = getattr(env, FAST_TRANSPORT_STATE_ATTR, None)
    if state is not None:
        state.start_offset[env_ids] = offset


def show_goal_marker(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    goal: FastTransportGoal,
    usd_path: str,
    scale: tuple[float, float, float],
    color: tuple[float, float, float],
    opacity: float,
) -> None:
    """Reset event: show the goal as a translucent, collision-free copy of the object at the goal pose.

    The marker is created on the first call and kept on the environment; it has no physics, so it never touches
    the robot or the object.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    state = FastTransportState.get(env, goal)
    marker = getattr(env, "_fast_transport_goal_marker", None)
    if marker is None:
        _enable_translucency()
        marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/FastTransportGoal",
                markers={
                    "goal": sim_utils.UsdFileCfg(
                        usd_path=usd_path,
                        scale=scale,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=color, emissive_color=color, opacity=opacity, roughness=0.3
                        ),
                    )
                },
            )
        )
        env._fast_transport_goal_marker = marker
    translations = state.goal_pos_e + env.scene.env_origins
    marker.visualize(translations=translations, orientations=state.goal_quat)


def _enable_translucency() -> None:
    """Let the RTX renderer draw partially transparent preview-surface materials (the goal marker)."""
    try:
        import carb

        settings = carb.settings.get_settings()
        settings.set("/rtx/raytracing/fractionalCutoutOpacity", True)
    except Exception as error:  # Rendering settings are cosmetic; never fail an episode over them.
        print(f"[g1_fast_transport] Could not enable translucent materials: {error}")


# ---------------------------------------------------------------------------------------------------------------------
# Table.
# ---------------------------------------------------------------------------------------------------------------------


def table_usd_path(size_x: float, size_y: float, height: float) -> str:
    """Return the path of a kinematic four-legged table USD of the given top size and height, writing it if needed.

    The table's origin is on the floor under the center of its top; its collision covers the top and the legs, so
    feet fit under the top's edge.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    directory = Path.home() / ".cache" / "isaaclab_arena" / "challenge"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"fast_transport_table_{size_x:.3f}x{size_y:.3f}x{height:.3f}_v1.usda"
    if path.exists():
        return str(path)

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Table")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)

    material = UsdShade.Material.Define(stage, "/Table/Looks/Wood")
    shader = UsdShade.Shader.Define(stage, "/Table/Looks/Wood/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.55, 0.36, 0.20))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    top_thickness = 0.04
    leg_width = 0.05
    leg_inset = 0.04
    parts = {"top": ((0.0, 0.0, height - top_thickness / 2.0), (size_x, size_y, top_thickness))}
    leg_height = height - top_thickness
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        x = sx * (size_x / 2.0 - leg_inset - leg_width / 2.0)
        y = sy * (size_y / 2.0 - leg_inset - leg_width / 2.0)
        parts[f"leg_{i}"] = ((x, y, leg_height / 2.0), (leg_width, leg_width, leg_height))
    for name, (center, size) in parts.items():
        cube = UsdGeom.Cube.Define(stage, f"/Table/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.XformCommonAPI(cube)
        xform.SetTranslate(Gf.Vec3d(*center))
        xform.SetScale(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    stage.GetRootLayer().Export(str(path))
    return str(path)


# ---------------------------------------------------------------------------------------------------------------------
# Task.
# ---------------------------------------------------------------------------------------------------------------------


class G1FastTransportTask(TaskBase):
    """Carry ``pick_up_object`` to ``goal`` (on the table) and leave it there, as fast as possible."""

    def __init__(
        self,
        pick_up_object: Asset,
        goal: FastTransportGoal,
        episode_length_s: float,
        position_tolerance: float = 0.03,
        orientation_tolerance_deg: float = 15.0,
        hold_time_s: float = 0.5,
        control_dt: float = 0.02,
        velocity_threshold: float = 0.05,
        robot_start_xy_noise: float = 0.05,
        minimum_pelvis_height: float = 0.40,
        maximum_pelvis_tilt_deg: float = 60.0,
        object_drop_height: float = 0.50,
        goal_marker_color: tuple[float, float, float] = (0.1, 0.9, 0.3),
        goal_marker_opacity: float = 0.35,
        viewer_eye: tuple[float, float, float] | None = None,
        viewer_lookat: tuple[float, float, float] | None = None,
        task_description: str | None = None,
    ):
        """Create the task.

        Args:
            pick_up_object: The object to carry.
            goal: Goal pose of the object in the environment frame.
            episode_length_s: Time limit in simulated seconds.
            position_tolerance: Success radius around the goal position [m].
            orientation_tolerance_deg: Success angle around the goal orientation [deg].
            hold_time_s: How long the object must stay within the tolerances, nearly still [s].
            control_dt: Control period, used to turn ``hold_time_s`` into steps [s].
            velocity_threshold: Object speed below which it counts as still [m/s].
            robot_start_xy_noise: Half-range of the robot's random start offset in x and y [m].
            minimum_pelvis_height: Pelvis height below which the robot counts as fallen [m].
            maximum_pelvis_tilt_deg: Pelvis tilt beyond which the robot counts as fallen [deg].
            object_drop_height: Object height (world z) below which it counts as dropped [m].
            goal_marker_color: RGB color of the translucent goal marker.
            goal_marker_opacity: Opacity of the goal marker, 0 (invisible) to 1 (opaque).
            viewer_eye: Viewport camera position, or None for the default.
            viewer_lookat: Viewport camera target, or None for the default.
            task_description: Language description of the task.
        """
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.pick_up_object = pick_up_object
        self.goal = goal
        self.position_tolerance = position_tolerance
        self.orientation_tolerance_rad = math.radians(orientation_tolerance_deg)
        self.hold_steps = max(1, round(hold_time_s / control_dt))
        self.velocity_threshold = velocity_threshold
        self.robot_start_xy_noise = robot_start_xy_noise
        self.minimum_pelvis_height = minimum_pelvis_height
        self.maximum_pelvis_tilt_rad = math.radians(maximum_pelvis_tilt_deg)
        self.object_drop_height = object_drop_height
        self.goal_marker_color = goal_marker_color
        self.goal_marker_opacity = goal_marker_opacity
        self.viewer_eye = viewer_eye
        self.viewer_lookat = viewer_lookat

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        at_goal = partial(
            object_at_goal,
            object_name=self.pick_up_object.name,
            goal=self.goal,
            position_tolerance=self.position_tolerance,
            orientation_tolerance_rad=self.orientation_tolerance_rad,
            velocity_threshold=self.velocity_threshold,
        )
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                CompletionCriteria(
                    name="object_at_goal",
                    predicate_sequence=[TrueForConsecutiveStepsCfg(predicate=at_goal, required_steps=self.hold_steps)],
                )
            ],
            failures={
                "robot_fell": TerminationTermCfg(
                    func=robot_fell,
                    params={
                        "minimum_pelvis_height": self.minimum_pelvis_height,
                        "maximum_tilt_rad": self.maximum_pelvis_tilt_rad,
                    },
                ),
                "object_dropped": TerminationTermCfg(
                    func=object_dropped,
                    params={"object_name": self.pick_up_object.name, "minimum_height": self.object_drop_height},
                ),
                # A policy may end its own episode early, as a failure, by setting env.policy_gave_up_buf.
                "policy_gave_up": TerminationTermCfg(func=policy_gave_up),
            },
        )

    def get_events_cfg(self):
        usd_path = self.pick_up_object.usd_path
        scale = tuple(getattr(self.pick_up_object, "scale", (1.0, 1.0, 1.0)))

        @configclass
        class FastTransportEventsCfg:
            randomize_robot_start = EventTermCfg(
                func=randomize_robot_start,
                mode="reset",
                params={"xy_half_range": self.robot_start_xy_noise},
            )
            show_goal = EventTermCfg(
                func=show_goal_marker,
                mode="reset",
                params={
                    "goal": self.goal,
                    "usd_path": usd_path,
                    "scale": scale,
                    "color": self.goal_marker_color,
                    "opacity": self.goal_marker_opacity,
                },
            )

        return FastTransportEventsCfg()

    def get_observation_cfg(self):
        object_name = self.pick_up_object.name
        goal = self.goal

        @configclass
        class ChallengeObsCfg(ObservationGroupCfg):
            """Privileged task state for challenge policies, concatenated into a (num_envs, 22) tensor."""

            object_pose = ObservationTermCfg(func=object_pose_e, params={"object_name": object_name})
            goal_pose = ObservationTermCfg(func=goal_pose_e, params={"goal": goal})
            pelvis_pose = ObservationTermCfg(func=pelvis_pose_e)
            elapsed_time = ObservationTermCfg(func=elapsed_sim_time)

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        @configclass
        class FastTransportObservationsCfg:
            challenge: ChallengeObsCfg = ChallengeObsCfg()

        return FastTransportObservationsCfg()

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("The fast-transport challenge has no Mimic configuration.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.pick_up_object)]

    def get_viewer_cfg(self) -> ViewerCfg:
        if self.viewer_eye is None or self.viewer_lookat is None:
            return ViewerCfg()
        return ViewerCfg(eye=self.viewer_eye, lookat=self.viewer_lookat)


def yaw_quat_xyzw(yaw_rad: float) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) of a rotation by ``yaw_rad`` about the vertical."""
    return (0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0))


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
