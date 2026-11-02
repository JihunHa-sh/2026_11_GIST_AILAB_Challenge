# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv

    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


# The G1 embodiments driven by the whole-body controller, which walk on the kitchen floor.
G1_WBC_EMBODIMENTS = ("g1_wbc_pink", "g1_wbc_agile_pink", "g1_wbc_joint", "g1_wbc_agile_joint")

# High-friction contact material for the Dex3 fingers, so a one-hand grasp holds small kitchen objects.
G1_FINGER_FRICTION_MATERIAL_PATH = "/World/Materials/g1_kitchen_high_friction_fingers"
G1_FINGER_PRIM_NAME_MARKERS = ("hand", "thumb", "index", "middle")


POLICY_GAVE_UP_BUFFER = "policy_gave_up_buf"
"""Attribute of the environment a policy sets, per env, to end its episode as a failure."""


def policy_gave_up(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Failure term: true for the environments whose policy set ``env.policy_gave_up_buf``.

    Scripted experts set the flag once their attempt cannot succeed any more (the object tipped over, the grasp
    failed too often), so the episode ends right away instead of running until the time-out.
    """
    import torch

    flags = getattr(env, POLICY_GAVE_UP_BUFFER, None)
    if flags is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    triggered = flags.clone()
    flags[:] = False
    return triggered


def articulation_corners_w(env: ManagerBasedRLEnv, name: str, local_min, local_max) -> torch.Tensor:
    """World corners, shape (num_envs, 8, 3), of an articulation's local bounding box at its current root pose."""
    import itertools
    import torch

    from isaaclab.utils.math import quat_apply

    root_pose = env.scene[name].data.root_pose_w.torch
    corners = torch.tensor(
        list(itertools.product(*zip(local_min, local_max))), dtype=root_pose.dtype, device=root_pose.device
    )
    num_envs = root_pose.shape[0]
    quat = root_pose[:, 3:7].unsqueeze(1).expand(num_envs, 8, 4)
    return quat_apply(quat, corners.unsqueeze(0).expand(num_envs, 8, 3)) + root_pose[:, None, :3]


def object_inside_destination(
    env: ManagerBasedRLEnv,
    object_name: str,
    destination_name: str,
    destination_local_min: tuple[float, float, float],
    destination_local_max: tuple[float, float, float],
    margin: float = 0.02,
    velocity_threshold: float = 0.05,
    hand_clearance: float = 0.15,
) -> torch.Tensor:
    """Predicate: the object's center is inside the destination's bounding box (shrunk by ``margin``), it is nearly
    still and both G1 wrists are at least ``hand_clearance`` away from it, i.e. it was let go. Used for articulated
    containers such as a microwave or toaster oven, where contact-based placement checks do not apply."""
    destination_vertices = articulation_corners_w(env, destination_name, destination_local_min, destination_local_max)
    low = destination_vertices.min(dim=1).values + margin
    high = destination_vertices.max(dim=1).values - margin
    position = env.arena_world.get_position_w(object_name)
    inside = ((position > low) & (position < high)).all(dim=-1)
    speed = env.scene[object_name].data.root_lin_vel_w.torch.norm(dim=-1)
    robot = env.scene["robot"]
    wrist_ids = [robot.body_names.index(f"{side}_wrist_yaw_link") for side in ("left", "right")]
    wrists = robot.data.body_link_pose_w.torch[:, wrist_ids, :3]
    hands_clear = (wrists - position.unsqueeze(1)).norm(dim=-1).min(dim=-1).values > hand_clearance
    return inside & (speed < velocity_threshold) & hands_clear


def hold_joint_open(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_name: str, joint_name: str, position: float | None
) -> None:
    """Reset event: put a joint at ``position`` and make its drive target that position, so it stays there.

    With ``position`` None the joint goes just short of its limit farthest from zero, which is the open end for
    doors closed at zero (microwave doors swing to -90 deg, toaster oven doors fold down to +90 deg).
    """
    articulation = env.scene[asset_name]
    joint_index = articulation.joint_names.index(joint_name)
    if position is None:
        lower, upper = articulation.data.joint_pos_limits.torch[0, joint_index].tolist()
        position = 0.98 * (lower if abs(lower) > abs(upper) else upper)
    joint_pos = articulation.data.joint_pos.torch[env_ids].clone()
    joint_pos[:, joint_index] = position
    joint_vel = articulation.data.joint_vel.torch[env_ids].clone()
    joint_vel[:, joint_index] = 0.0
    articulation.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    targets = articulation.data.joint_pos_target.torch[env_ids].clone()
    targets[:, joint_index] = position
    articulation.set_joint_position_target(targets, env_ids=env_ids)


@dataclass
class G1KitchenPickAndPlaceEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the G1 kitchen pick-and-place environment.

    The defaults use the right-hand counter of the Lightwheel RoboCasa one-wall kitchens, with the object and
    the bowl near its front edge, within reach of a G1 standing at the counter, and the G1 placed further back
    so it walks up to it. The counter sits a few centimeters apart in each style, so x positions are measured
    from the counter's left end and work in all ten one-wall styles. Other layouts need their own counter prim.
    """

    kitchen_background: str = "lightwheel_kitchen_one_wall_farmhouse1"
    counter_prim_path: str = "counter_right_main_group/top_geometry"
    object: str = "ranch_dressing_hope_robolab"
    object_counter_x: float = 0.44
    """Object x measured from the counter's left end [m]."""
    object_y: float = -0.55
    """Object world y [m]; the counter front edge is at y = -0.65."""
    object_yaw_deg: float = -111.55
    object_xy_noise: float = 0.03
    """Half-range of the object's random xy offset at each reset [m]."""
    object_yaw_noise_deg: float = 15.0
    """Half-range of the object's random yaw offset at each reset [deg]."""
    destination: str = "bowl_ycb_robolab"
    destination_on: str = "counter"
    """Where the destination stands: "counter" (next to the object) or "floor" (in the room behind the robot, so it
    turns, walks and squats to place). Floor destinations should be bins or crates about 0.15 m tall or more."""
    destination_counter_x: float | None = 0.14
    """Destination x measured from the counter's left end, or None to let the placement solver pick a spot [m]."""
    destination_y: float | None = -0.51
    """Destination world y on the counter, or None to let the placement solver pick a spot [m]."""
    destination_floor_counter_x: float = -0.2
    """Floor destination x measured from the counter's left end [m]."""
    destination_floor_y: float = -1.3
    """Floor destination world y [m]; the counter front edge is at y = -0.65."""
    destination_yaw_deg: float = 0.0
    """Yaw of the destination about the vertical [deg], e.g. to turn a microwave's door toward the robot."""
    destination_open: bool = False
    """Reset an openable destination (e.g. ``microwave``) open at every episode start."""
    destination_open_joint_pos: float | None = None
    """Joint position [rad] an open destination's door is held at by a position drive, so a passing arm cannot
    swing it shut; None picks just short of the door's limit farthest from zero (its open end)."""
    destination_open_stiffness: float = 30.0
    """Stiffness of the drive holding an open destination's door [N m/rad]."""
    destination_xy_noise: float = 0.02
    """Half-range of the destination's random xy offset at each reset [m]."""
    robot_counter_x: float = 0.41
    """Robot x measured from the counter's left end [m]."""
    robot_y: float = -1.4
    robot_yaw_deg: float = 90.0
    embodiment: str = "g1_wbc_pink"
    arms_down_start: bool = True
    """Spawn the G1 with its arms hanging at its sides instead of the forearms-forward joint zero pose."""
    finger_friction: float | None = 6.0
    """Static friction of the G1 finger contacts (dynamic is 5/6 of it), or None to keep the asset's material."""
    episode_length_s: float = 90.0
    """Episode time limit [s]; destinations off the counter take a walk and a squat, about 55 s for the expert."""
    teleop_device: str | None = None
    task_description: str | None = None


@register_environment
class G1KitchenPickAndPlaceEnvironment(ArenaEnvironmentFactory[G1KitchenPickAndPlaceEnvironmentCfg]):
    """G1 walks up to a kitchen counter and moves an object into a bowl, in a Lightwheel RoboCasa kitchen."""

    name: str = "g1_kitchen_pick_and_place"
    _legacy_argparse_cfg_type = G1KitchenPickAndPlaceEnvironmentCfg

    def build(self, cfg: G1KitchenPickAndPlaceEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from functools import partial

        from isaaclab.managers import TerminationTermCfg

        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz
        from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
        from isaaclab_arena.relations.relations import (
            AtPosition,
            IsAnchor,
            On,
            RandomAroundSolution,
            RotateAroundSolution,
        )
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.tasks.predicates.object_settling import objects_settled
        from isaaclab_arena.tasks.predicates.spatial import object_is_above_height
        from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
        from isaaclab_arena.utils.pose import Pose

        assert (
            cfg.embodiment in G1_WBC_EMBODIMENTS
        ), f"{self.name} only supports G1 WBC embodiments {G1_WBC_EMBODIMENTS}, got '{cfg.embodiment}'."
        # Objects imported on this machine (e.g. from Objaverse) and extra fixtures become available by name.
        from isaaclab_arena_g1.tools.local_objects import register_local_objects

        register_local_objects()
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        if cfg.arms_down_start:
            from isaaclab_arena_g1.policy.g1_scripted_expert_base import G1_ARMS_DOWN_JOINT_POS

            embodiment.set_joint_initial_pos(G1_ARMS_DOWN_JOINT_POS)
        if cfg.finger_friction is not None:
            embodiment.set_finger_contact_friction(
                material_path=G1_FINGER_FRICTION_MATERIAL_PATH,
                static_friction=cfg.finger_friction,
                dynamic_friction=cfg.finger_friction * 5.0 / 6.0,
                prim_name_markers=G1_FINGER_PRIM_NAME_MARKERS,
            )
        kitchen = self.asset_registry.get_asset_by_name(cfg.kitchen_background)()
        counter_top = ObjectReference(
            name="kitchen_counter_top",
            prim_path=f"{{ENV_REGEX_NS}}/{kitchen.name}/{cfg.counter_prim_path}",
            parent_asset=kitchen,
        )
        counter_top.add_relation(IsAnchor())
        # World x of the counter's left end; the bounds are measured from the counter prim's origin.
        counter_left_x = counter_top.get_initial_pose().position_xyz[0] + float(
            counter_top.get_bounding_box().min_point[0, 0]
        )

        # Pelvis at the G1 standing height above the kitchen floor (z = 0), facing the counter.
        half_yaw = math.radians(cfg.robot_yaw_deg) / 2.0
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(counter_left_x + cfg.robot_counter_x, cfg.robot_y, 0.78),
                rotation_xyzw=(0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
            )
        )

        pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)()
        pick_up_object.add_relation(On(counter_top))
        pick_up_object.add_relation(AtPosition(x=counter_left_x + cfg.object_counter_x, y=cfg.object_y))
        pick_up_object.add_relation(RotateAroundSolution(yaw_rad=math.radians(cfg.object_yaw_deg)))
        pick_up_object.add_relation(
            RandomAroundSolution(
                x_half_m=cfg.object_xy_noise,
                y_half_m=cfg.object_xy_noise,
                yaw_half_rad=math.radians(cfg.object_yaw_noise_deg),
            )
        )

        destination = self.asset_registry.get_asset_by_name(cfg.destination)()
        assert cfg.destination_on in (
            "counter",
            "floor",
        ), f"destination_on must be counter or floor, got {cfg.destination_on}"
        extra_assets = []
        if cfg.destination_on == "floor":
            floor = ObjectReference(
                name="kitchen_floor",
                prim_path=f"{{ENV_REGEX_NS}}/{kitchen.name}/floor_room/geometry",
                parent_asset=kitchen,
            )
            floor.add_relation(IsAnchor())
            extra_assets.append(floor)
            destination.add_relation(On(floor))
            destination.add_relation(
                AtPosition(x=counter_left_x + cfg.destination_floor_counter_x, y=cfg.destination_floor_y)
            )
            destination.add_relation(
                RandomAroundSolution(x_half_m=cfg.destination_xy_noise, y_half_m=cfg.destination_xy_noise)
            )
        else:
            destination.add_relation(On(counter_top))
        if cfg.destination_on == "counter" and (cfg.destination_counter_x is not None or cfg.destination_y is not None):
            destination_x = None if cfg.destination_counter_x is None else counter_left_x + cfg.destination_counter_x
            destination.add_relation(AtPosition(x=destination_x, y=cfg.destination_y))
            destination.add_relation(
                RandomAroundSolution(x_half_m=cfg.destination_xy_noise, y_half_m=cfg.destination_xy_noise)
            )

        if cfg.destination_open:
            from isaaclab.actuators import ImplicitActuatorCfg

            # Hold the door open with a position drive: the asset's door joint has none, so it swings freely.
            door_joint = destination.openable_joint_name
            destination.object_cfg.actuators = {
                "open_door_hold": ImplicitActuatorCfg(
                    joint_names_expr=[door_joint],
                    stiffness=cfg.destination_open_stiffness,
                    damping=0.1 * cfg.destination_open_stiffness,
                )
            }
            if cfg.destination_open_joint_pos is not None:
                destination.object_cfg.init_state.joint_pos = {door_joint: cfg.destination_open_joint_pos}

        if cfg.destination_yaw_deg != 0.0:
            destination.add_relation(RotateAroundSolution(yaw_rad=math.radians(cfg.destination_yaw_deg)))

        light = self.asset_registry.get_asset_by_name("light")()

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None

        if cfg.task_description is not None:
            task_description = cfg.task_description
        else:
            object_label = cfg.object.replace("_", " ")
            destination_label = cfg.destination.replace("_", " ")
            task_description = (
                f"Walk to the kitchen counter, pick up the {object_label} and put it in the {destination_label}."
            )

        local_bounds = destination.get_bounding_box()
        destination_local_min = tuple(float(v) for v in local_bounds.min_point[0])
        destination_local_max = tuple(float(v) for v in local_bounds.max_point[0])

        class PickAndPlaceTaskWithGiveUp(PickAndPlaceTask):
            """Pick-and-place whose episode also ends, as a failure, when the policy gives up."""

            def make_scene_cfg(self):
                # Articulated destinations (e.g. a microwave) take no contact sensor; see get_termination_cfg.
                if self.destination_location.object_type == ObjectType.ARTICULATION:
                    return None
                return super().make_scene_cfg()

            def get_termination_cfg(self):
                termination_cfg = super().get_termination_cfg()
                if self.destination_location.object_type == ObjectType.ARTICULATION:
                    # Success once the object rests inside the container, after it was lifted off its support.
                    termination_cfg.success = [
                        CompletionCriteria(
                            name="place_inside",
                            predicate_sequence=[
                                partial(objects_settled, object_names=[self.pick_up_object.name]),
                                partial(
                                    object_is_above_height, object_name=self.pick_up_object.name, use_settled_state=True
                                ),
                                TrueForConsecutiveStepsCfg(
                                    predicate=partial(
                                        object_inside_destination,
                                        object_name=self.pick_up_object.name,
                                        destination_name=self.destination_location.name,
                                        destination_local_min=destination_local_min,
                                        destination_local_max=destination_local_max,
                                    ),
                                    required_steps=self.placement_consecutive_steps,
                                ),
                            ],
                        )
                    ]
                termination_cfg.failures["policy_gave_up"] = TerminationTermCfg(func=policy_gave_up)
                return termination_cfg

            def get_events_cfg(self):
                if not cfg.destination_open:
                    return super().get_events_cfg()
                from isaaclab.managers import EventTermCfg
                from isaaclab.utils import configclass

                from isaaclab_arena.affordances.openable import Openable

                assert isinstance(
                    destination, Openable
                ), f"destination_open needs an openable destination, got {cfg.destination}"

                @configclass
                class OpenDestinationEventCfg:
                    hold_destination_open = EventTermCfg(
                        func=hold_joint_open,
                        mode="reset",
                        params={
                            "asset_name": destination.name,
                            "joint_name": destination.openable_joint_name,
                            "position": cfg.destination_open_joint_pos,
                        },
                    )

                return OpenDestinationEventCfg()

        scene = Scene(assets=[kitchen, counter_top, *extra_assets, pick_up_object, destination, light])
        task = PickAndPlaceTaskWithGiveUp(
            pick_up_object=pick_up_object,
            destination_location=destination,
            background_scene=kitchen,
            episode_length_s=cfg.episode_length_s,
            task_description=task_description,
            force_threshold=0.5,
            velocity_threshold=0.1,
        )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            # 50 Hz control, required by the whole-body controller (WBC).
            env_cfg_callback=set_control_rate_50hz,
        )
