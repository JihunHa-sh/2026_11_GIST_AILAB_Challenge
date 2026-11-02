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
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class G1FastTransportEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the G1 fast-transport challenge.

    The G1 starts behind the right-hand counter of a Lightwheel RoboCasa one-wall kitchen, walks up to it, picks up
    the object, turns around and carries it to a goal pose on a table on the other side of the room. x
    positions are measured from the counter's left end, so the layout works in all ten one-wall styles. By default the
    table stands across the room, so the carry is about 3.3 m long.
    """

    kitchen_background: str = "lightwheel_kitchen_one_wall_farmhouse1"
    counter_prim_path: str = "counter_right_main_group/top_geometry"
    object: str = "bbq_sauce_bottle_hope_robolab"
    object_counter_x: float = 0.44
    """Object x measured from the counter's left end [m]."""
    object_y: float = -0.55
    """Object world y [m]; the counter front edge is at y = -0.65."""
    object_yaw_deg: float = -90.0
    """Object yaw at the start [deg]; -90 turns the bottle's thin side toward the robot's grasping hand."""
    object_xy_noise: float = 0.0
    """Half-range of the object's random xy offset at each reset [m]."""
    object_yaw_noise_deg: float = 0.0
    """Half-range of the object's random yaw offset at each reset [deg]; keep 0, the goal yaw does not follow it."""
    robot_counter_x: float = 0.41
    """Robot x measured from the counter's left end [m]."""
    robot_y: float = -1.4
    """Robot world y [m]."""
    robot_yaw_deg: float = 90.0
    """Robot heading at the start [deg]; 90 faces the counter."""
    robot_start_xy_noise: float = 0.05
    """Half-range of the robot's random start offset in x and y at each reset [m]."""
    table_counter_x: float = -2.3
    """Table center x measured from the counter's left end [m]; the default puts the table across the room, about
    3.3 m from the object."""
    table_y: float = -2.3
    """Table center world y [m]."""
    table_size_x: float = 0.80
    """Table top width along x [m]."""
    table_size_y: float = 0.40
    """Table top depth along y [m]."""
    table_height: float = 0.75
    """Table top height above the floor [m], a desk height; the counter top is at about 0.92 m."""
    goal_offset_x: float = 0.0
    """Goal x relative to the table center [m]."""
    goal_offset_y: float = 0.0
    """Goal y relative to the table center [m]."""
    goal_yaw_deg: float | None = None
    """Goal yaw of the object [deg]. None keeps the object's orientation relative to the robot: the start yaw plus
    180 deg, since the robot turns around to reach the table. Set it to ``object_yaw_deg`` for the same world yaw."""
    position_tolerance: float = 0.03
    """Success radius around the goal position [m]."""
    orientation_tolerance_deg: float = 15.0
    """Success angle around the goal orientation [deg]."""
    hold_time_s: float = 0.5
    """How long the object must stay within the tolerances, nearly still, to succeed [s]."""
    episode_length_s: float = 90.0
    """Time limit in simulated seconds."""
    embodiment: str = "g1_wbc_pink"
    """``g1_wbc_pink`` (built-in low level: walking policy + arm IK) or ``g1_joint_position`` (a team's own low level
    sends all 43 joint targets); challenge_main.py picks it from the team file."""
    arms_down_start: bool = True
    """Spawn the G1 with its arms hanging at its sides instead of the forearms-forward joint zero pose."""
    finger_friction: float | None = 6.0
    """Static friction of the G1 finger contacts (dynamic is 5/6 of it), or None to keep the asset's material."""
    goal_marker_opacity: float = 0.35
    """Opacity of the translucent goal marker, 0 (hidden) to 1 (opaque)."""
    teleop_device: str | None = None
    task_description: str | None = None


@register_environment
class G1FastTransportEnvironment(ArenaEnvironmentFactory[G1FastTransportEnvironmentCfg]):
    """G1 carries an object from a kitchen counter to a goal pose on a table behind it, as fast as possible."""

    name: str = "g1_fast_transport"
    _legacy_argparse_cfg_type = G1FastTransportEnvironmentCfg

    def build(self, cfg: G1FastTransportEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import set_control_rate_50hz
        from isaaclab_arena.relations.relations import (
            AtPosition,
            IsAnchor,
            On,
            RandomAroundSolution,
            RotateAroundSolution,
        )
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments.g1_kitchen_pick_and_place_environment import (
            G1_FINGER_FRICTION_MATERIAL_PATH,
            G1_FINGER_PRIM_NAME_MARKERS,
            G1_WBC_EMBODIMENTS,
        )
        from isaaclab_arena_g1.challenge.fast_transport_task import (
            FastTransportGoal,
            G1FastTransportTask,
            table_usd_path,
            wrap_angle,
            yaw_quat_xyzw,
        )
        from isaaclab_arena_g1.challenge.joint_position_embodiment import (
            G1_JOINT_POSITION_EMBODIMENT,
            register_joint_position_embodiment,
        )
        from isaaclab_arena_g1.tools.local_objects import register_local_objects

        assert cfg.embodiment in G1_WBC_EMBODIMENTS + (G1_JOINT_POSITION_EMBODIMENT,), (
            f"{self.name} supports G1 WBC embodiments {G1_WBC_EMBODIMENTS} and {G1_JOINT_POSITION_EMBODIMENT}, got"
            f" '{cfg.embodiment}'."
        )
        # Teams' own low levels drive all joints directly through this embodiment.
        register_joint_position_embodiment()
        # Objects imported on this machine (e.g. from Objaverse) become available by name.
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
        if cfg.object_xy_noise > 0.0 or cfg.object_yaw_noise_deg > 0.0:
            pick_up_object.add_relation(
                RandomAroundSolution(
                    x_half_m=cfg.object_xy_noise,
                    y_half_m=cfg.object_xy_noise,
                    yaw_half_rad=math.radians(cfg.object_yaw_noise_deg),
                )
            )

        # Kinematic four-legged table standing on the kitchen floor.
        table_x = counter_left_x + cfg.table_counter_x
        table = Object(
            name="transport_table",
            prim_path="{ENV_REGEX_NS}/transport_table",
            object_type=ObjectType.RIGID,
            usd_path=table_usd_path(cfg.table_size_x, cfg.table_size_y, cfg.table_height),
            initial_pose=Pose(position_xyz=(table_x, cfg.table_y, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
        )
        table.add_relation(IsAnchor())

        # Goal: the object upright on the table top, its bottom touching the top.
        object_bottom_z = float(pick_up_object.get_bounding_box().min_point[0, 2])
        if cfg.goal_yaw_deg is None:
            goal_yaw = wrap_angle(math.radians(cfg.object_yaw_deg) + math.pi)
        else:
            goal_yaw = math.radians(cfg.goal_yaw_deg)
        goal = FastTransportGoal(
            position=(
                table_x + cfg.goal_offset_x,
                cfg.table_y + cfg.goal_offset_y,
                cfg.table_height - object_bottom_z,
            ),
            quat_xyzw=yaw_quat_xyzw(goal_yaw),
        )

        light = self.asset_registry.get_asset_by_name("light")()

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None

        if cfg.task_description is not None:
            task_description = cfg.task_description
        else:
            object_label = cfg.object.replace("_", " ")
            task_description = (
                f"Pick up the {object_label} from the kitchen counter and put it on the table behind you, at the"
                " translucent goal, as fast as possible."
            )

        task = G1FastTransportTask(
            pick_up_object=pick_up_object,
            goal=goal,
            episode_length_s=cfg.episode_length_s,
            position_tolerance=cfg.position_tolerance,
            orientation_tolerance_deg=cfg.orientation_tolerance_deg,
            hold_time_s=cfg.hold_time_s,
            robot_start_xy_noise=cfg.robot_start_xy_noise,
            goal_marker_opacity=cfg.goal_marker_opacity,
            # Through the open front of the kitchen, over the counter, the walk and the table.
            viewer_eye=(
                (table_x + counter_left_x + cfg.object_counter_x) / 2.0,
                cfg.table_y - 2.6,
                2.3,
            ),
            viewer_lookat=(
                (table_x + counter_left_x + cfg.object_counter_x) / 2.0,
                (cfg.table_y + cfg.object_y) / 2.0,
                0.7,
            ),
            task_description=task_description,
        )
        scene = Scene(assets=[kitchen, counter_top, pick_up_object, table, light])
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            # 50 Hz control, required by the whole-body controller (WBC).
            env_cfg_callback=set_control_rate_50hz,
        )
