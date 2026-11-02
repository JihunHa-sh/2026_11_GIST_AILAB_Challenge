# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Collision-aware arm motion planning for the G1 with cuRobo, in the pelvis frame.

The legs stay with the whole-body controller; cuRobo only plans the two 7-DoF arms, with the waist and the Dex3
fingers locked. The robot model is built at runtime from the G1 URDF used by the PINK IK: collision spheres are
fitted to the link meshes, and the standing legs are approximated by fixed spheres on the pelvis. A planned joint
trajectory is turned back into wrist poses with cuRobo's forward kinematics, so the result is still a sequence of
wrist targets for ``G1DecoupledWBCPinkAction`` and stays compatible with teleoperated demonstrations.

Requires the optional ``nvidia-curobo`` package and a CUDA device.
"""

from __future__ import annotations

import copy
import numpy as np
import os
import trimesh
import xml.etree.ElementTree as ET
from dataclasses import dataclass

SIDES = ("left", "right")
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
ARM_JOINT_SUFFIXES = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
)
HAND_JOINT_SUFFIXES = (
    "hand_thumb_0_joint",
    "hand_thumb_1_joint",
    "hand_thumb_2_joint",
    "hand_middle_0_joint",
    "hand_middle_1_joint",
    "hand_index_0_joint",
    "hand_index_1_joint",
)
ARM_LINK_SUFFIXES = (
    "shoulder_roll_link",
    "shoulder_yaw_link",
    "elbow_link",
    "wrist_roll_link",
    "wrist_pitch_link",
    "wrist_yaw_link",
    "hand_palm_link",
    "hand_thumb_1_link",
    "hand_thumb_2_link",
    "hand_index_0_link",
    "hand_index_1_link",
    "hand_middle_0_link",
    "hand_middle_1_link",
)
BODY_LINKS = ("torso_link", "pelvis")
SPHERES_PER_LINK = {"torso_link": 16, "pelvis": 6, "elbow_link": 4, "wrist_yaw_link": 3, "hand_palm_link": 6}
"""Number of collision spheres fitted to each link mesh (by name suffix); other links get two."""

LEG_PROXY_LINK = "leg_proxy"
# Standing legs, as fixed spheres below the pelvis (x, y, z, radius) in the pelvis frame [m].
LEG_PROXY_SPHERES = [(0.0, side * 0.085, z, 0.045) for side in (1.0, -1.0) for z in (-0.12, -0.22, -0.32)]
WRIST_LINK = {side: f"{side}_wrist_yaw_link" for side in SIDES}

# Arms hanging with the elbows bent forward, between the attention pose and the working poses [rad].
RETRACT_JOINT_POS = {
    f"{side}_{suffix}": sign * value if suffix.endswith(("roll_joint", "yaw_joint")) else value
    for side, sign in (("left", 1.0), ("right", -1.0))
    for suffix, value in (
        ("shoulder_pitch_joint", -0.2),
        ("shoulder_roll_joint", 0.15),
        ("shoulder_yaw_joint", 0.0),
        ("elbow_joint", 0.8),
        ("wrist_roll_joint", 0.0),
        ("wrist_pitch_joint", 0.0),
        ("wrist_yaw_joint", 0.0),
    )
}


def arm_joint_names(side: str) -> list[str]:
    """Return the seven arm joint names of one side, shoulder to wrist."""
    return [f"{side}_{suffix}" for suffix in ARM_JOINT_SUFFIXES]


@dataclass
class Cuboid:
    """Box obstacle in the pelvis frame."""

    name: str
    center: np.ndarray
    """Center position (xyz) [m]."""

    size: np.ndarray
    """Full extents along the pelvis axes (xyz) [m]."""


def _geometry_mesh(element: ET.Element, asset_dir: str) -> trimesh.Trimesh | None:
    """Return the mesh of a URDF ``visual``/``collision`` element, in the link frame, or None if it has none."""
    geometry = element.find("geometry")
    if geometry is None or len(geometry) == 0:
        return None
    shape = geometry[0]
    if shape.tag == "mesh":
        mesh = trimesh.load(os.path.join(asset_dir, shape.get("filename")), force="mesh")
    elif shape.tag == "box":
        mesh = trimesh.creation.box(extents=[float(v) for v in shape.get("size").split()])
    elif shape.tag == "cylinder":
        mesh = trimesh.creation.cylinder(radius=float(shape.get("radius")), height=float(shape.get("length")))
    elif shape.tag == "sphere":
        mesh = trimesh.creation.icosphere(radius=float(shape.get("radius")))
    else:
        return None
    origin = element.find("origin")
    if origin is not None:
        transform = trimesh.transformations.euler_matrix(*[float(v) for v in origin.get("rpy", "0 0 0").split()])
        transform[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        mesh.apply_transform(transform)
    return mesh


def _link_meshes(urdf_path: str) -> dict[str, trimesh.Trimesh]:
    """Load each link's geometry in the link frame, preferring the visual mesh over collision primitives."""
    root = ET.parse(urdf_path).getroot()
    asset_dir = os.path.dirname(urdf_path)
    meshes = {}
    for link in root.iter("link"):
        candidates = [element for tag in ("visual", "collision") if (element := link.find(tag)) is not None]
        mesh_first = sorted(candidates, key=lambda element: element.find("geometry/mesh") is None)
        for element in mesh_first:
            mesh = _geometry_mesh(element, asset_dir)
            if mesh is not None:
                meshes[link.get("name")] = mesh
                break
    return meshes


def _fit_collision_spheres(urdf_path: str, link_names: list[str]) -> dict[str, list[dict]]:
    """Fit cuRobo collision spheres to the URDF meshes of ``link_names``."""
    from curobo.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh

    meshes = _link_meshes(urdf_path)
    spheres = {}
    for link_name in link_names:
        n_spheres = next((n for suffix, n in SPHERES_PER_LINK.items() if link_name.endswith(suffix)), 2)
        points, radii = fit_spheres_to_mesh(
            meshes[link_name],
            n_spheres,
            surface_sphere_radius=0.005,
            fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        )
        spheres[link_name] = [
            {"center": [float(v) for v in point], "radius": float(radius)} for point, radius in zip(points, radii)
        ]
    spheres[LEG_PROXY_LINK] = [{"center": [x, y, z], "radius": r} for x, y, z, r in LEG_PROXY_SPHERES]
    return spheres


def build_g1_curobo_robot_cfg(urdf_path: str) -> dict:
    """Build a cuRobo robot configuration of the G1 upper body, rooted at the pelvis.

    Args:
        urdf_path: Local path of ``g1_29dof_with_hand.urdf``; its meshes must sit next to it.

    Returns:
        The ``robot_cfg`` dictionary cuRobo's ``RobotConfig.from_dict`` expects.
    """
    arm_links = [f"{side}_{suffix}" for side in SIDES for suffix in ARM_LINK_SUFFIXES]
    collision_links = list(BODY_LINKS) + arm_links
    hand_joints = [f"{side}_{suffix}" for side in SIDES for suffix in HAND_JOINT_SUFFIXES]
    arm_joints = arm_joint_names("left") + arm_joint_names("right")
    joint_names = list(WAIST_JOINTS) + arm_joints + hand_joints

    # Links next to each other in the chain overlap by construction; never check them against each other.
    self_collision_ignore = {"pelvis": ["torso_link", LEG_PROXY_LINK], "torso_link": [LEG_PROXY_LINK]}
    for side in SIDES:
        chain = [f"{side}_{suffix}" for suffix in ARM_LINK_SUFFIXES[:7]]
        self_collision_ignore["torso_link"].append(chain[0])
        for parent, child in zip(chain, chain[1:]):
            self_collision_ignore.setdefault(parent, []).append(child)
        fingers = [f"{side}_{suffix}" for suffix in ARM_LINK_SUFFIXES[7:]]
        for finger in fingers:
            self_collision_ignore.setdefault(f"{side}_hand_palm_link", []).append(finger)
            self_collision_ignore.setdefault(f"{side}_wrist_yaw_link", []).append(finger)
        # With the arms hanging, the thumbs rest against the thighs.
        self_collision_ignore[f"{side}_hand_thumb_1_link"] = [f"{side}_hand_thumb_2_link", LEG_PROXY_LINK]
        self_collision_ignore[f"{side}_hand_thumb_2_link"] = [LEG_PROXY_LINK]
        self_collision_ignore[f"{side}_hand_index_0_link"] = [f"{side}_hand_index_1_link", f"{side}_hand_middle_0_link"]
        self_collision_ignore[f"{side}_hand_middle_0_link"] = [f"{side}_hand_middle_1_link"]

    return {
        "kinematics": {
            "urdf_path": urdf_path,
            "asset_root_path": os.path.dirname(urdf_path),
            "base_link": "pelvis",
            "ee_link": WRIST_LINK["right"],
            "link_names": [WRIST_LINK["left"], WRIST_LINK["right"]],
            "collision_link_names": collision_links + [LEG_PROXY_LINK],
            "collision_spheres": _fit_collision_spheres(urdf_path, collision_links),
            "collision_sphere_buffer": 0.01,
            "self_collision_ignore": self_collision_ignore,
            "self_collision_buffer": {},
            "extra_links": {
                LEG_PROXY_LINK: {
                    "parent_link_name": "pelvis",
                    "link_name": LEG_PROXY_LINK,
                    "fixed_transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    "joint_type": "FIXED",
                    "joint_name": f"{LEG_PROXY_LINK}_joint",
                }
            },
            "lock_joints": {name: 0.0 for name in list(WAIST_JOINTS) + hand_joints},
            "cspace": {
                "joint_names": joint_names,
                # Bias redundant solutions toward the arms-down pose, elbows low, rather than winged out.
                "retract_config": [RETRACT_JOINT_POS.get(name, 0.0) for name in joint_names],
                "null_space_weight": [1.0] * len(joint_names),
                # Shoulder roll swings the arm out sideways; making it costly keeps motions in front of the body.
                "cspace_distance_weight": [3.0 if "shoulder_roll" in name else 1.0 for name in joint_names],
                "max_jerk": 500.0,
                "max_acceleration": 15.0,
            },
        }
    }


class G1ArmMotionPlanner:
    """Plans collision-free motions of both G1 arms to wrist goals, given box obstacles in the pelvis frame."""

    def __init__(self, urdf_path: str, step_dt: float, time_dilation: float = 0.5, device: str = "cuda:0"):
        """Build the cuRobo robot model and motion generator; this takes several seconds.

        Args:
            urdf_path: Local path of the G1 URDF used by the PINK IK.
            step_dt: Control period the returned wrist trajectories are sampled at [s].
            time_dilation: Fraction of the robot's velocity and acceleration limits the trajectories use;
                lower is slower and smoother.
            device: CUDA device cuRobo runs on.
        """
        import torch

        self._torch = torch
        # cuRobo optimizes with autograd, which the simulation loop's inference mode would switch off.
        with torch.inference_mode(False), torch.enable_grad():
            self._build(urdf_path, step_dt, time_dilation, device)

    def _build(self, urdf_path: str, step_dt: float, time_dilation: float, device: str) -> None:
        import torch

        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.types.robot import RobotConfig
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

        import copy

        self.tensor_args = TensorDeviceType(device=torch.device(device))
        self.step_dt = step_dt
        robot_dict = build_g1_curobo_robot_cfg(urdf_path)

        def make_motion_gen(robot_cfg_dict: dict) -> MotionGen:
            robot_cfg = RobotConfig.from_dict(copy.deepcopy(robot_cfg_dict), self.tensor_args)
            # Placeholder world sized for the cuboids planned around (counter, container, object).
            world = WorldConfig.from_dict({
                "cuboid": {
                    f"slot_{i}": {"dims": [0.01, 0.01, 0.01], "pose": [5.0 + i, 5.0, 5.0, 1, 0, 0, 0]}
                    for i in range(4)
                }
            })
            config = MotionGenConfig.load_from_robot_config(
                robot_cfg,
                world,
                self.tensor_args,
                interpolation_dt=step_dt,
                collision_cache={"obb": 8},
                # Boxes only: the mesh checker needs an older Warp than Isaac Sim ships.
                collision_checker_type=CollisionCheckerType.PRIMITIVE,
                use_cuda_graph=False,
            )
            motion_gen = MotionGen(config)
            motion_gen.warmup()
            return motion_gen

        # Both arms together, for moves of both wrists at once.
        self.motion_gen = make_motion_gen(robot_dict)
        # One arm each, with the other arm locked where it is: moving only the arm that needs to move keeps the
        # other one still instead of letting the optimizer wiggle it.
        self._single_arm_dicts = {}
        self._single_arm_motion_gens = {}
        for side in SIDES:
            other = SIDES[1 - SIDES.index(side)]
            side_dict = copy.deepcopy(robot_dict)
            kinematics = side_dict["kinematics"]
            kinematics["ee_link"] = WRIST_LINK[side]
            kinematics["link_names"] = [WRIST_LINK[side]]
            kinematics["lock_joints"].update({name: RETRACT_JOINT_POS[name] for name in arm_joint_names(other)})
            self._single_arm_dicts[side] = side_dict
            self._single_arm_motion_gens[side] = make_motion_gen(side_dict)
        self.last_failure = ""
        """Status of the last failed plan, for diagnostics."""
        self.joint_names = self.motion_gen.kinematics.joint_names
        self.plan_config = MotionGenPlanConfig(
            enable_graph=True, max_attempts=4, time_dilation_factor=time_dilation, enable_finetune_trajopt=True
        )

    def set_obstacles(self, cuboids: list[Cuboid]) -> None:
        """Replace the obstacles, given in the pelvis frame."""
        with self._torch.inference_mode(False), self._torch.enable_grad():
            self._set_obstacles(cuboids)

    def _set_obstacles(self, cuboids: list[Cuboid]) -> None:
        from curobo.geom.types import WorldConfig

        world_dict = {
            "cuboid": {
                box.name: {"dims": [float(v) for v in box.size], "pose": [*map(float, box.center), 1, 0, 0, 0]}
                for box in cuboids
            }
        }
        for motion_gen in (self.motion_gen, *self._single_arm_motion_gens.values()):
            motion_gen.update_world(WorldConfig.from_dict(world_dict))

    def wrist_poses(self, joint_pos: dict[str, float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Forward kinematics: wrist (position, xyzw quaternion) per side in the pelvis frame for named joints."""
        with self._torch.inference_mode(False):
            return self._wrist_poses(joint_pos)

    def _wrist_poses(self, joint_pos: dict[str, float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        q = self._torch.tensor(
            [[joint_pos.get(name, 0.0) for name in self.joint_names]], **self.tensor_args.as_torch_dict()
        )
        state = self.motion_gen.kinematics.get_state(q)
        return {side: self._pose_to_numpy(state.link_poses[WRIST_LINK[side]], 0) for side in SIDES}

    def plan(
        self,
        joint_pos: dict[str, float],
        goals: dict[str, tuple[np.ndarray, np.ndarray]] | list[dict[str, tuple[np.ndarray, np.ndarray]]],
        sides: tuple[str, ...] = SIDES,
    ) -> list[dict[str, tuple[np.ndarray, np.ndarray]]] | None:
        """Plan the arms from a joint configuration to wrist goals, avoiding the obstacles.

        Args:
            joint_pos: Current arm joint positions by name [rad]; missing joints read as zero.
            goals: Wrist (position, xyzw quaternion) goal per side, in the pelvis frame, or a list of them to pass
                through in order.
            sides: Arms to move. With one side, the other arm stays locked where it is (and keeps its goal pose
                in the returned trajectory); with both, the arms are planned together.

        Returns:
            Wrist (position, xyzw quaternion) per side for every control step of the trajectory, or None when no
            collision-free trajectory was found.
        """
        with self._torch.inference_mode(False), self._torch.enable_grad():
            if len(sides) == 1:
                return self._plan_single_arm(joint_pos, goals, sides[0])
            return self._plan(joint_pos, goals)

    def _plan_single_arm(
        self,
        joint_pos: dict[str, float],
        goals: dict[str, tuple[np.ndarray, np.ndarray]] | list[dict[str, tuple[np.ndarray, np.ndarray]]],
        side: str,
    ) -> list[dict[str, tuple[np.ndarray, np.ndarray]]] | None:
        from curobo.types.math import Pose
        from curobo.types.state import JointState

        torch = self._torch
        motion_gen = self._single_arm_motion_gens[side]
        other = SIDES[1 - SIDES.index(side)]
        # Lock the other arm (and the waist and fingers) at its current joint positions.
        lock_joints = dict(self._single_arm_dicts[side]["kinematics"]["lock_joints"])
        lock_joints.update({name: float(joint_pos.get(name, 0.0)) for name in arm_joint_names(other)})
        motion_gen.update_locked_joints(lock_joints, copy.deepcopy(self._single_arm_dicts[side]))
        joint_names = motion_gen.kinematics.joint_names
        start = JointState.from_position(
            torch.tensor([[joint_pos.get(name, 0.0) for name in joint_names]], **self.tensor_args.as_torch_dict()),
            joint_names=joint_names,
        )
        waypoints = goals if isinstance(goals, list) else [goals]
        poses = []
        for waypoint in waypoints:
            x, y, z, w = waypoint[side][1]
            goal = Pose(
                position=torch.tensor([list(waypoint[side][0])], **self.tensor_args.as_torch_dict()),
                quaternion=torch.tensor([[w, x, y, z]], **self.tensor_args.as_torch_dict()),
            )
            result = motion_gen.plan_single(start, goal, self.plan_config.clone())
            if not bool(result.success.item()):
                self.last_failure = f"{side} arm: {result.status}"
                return None
            trajectory = result.get_interpolated_plan()
            state = motion_gen.kinematics.get_state(trajectory.position)
            first = 0 if not poses else 1
            for i in range(first, trajectory.position.shape[0]):
                poses.append({side: self._pose_to_numpy(state.ee_pose, i), other: waypoint[other]})
            start = JointState.from_position(trajectory.position[-1:].clone(), joint_names=joint_names)
        return poses

    def _plan(
        self,
        joint_pos: dict[str, float],
        goals: dict[str, tuple[np.ndarray, np.ndarray]] | list[dict[str, tuple[np.ndarray, np.ndarray]]],
    ) -> list[dict[str, tuple[np.ndarray, np.ndarray]]] | None:
        from curobo.types.math import Pose
        from curobo.types.state import JointState

        torch = self._torch
        waypoints = goals if isinstance(goals, list) else [goals]
        start = JointState.from_position(
            torch.tensor([[joint_pos.get(name, 0.0) for name in self.joint_names]], **self.tensor_args.as_torch_dict()),
            joint_names=self.joint_names,
        )

        def to_pose(position: np.ndarray, quat_xyzw: np.ndarray) -> Pose:
            x, y, z, w = quat_xyzw
            return Pose(
                position=torch.tensor([list(position)], **self.tensor_args.as_torch_dict()),
                quaternion=torch.tensor([[w, x, y, z]], **self.tensor_args.as_torch_dict()),
            )

        poses = []
        for waypoint in waypoints:
            result = self.motion_gen.plan_single(
                start,
                to_pose(*waypoint["right"]),
                self.plan_config.clone(),
                link_poses={WRIST_LINK[side]: to_pose(*waypoint[side]) for side in SIDES},
            )
            if not bool(result.success.item()):
                self.last_failure = str(result.status)
                return None
            trajectory = result.get_interpolated_plan()
            state = self.motion_gen.kinematics.get_state(trajectory.position)
            first = 0 if not poses else 1
            poses.extend(
                {side: self._pose_to_numpy(state.link_poses[WRIST_LINK[side]], i) for side in SIDES}
                for i in range(first, trajectory.position.shape[0])
            )
            start = JointState.from_position(trajectory.position[-1:].clone(), joint_names=self.joint_names)
        return poses

    @staticmethod
    def _pose_to_numpy(pose, index: int) -> tuple[np.ndarray, np.ndarray]:
        position = pose.position[index].detach().cpu().numpy().astype(np.float64)
        w, x, y, z = pose.quaternion[index].detach().cpu().numpy().astype(np.float64)
        return position, np.array([x, y, z, w])


FRONT_VIA_HEIGHT = 0.02
"""Wrist height in the pelvis frame separating the hanging arm from working poses [m]."""

FRONT_VIA_POSE = {
    side: (np.array([0.08, sign * 0.19, 0.05]), np.array([0.0, np.sin(np.radians(35)), 0.0, np.cos(np.radians(35))]))
    for side, sign in (("left", 1.0), ("right", -1.0))
}
"""Wrist pose in front of the hip, fingers tilted 70 degrees down, that rising and lowering arms pass through.
It stays close to the body so the fingers clear a counter the robot stands at."""


def g1_urdf_path() -> str:
    """Download (once) and return the local path of the G1 URDF the PINK IK uses, with its meshes."""
    from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
    from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.utils.g1 import _retrieve_urdf_assets

    urdf_path, _ = _retrieve_urdf_assets(f"{ARENA_NUCLEUS_DIR}/Arena/wbc_policy/robot_model/g1/g1_29dof_with_hand.urdf")
    return urdf_path


def _quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class G1SceneArmPlanner:
    """Plans the arms of a scripted expert around the object, the destination and the support furniture.

    Implements the expert's ``ArmPlanner`` protocol: obstacles come from the bounding boxes in the scene state,
    re-expressed as boxes in the current pelvis frame, and the support is extended down to the floor so that
    the cabinet below a counter top is an obstacle too.
    """

    def __init__(self, planner: G1ArmMotionPlanner, object_margin: float = 0.015, support_margin: float = 0.01):
        """Wrap a motion planner.

        Args:
            planner: cuRobo planner of the two arms.
            object_margin: How much the object and destination boxes are grown on every side [m].
            support_margin: How much the support box is grown on every side [m].
        """
        self.planner = planner
        self.object_margin = object_margin
        self.support_margin = support_margin

    def plan(self, state, goals, ignore, sides=SIDES):
        """Plan both arms to ``goals`` from the commanded arm configuration, avoiding the obstacles not ignored."""
        T_W_P_rotation = _quat_xyzw_to_matrix(state.pelvis_quat_xyzw)
        obstacles = {"object": state.object_vertices, "destination": state.destination_vertices}
        if state.support_vertices is not None:
            support = np.array(state.support_vertices, dtype=np.float64)
            support[support[:, 2] <= support[:, 2].min() + 1e-6, 2] = 0.0
            obstacles["support"] = support
        cuboids = []
        for name, vertices in obstacles.items():
            if name in ignore:
                continue
            local = (np.asarray(vertices) - state.pelvis_pos) @ T_W_P_rotation
            # The simulated arm sags and lags behind its targets by a few centimeters; keep that much margin.
            margin = self.support_margin if name == "support" else self.object_margin
            low, high = local.min(axis=0) - margin, local.max(axis=0) + margin
            cuboids.append(Cuboid(name, 0.5 * (low + high), high - low))
        self.planner.set_obstacles(cuboids)
        return self.planner.plan(state.arm_joint_pos, self._with_front_via(state.arm_joint_pos, goals), sides=sides)

    def _with_front_via(self, joint_pos, goals):
        """Route a wrist moving between the hanging pose and a working pose through a pose in front of the body.

        cuRobo optimizes in joint space, where the shortest way up from the sides swings the arm out sideways;
        passing in front of the hip makes the arm rise forward, as a person's would.
        """
        start = self.planner.wrist_poses(joint_pos)
        via = {}
        for side in SIDES:
            start_low = start[side][0][2] < FRONT_VIA_HEIGHT
            goal_low = goals[side][0][2] < FRONT_VIA_HEIGHT
            via[side] = FRONT_VIA_POSE[side] if start_low != goal_low else goals[side]
        if all(via[side] is goals[side] for side in SIDES):
            return goals
        return [via, goals]
